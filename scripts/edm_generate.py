# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import pickle
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import PIL.Image
import torch
import torch.distributed
import torch.nn as nn
import tqdm
from omegaconf import DictConfig, OmegaConf

import dnnlib
from torch_utils import distributed as dist


def make_image_np(images: torch.Tensor) -> np.ndarray:
    return (
        (images * 127.5 + 128)
        .clip(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def save_images(images: torch.Tensor, seeds: torch.Tensor, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    images_np = make_image_np(images)
    for seed, image_np in zip(seeds, images_np):
        image_path = os.path.join(save_dir, f"{seed:06d}.png")
        if image_np.shape[2] == 1:
            warnings.warn(
                "Saving a single channel image. This is not recommended for most use cases."
            )
            PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
        else:
            PIL.Image.fromarray(image_np, "RGB").save(image_path)


class StackedRandomGenerator:
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.seeds = seeds
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


def edm_sampler(
    net: nn.Module,
    rnd: StackedRandomGenerator,
    latents: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: int = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    snapshot_interval: int = 1,
    snapshot_save_dir: Optional[str] = None,
    n_images_save: int = 16,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[float, Dict[str, Any]]]:
    """
    Proposed EDM sampler (Algorithm 2 in EDM paper)
    """
    if net.__class__.__name__ != "EDMPrecond":
        raise ValueError(
            "Network must be an instance of EDMPrecond from EDM work (Karras et al)"
        )

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    print(f"sigma_min: {sigma_min}, sigma_max: {sigma_max}")
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0
    if dist.get_rank() == 0:
        if verbose:
            print(f"t_steps: {t_steps}")
        np.save("t_steps.npy", t_steps.detach().cpu().numpy())

    # Initialize lists to store norms and intermediate images
    snapshot_dict: Dict[float, Dict[str, Any]] = {}

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * rnd.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Save norms and intermediate images at specified intervals
        if i % snapshot_interval == 0 or i == num_steps - 1:
            snapshot_dict[t_cur.item()] = {
                "norms": torch.norm(x_cur, p=2, dim=(1, 2, 3)).detach().cpu().numpy(),
                "norms_denoised": torch.norm(denoised, p=2, dim=(1, 2, 3))
                .detach()
                .cpu()
                .numpy(),
                "norms_dxdt": torch.norm(d_cur, p=2, dim=(1, 2, 3))
                .detach()
                .cpu()
                .numpy(),
                "norms_dxdt_prime": torch.norm(d_prime, p=2, dim=(1, 2, 3))
                .detach()
                .cpu()
                .numpy(),
            }
            if snapshot_save_dir:
                for snapshot_name, snapshot_images in zip(
                    ["x_cur", "denoised", "d_cur"], [x_cur, denoised, d_cur]
                ):
                    snapshot_images = snapshot_images.clone().cpu()[:n_images_save]
                    save_seeds = rnd.seeds[:n_images_save]
                    snapshot_save_subdir = os.path.join(
                        snapshot_save_dir, snapshot_name, f"step_{i:03d}"
                    )
                    if verbose:
                        dist.print0(
                            f"Saving batch of snapshot images of shape {snapshot_images.shape} to {snapshot_save_subdir}"
                        )
                    save_images(
                        snapshot_images,
                        seeds=save_seeds,
                        save_dir=snapshot_save_subdir,
                    )

    return x_next, snapshot_dict


def parse_int_list(s):
    """
    Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    cfg_dict = OmegaConf.to_container(cfg.edm, resolve=True)
    print(cfg_dict)  # Print the configuration for debugging

    dist.init()
    seeds = parse_int_list(cfg.edm.seeds)
    num_batches = (
        (len(seeds) - 1) // (cfg.edm.max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    # Load network.
    dist.print0(f'Loading network from "{cfg.edm.network_pkl}"...')
    with dnnlib.util.open_url(cfg.edm.network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)["ema"].to(cfg.edm.device)

    # dist.print0(f"net: {net}")

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    combined_snapshot_dict = {}
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{cfg.edm.save_dir}"...')
    for batch_seeds in tqdm.tqdm(
        rank_batches, unit="batch", disable=(dist.get_rank() != 0)
    ):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(cfg.edm.device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=cfg.edm.device,
        )
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=cfg.edm.device)[
                rnd.randint(net.label_dim, size=[batch_size], device=cfg.edm.device)
            ]
        if cfg.edm.class_idx is not None:
            class_labels[:, :] = 0  # type: ignore
            class_labels[:, cfg.edm.class_idx] = 1  # type: ignore

        # Generate images.
        sampler_kwargs = {
            key: value for key, value in cfg.sampler.items() if value is not None
        }
        # dist.print0(f"sampler kwargs: {sampler_kwargs}")

        images, snapshot_dict = edm_sampler(
            net,
            rnd=rnd,
            latents=latents,
            class_labels=class_labels,
            snapshot_save_dir=cfg.snapshots.save_dir,
            snapshot_interval=cfg.snapshots.interval,
            n_images_save=cfg.snapshots.n_images_to_save_per_batch,
            verbose=False,
            **sampler_kwargs,
        )
        # Save images
        save_images(images, seeds=batch_seeds, save_dir=cfg.edm.save_dir)
        if snapshot_dict is None:
            continue

        # Gather snapshot_dict from all processes
        gathered_snapshot_dicts: List[Dict[float, Dict[str, Any]]] = [
            {} for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered_snapshot_dicts, snapshot_dict)

        # Combine gathered data on rank 0
        if dist.get_rank() == 0:
            for gathered_snapshot_dict in gathered_snapshot_dicts:
                for timestep, data in gathered_snapshot_dict.items():
                    if timestep not in combined_snapshot_dict:
                        combined_snapshot_dict[timestep] = defaultdict(list)
                    for key, values in data.items():
                        combined_snapshot_dict[timestep][key].extend(values.tolist())

    torch.distributed.barrier()
    dist.print0("Done.")

    # Convert to numpy arrays and save as .npy files
    for timestep, data in combined_snapshot_dict.items():
        for key in data.keys():
            combined_snapshot_dict[timestep][key] = np.array(data[key])

    # Save the combined snapshot dict as a pickle file
    snapshot_dict_save_path = os.path.join(
        cfg.snapshots.save_dir,
        f"snapshot_dict_seeds_{seeds[0]}-{seeds[-1]}.pkl",
    )
    with open(snapshot_dict_save_path, "wb") as f:
        pickle.dump(combined_snapshot_dict, f)

    dist.print0(f"Combined snapshot dict saved to {snapshot_dict_save_path}")


if __name__ == "__main__":
    main()
