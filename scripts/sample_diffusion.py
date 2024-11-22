import os

import hydra
import numpy as np
import torch

from diffusion_gmm.diffusions import generate_ddpm_exposed
from diffusion_gmm.utils import set_seed

FIGS_DIR = "figs"


def default_image_processing_fn(
    samples: np.ndarray, verbose: bool = True
) -> np.ndarray:
    """
    Default processing function for samples
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    min_val, max_val = samples.min(), samples.max()
    if verbose:
        print("min_val, max_val: ", min_val, max_val)
    samples = np.clip(samples, min_val, max_val)
    samples = ((samples - min_val) / (max_val - min_val)) * 255
    samples = samples.astype(np.uint8)
    return samples


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    os.makedirs(FIGS_DIR, exist_ok=True)

    set_seed(cfg.rseed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = generate_ddpm_exposed(
        num_inference_steps=cfg.diffusion.steps,
        num_images=cfg.diffusion.n_samples,
        save_dir=cfg.diffusion.save_dir,
        plot_kwargs={
            "save_grid_dir": FIGS_DIR,
            "save_grid_shape": (5, 5),
            "process_fn": default_image_processing_fn,
            "overwrite": True,
        },
        device=device,
    )

    # samples = generate_DiTPipe(
    #     class_ids=class_ids,
    #     guidance_scale=args.guidance_scale,
    #     save_dir=save_dir,
    #     plot_kwargs=PLOT_KWARGS,
    #     rseed=args.random_seed,
    #     device=device,
    # )

    print("Samples shape: ", samples.shape)


if __name__ == "__main__":
    main()
