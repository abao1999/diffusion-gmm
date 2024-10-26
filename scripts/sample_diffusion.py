import os

import hydra
import torch

from diffusion_gmm.diffusions import generate_ddpm_exposed
from diffusion_gmm.utils import default_image_processing_fn

FIGS_DIR = "figs"
PLOT_KWARGS = {
    "save_grid_dir": FIGS_DIR,
    "save_grid_shape": (5, 5),
    "process_fn": default_image_processing_fn,
    "overwrite": True,
}


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    os.makedirs(FIGS_DIR, exist_ok=True)
    torch.manual_seed(cfg.rseed)
    # If using CUDA, you should also set the seed for CUDA for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.rseed)
        torch.cuda.manual_seed_all(cfg.rseed)  # if you have multiple GPUs

    # For deterministic behavior on GPU (reproducibility), use the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = generate_ddpm_exposed(
        num_inference_steps=cfg.diffusion.steps,
        num_images=cfg.diffusion.n_samples,
        save_dir=cfg.diffusion.save_dir,
        plot_kwargs=PLOT_KWARGS,
        device=device,
    )

    # class_id = 25
    # class_ids = [25] * args.n_samples  # salamander
    # class_name = "salamander"
    # save_dir = os.path.join(DATA_DIR, "diffusion_imagenet", class_name)
    # os.makedirs(save_dir, exist_ok=True)

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
