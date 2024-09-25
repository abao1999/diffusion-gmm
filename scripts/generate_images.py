import argparse
import os

import torch

from diffusion_gmm.diffusions import (
    generate_ddpm_exposed,
    generate_DiTPipe,
)
from diffusion_gmm.utils import default_image_processing_fn

FIGS_DIR = "figs"
WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")

PLOT_KWARGS = {
    "save_grid_dir": FIGS_DIR,
    "save_grid_shape": (5, 5),
    "process_fn": default_image_processing_fn,
    "overwrite": False,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    args = parser.parse_args()

    os.makedirs(FIGS_DIR, exist_ok=True)
    torch.manual_seed(0)
    # If using CUDA, you should also set the seed for CUDA for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)  # if you have multiple GPUs

    # For deterministic behavior on GPU (reproducibility), use the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # save_dir = os.path.join(DATA_DIR, "diffusion_cifar10", "unknown")
    # samples = generate_ddpm_exposed(
    #     num_inference_steps=args.steps,
    #     num_images=args.n_samples,
    #     save_dir=save_dir,
    #     plot_kwargs=PLOT_KWARGS,
    #     device=device,
    # )

    class_id = 25
    class_ids = [25] * args.n_samples # salamander
    class_name = "salamander"
    save_dir = os.path.join(DATA_DIR, "diffusion_imagenet", class_name)
    os.makedirs(save_dir, exist_ok=True)

    samples = generate_DiTPipe(
        class_ids=class_ids, 
        guidance_scale=args.guidance_scale,
        save_dir=save_dir,
        plot_kwargs=PLOT_KWARGS,
        rseed=args.random_seed,
        device=device, 
    )

    print("Samples shape: ", samples.shape)
