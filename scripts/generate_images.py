import argparse
import os

import torch

from diffusion_gmm.diffusions import (
    generate_ddpm_exposed,
)
from diffusion_gmm.utils import (
    default_image_processing_fn,
    plot_pixel_intensity_hist,
)

FIGS_DIR = "figs"
WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=1024)
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

    dataset_name = "cifar10"
    save_dir = os.path.join(DATA_DIR, "diffusion_cifar10", "unknown")
    save_name = f"diffusion_{dataset_name}"

    # generate_sb3(device=device)
    samples = generate_ddpm_exposed(
        num_inference_steps=args.steps,
        num_images=args.n_samples,
        save_dir=save_dir,
        plot_kwargs={
            "save_grid_dir": FIGS_DIR,
            "save_grid_shape": (10, 10),
            "process_fn": default_image_processing_fn,
        },
        device=device,
    )

    print("Samples shape: ", samples.shape)

    # Plot the histogram of samples generated from the fitted GMM
    print("Plotting histogram of computed pixel statistics...")
    plot_pixel_intensity_hist(samples, bins=100)

    # ldm_pipeline(
    #     num_inference_steps=50,
    #     num_images=2,
    #     save_grid_shape=None,
    #     save_dir=save_dir,
    #     device=device,
    # )
