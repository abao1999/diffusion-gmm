import torch
import os
from diffusion_gmm.diffusions import (
    generate_ddpm, 
    generate_ddpm_exposed, 
    generate_sb2,
    ldm_pipeline,
)
import argparse

from diffusion_gmm.utils import (
    save_images_grid, 
    plot_pixel_intensity_hist,
    default_image_processing_fn,
)


FIGS_DIR = "figs"
WORK_DIR = os.getenv("WORK")
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
        save_grid_shape=None,
        save_fig_dir=save_dir,
        process_fn=default_image_processing_fn,
        device=device,
    )

    print("Samples shape: ", samples.shape)
    print("Saving a 10x10 grid of the first 100 samples to figs directory...")
    save_images_grid(
        samples[:100], 
        file_path=os.path.join('figs', f"{save_name}_sample_grid.png"), 
        grid_shape=(10, 10),
        process_fn=None, # already processed
    )

    # Plot the histogram of samples generated from the fitted GMM
    print("Plotting histogram of computed pixel statistics...")
    plot_pixel_intensity_hist(samples, bins=100)

    # ldm_pipeline(
    #     num_inference_steps=50,
    #     num_images=2,
    #     save_grid_shape=None,
    #     save_fig_dir=save_dir,
    #     device=device,
    # )