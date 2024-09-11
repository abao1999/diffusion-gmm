import torch
import os
from diffusion_gmm.diffusions import (
    generate_ddpm, 
    generate_ddpm_exposed, 
    generate_sb2,
    ldm_pipeline,
)

FIGS_DIR = "figs"
WORK_DIR = os.getenv("WORK")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")

if __name__ == "__main__":
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

    save_dir = os.path.join(DATA_DIR, "diffusion_cifar10", "unknown")

    # generate_sb3(device=device)
    # # generate_ddpm(
    # generate_ddpm_exposed(
    #     num_inference_steps=50,
    #     num_images=1024,
    #     save_grid_shape=None,
    #     save_fig_dir=save_dir,
    #     device=device,
    # )

    ldm_pipeline(
        num_inference_steps=50,
        num_images=2,
        save_grid_shape=None,
        save_fig_dir=save_dir,
        device=device,
    )