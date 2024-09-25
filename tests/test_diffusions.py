from diffusion_gmm.diffusions import (
    generate_DiTPipe,
    generate_ddpm_exposed,
)
from diffusion_gmm.utils import default_image_processing_fn
import os
import torch

NUM_SAMPLES = 4
FIGS_DIR = "tests/figs"
DATA_DIR = "tests"
PLOT_KWARGS={
    "save_grid_dir": FIGS_DIR,
    "save_grid_shape": (2, 2),
    "process_fn": default_image_processing_fn,
}

def test_generate_DiTPipe(
    save_dir: str = "tests/figs",
    device: str = "cpu",
):
    class_ids = [25] * NUM_SAMPLES # salamander

    samples = generate_DiTPipe(
        class_ids=class_ids, 
        save_dir=save_dir,
        plot_kwargs=PLOT_KWARGS,
        device=device, 
    )
    print("Samples shape: ", samples.shape)

def test_generate_ddpm_exposed(
    save_dir: str = "tests/figs",
    device: str = "cpu",
):
    samples = generate_ddpm_exposed(
        num_inference_steps=50,
        num_images=NUM_SAMPLES,
        save_dir=save_dir,
        plot_kwargs=PLOT_KWARGS,
        device=device,
    )
    print("Samples shape: ", samples.shape)

if __name__ == "__main__":
    os.makedirs(FIGS_DIR, exist_ok=True)
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("plot kwargs: ", PLOT_KWARGS)

    # sdfsdf

    # save_dir = os.path.join(DATA_DIR, "diffusion_cifar10")
    # test_generate_ddpm_exposed(save_dir, device)
    
    save_dir = os.path.join(DATA_DIR, "diffusion_imagenet")
    test_generate_DiTPipe(save_dir, device)