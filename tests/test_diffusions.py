"""
Test the diffusion pipelines
"""

import argparse
import os
from typing import List

import torch

from diffusion_gmm.diffusions import (
    generate_ddpm_exposed,
    generate_DiTPipe,
)
from diffusion_gmm.utils import default_image_processing_fn

NUM_SAMPLES = 4
FIGS_DIR = "tests/figs"
PLOT_KWARGS = {
    "save_grid_dir": FIGS_DIR,
    "save_grid_shape": (2, 2),
    "process_fn": default_image_processing_fn,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_generate_DiTPipe(
    model_id: str = "facebook/DiT-XL-2-256",
    class_ids: List[int] = [0],
    guidance_scale: float = 1.0,
    num_inference_steps: int = 50,
    save_dir: str = "tests/figs",
):
    """
    Test the DiT pipeline with the given class ids, guidance scale, and number of inference steps
    """
    os.makedirs(save_dir, exist_ok=True)
    samples = generate_DiTPipe(
        model_id=model_id,
        class_ids=class_ids,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        save_dir=save_dir,
        plot_kwargs=PLOT_KWARGS,
        device=DEVICE,
    )
    print("Samples shape: ", samples.shape)


def test_generate_ddpm_exposed(
    model_id: str = "google/ddpm-cifar10-32",
    num_inference_steps: int = 50,
    save_dir: str = "tests/figs",
):
    """
    Test the DDPM pipeline with the given number of inference steps and save the samples to the given directory
    """
    os.makedirs(save_dir, exist_ok=True)
    samples = generate_ddpm_exposed(
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        num_images=NUM_SAMPLES,
        save_dir=save_dir,
        plot_kwargs=PLOT_KWARGS,
        device=DEVICE,
    )
    print("Samples shape: ", samples.shape)


if __name__ == "__main__":
    print("Using device: ", DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, choices=["cifar10", "imagenet", "imagenet64"]
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    print("plot kwargs: ", PLOT_KWARGS)

    if dataset_name == "cifar10":
        save_dir = os.path.join(FIGS_DIR, "diffusion_cifar10")
        model_id = "google/ddpm-cifar10-32"
        print(
            f"Generating Diffusion CIFAR10 samples using {model_id} and saving to {save_dir}"
        )
        num_inference_steps = 50
        print(
            f"Using {num_inference_steps} inference steps. This is an UNCONDITIONAL model"
        )
        test_generate_ddpm_exposed(model_id, num_inference_steps, save_dir)

    # Imagenet 1000 classes
    elif dataset_name == "imagenet":
        save_dir = os.path.join(FIGS_DIR, "diffusion_imagenet")
        model_id = "facebook/DiT-XL-2-256"
        print(
            f"Generating Diffusion Imagenet samples using {model_id} and saving to {save_dir}"
        )
        num_inference_steps = 100
        class_ids = [0] * NUM_SAMPLES
        guidance_scale = 1.5
        print(
            f"Using {num_inference_steps} inference steps. This is a CONDITIONAL model with guidance scale {guidance_scale} for class ids {class_ids}"
        )
        test_generate_DiTPipe(
            model_id, class_ids, guidance_scale, num_inference_steps, save_dir
        )

    # Imagenet 1000 classes downsampled to 64x64
    elif dataset_name == "imagenet64":
        save_dir = os.path.join(FIGS_DIR, "diffusion_imagenet64")
        model_id = None
        raise NotImplementedError("Diffusion for Imagenet64 is not implemented yet")
        # print(
        #     f"Generating Diffusion Imagenet64 samples using {model_id} and saving to {save_dir}"
        # )
        # num_inference_steps = 100
        # class_ids = [0] * NUM_SAMPLES
        # guidance_scale = 1.5
        # print(
        #     f"Using {num_inference_steps} inference steps. This is a CONDITIONAL model with guidance scale {guidance_scale} for class ids {class_ids}"
        # )
        # test_generate_DiTPipe(
        #     model_id, class_ids, guidance_scale, num_inference_steps, save_dir
        # )
