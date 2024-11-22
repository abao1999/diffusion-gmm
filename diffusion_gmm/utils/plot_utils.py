import os
import warnings
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image


def save_images_grid(
    images: np.ndarray,
    file_path: str,
    grid_shape: Tuple[int, int] = (10, 10),
) -> None:
    """
    Save a grid of images to a file
    Args:
        images: Numpy array of images
        file_path: Path to save the image
        grid_shape: Shape of the grid
        process_fn: Function to process the images before saving
    """
    assert images.dtype == np.uint8, "Images must be in uint8 format."

    num_images = grid_shape[0] * grid_shape[1]
    assert num_images <= images.shape[0], "Fewer images than grid spaces!"

    print(f"Populating grid of images with the first {num_images} images")
    images = images[:num_images]
    _, _, w, h = images.shape
    n_rows = w * grid_shape[0]
    n_cols = h * grid_shape[1]

    # create a new RGB image to paste the images onto
    new_im = Image.new("RGB", (n_rows, n_cols))

    idx = 0
    for i in range(0, n_rows, w):
        for j in range(0, n_cols, h):
            img = images[idx : idx + 1].squeeze()
            # Reorder dimensions to (h, w, c)
            img = np.transpose(img, (1, 2, 0))
            # convert to Image PIL type
            im = Image.fromarray(img)
            # paste the image at location i,j:
            new_im.paste(im, (i, j))
            idx += 1

    new_im.save(file_path)


def save_and_plot_samples(
    samples: np.ndarray,
    save_dir: str,
    save_grid_dir: Optional[str] = None,
    save_grid_shape: Tuple[int, int] = (10, 10),
    process_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    overwrite: bool = False,
) -> None:
    """
    Save and plot samples to data directory, and optionally saves grid of subset of samples
    Args:
        samples: Generated samples from the model, converted to numpy array
        save_dir: Directory to save the samples as png images for loading as a torchvision ImageFolder

    Optional Args:
        process_fn: Callable post-processing function from np array Image PIL type (e.g. for scaling, clipping, etc.)
                    NOTE: the dtype must be made into np.uint8 before saving as image
                    If None, default behavior is to save unprocessed samples into npy files
        save_grid: Boolean flag, whether to save grid of subset of samples
        save_grid_shape: Tuple of grid shape for the grid of samples
        save_grid_dir: Directory to save the grid of samples

    """
    os.makedirs(save_dir, exist_ok=True)
    print("Saving samples to ", save_dir)
    if process_fn is not None:
        # apply postprocessing to samples
        samples = process_fn(samples)
        assert (
            samples.dtype == np.uint8
        ), "Samples must be converted to np.uint8 before saving as image."

        # save image grid of subset of samples
        if save_grid_dir is not None:
            os.makedirs(save_grid_dir, exist_ok=True)
            parent_name = os.path.basename(os.path.dirname(save_dir))
            curr_name = os.path.basename(save_dir)
            save_name = f"{parent_name}_{curr_name}"
            grid_save_path = os.path.join(save_grid_dir, f"{save_name}_sample_grid.png")
            print("Saving grid of samples to ", grid_save_path)
            save_images_grid(
                samples,
                file_path=grid_save_path,
                grid_shape=save_grid_shape,
            )

        last_sample_index = -1

        if not overwrite:
            print("Checking for existing samples in save_dir...")
            # find the last sample index in save_dir
            sample_files = [
                f
                for f in os.listdir(save_dir)
                if os.path.isfile(os.path.join(save_dir, f))
            ]
            sample_indices = [
                int(f.split("_")[-1].split(".")[0])
                for f in sample_files
                if "sample" in f
            ]
            if sample_indices:
                last_sample_index = max(sample_indices)
            print("Last sample index: ", last_sample_index)

        # save all samples as png images in save_dir
        for i, img in enumerate(samples):
            # Reorder dimensions to (h, w, c)
            img = np.transpose(img, (1, 2, 0))
            # convert to Image PIL type
            img = Image.fromarray(img)
            sample_idx = last_sample_index + i + 1
            save_path = os.path.join(save_dir, f"sample_{sample_idx}.png")
            img.save(save_path)

    else:
        warnings.warn(
            "No image post-processing function provided. Saving samples as numpy files. This may not be the desired behavior."
        )
        # Default behavior: save as npy files
        for i, img in enumerate(samples):
            save_path = os.path.join(save_dir, f"sample_{i}.npy")
            np.save(save_path, img)
