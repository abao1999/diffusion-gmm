import os
import warnings
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


# Function to compute the Gram matrix
def compute_gram_matrix(features: np.ndarray) -> np.ndarray:
    b, c, h, w = features.shape
    features = features.reshape(b, c, h * w)
    gram_matrix = np.matmul(features, features.transpose(0, 2, 1))
    # features = features.view(b, c, h * w)
    # gram_matrix = torch.bmm(features, features.transpose(1, 2))
    return gram_matrix


# Function to get eigenvalues of the Gram matrix
def get_gram_spectrum(gram_matrix: np.ndarray) -> np.ndarray:
    # Compute the eigenvalues (complex values) of each Gram matrix in the batch
    eigenvalues = np.linalg.eigvals(gram_matrix)
    # Extract the real part of the eigenvalues
    real_eigenvalues = eigenvalues.real
    # Flatten the eigenvalues to form a 1D array
    all_eigenvalues = real_eigenvalues.flatten()
    return all_eigenvalues


# plot histogram from npy file
def plot_gram_spectrum(
    all_eigenvalues: np.ndarray,
    verbose: bool = False,
    bins: Union[int, str] = 100,  # 'fd' for Freedman-Diaconis rule
    density: bool = True,
    save_dir: str = "figs",
    save_name: str = "gram_spectrum.png",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        print()
    plt.figure(figsize=(10, 6))
    plt.hist(all_eigenvalues, bins=bins, density=density, alpha=0.75, color="tab:blue")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title("Eigenvalues of Gram Matrices")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    print("Saved histogram to ", os.path.join(save_dir, save_name))


# plot histogram from npy file
def plot_from_npy(
    filepath: str = "gram_spectrum.npy",
    plot_fn: Callable[[np.ndarray], None] = plot_gram_spectrum,
    verbose: bool = False,
    plot_kwargs: dict = {},
):
    all_data = np.load(filepath)
    if verbose:
        print(f"Loaded data from file {filepath}")
        print("data shape: ", all_data.shape)
    plot_fn(all_data, **plot_kwargs)


def save_sample(
    sample: torch.Tensor,
    i: int,
    file_path: str,
    custom_figsize: Optional[Tuple[int, int]] = None,
) -> None:
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])

    print(f"Saving image at step {i} to {file_path}")

    if custom_figsize:
        plt.figure(figsize=custom_figsize)
        plt.imshow(image_pil)
        plt.savefig(file_path)
    else:
        image_pil.save(file_path)


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

    print(f"Populating grid of images wit the first {num_images} images")
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

    TODO: keep as torch.Tensor or compute to np.ndarray?
    """

    print("Saving samples to ", save_dir)
    if process_fn is not None:
        # apply postprocessing to samples
        samples = process_fn(samples)
        assert (
            samples.dtype == np.uint8
        ), "Samples must be converted to np.uint8 before saving as image."

        # save image grid of subset of samples
        if save_grid_dir is not None:
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

        # save all samples as png images in save_dir
        for i, img in enumerate(samples):
            # Reorder dimensions to (h, w, c)
            img = np.transpose(img, (1, 2, 0))
            # convert to Image PIL type
            img = Image.fromarray(img)
            save_path = os.path.join(save_dir, f"sample_{i}.png")
            img.save(save_path)

    else:
        warnings.warn(
            "No image post-processing function provided. Saving samples as numpy files. This may not be the desired behavior."
        )
        # Default behavior: save as npy files
        for i, img in enumerate(samples):
            save_path = os.path.join(save_dir, f"sample_{i}.npy")
            np.save(save_path, img)


def plot_pixel_intensity_hist(
    samples: np.ndarray,
    bins: int = 30,
    save_dir: str = "figs",
    save_name: str = "pixel_intensities.png",
    verbose: bool = False,
) -> None:
    """
    Plot the histogram of pixel intensity
    """
    # number of samples should be batch dimension (first dimension)
    num_samples = samples.shape[0]

    if verbose:
        print("Plotting histogram of pixel intensity...")
        print(f"Using {num_samples} samples and {bins} bins for histogram.")

    # Flatten the samples for visualization (optional)
    flattened_samples = samples.flatten()

    # Plot histogram of the generated samples
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_samples, bins=bins, color="tab:blue", alpha=0.7)
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)


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
    # samples = torch.clamp(samples, min_val, max_val)
    # samples = samples.to(torch.uint8)
    return samples
