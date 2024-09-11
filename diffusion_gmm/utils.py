import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple
import PIL.Image

# Function to compute the Gram matrix
def compute_gram_matrix(features: np.ndarray) -> np.ndarray:
    c, h, w = features.shape
    features = features.view(c, h * w)
    gram_matrix = torch.mm(features, features.t())
    return gram_matrix

# Function to get eigenvalues of the Gram matrix
def get_gram_spectrum(gram_matrix):
    # Compute the eigenvalues (complex values) of the Gram matrix
    eigenvalues = torch.linalg.eigvals(gram_matrix)
    # Return the real part of the eigenvalues
    return eigenvalues.real


# plot histogram from npy file
def plot_gram_spectrum(
    all_eigenvalues: np.ndarray, 
    verbose: bool = False,
    bins: Union[int, str] = 100, # 'fd' for Freedman-Diaconis rule
    density: bool = True,
    save_dir: str = 'figs',
    save_name: str = 'gram_spectrum.png'
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        print()
    plt.figure(figsize=(10, 6))
    plt.hist(
        all_eigenvalues, 
        bins=bins, 
        density=density, 
        alpha=0.75, 
        color='tab:blue'
    )
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.title('Eigenvalues of Gram Matrices')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    print("Saved histogram to ", os.path.join(save_dir, save_name))

# plot histogram from npy file
def plot_from_npy(
    filepath: str = 'gram_spectrum.npy',
    plot_fn: Callable[[np.ndarray], None] = plot_gram_spectrum,
    verbose: bool = False,
    plot_kwargs: dict = {}
):
    all_data = np.load(filepath)
    if verbose:
        print(f"Loaded data from file {filepath}")
        print("data shape: ", all_data.shape)
    plot_fn(all_data, verbose=verbose, **plot_kwargs)

def save_sample(sample, i, file_path, custom_figsize=None):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])

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
    grid_shape: Tuple[int, int] = (20,20),
) -> None:
    n, _, w, h = images.shape
    n_rows = w * grid_shape[0]
    n_cols = h * grid_shape[1]

    assert n >= grid_shape[0] * grid_shape[1], "you have fewer images than grid spaces!"

    new_im = PIL.Image.new('RGB', (n_rows, n_cols))

    idx = 0
    for i in range(0, n_rows, w):
        for j in range(0, n_cols, h):
            image_processed = images[idx:idx+1].squeeze()
            image_processed = ((image_processed + 1) / 2) * 255
            image_processed = image_processed.astype(np.uint8)
            image_processed = np.transpose(image_processed, (1, 2, 0))
            im = PIL.Image.fromarray(image_processed)
            # paste the image at location i,j:
            new_im.paste(im, (i, j))
            idx += 1

    new_im.save(file_path)

def plot_pixel_intensity_hist(
    samples: np.ndarray, 
    bins: int = 30, 
    save_dir: str = 'figs',
    save_name: str = 'pixel_intensities.png',
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
    plt.hist(flattened_samples, bins=bins, color='tab:blue', alpha=0.7)
    plt.title('Pixel Intensity Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)