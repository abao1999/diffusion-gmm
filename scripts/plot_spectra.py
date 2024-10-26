import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_spectra_from_npy(
    real_path: Optional[str] = None,
    diffusion_path: Optional[str] = None,
    gmm_path: Optional[str] = None,
    save_dir: str = "figs",
    save_name: str = "all_spectra.png",
):
    """
    Plot spectra of Gram matrices from all three modes considered:
        1. Real images, 2. Diffusion generated images, 3. GMM generated images
    """
    os.makedirs(save_dir, exist_ok=True)
    # Load the eigenvalues computed from the Gram matrices if filepaths are valid
    real_eigenvalues = np.array([])
    diffusion_eigenvalues = np.array([])
    gmm_eigenvalues = np.array([])

    if real_path is not None and os.path.isfile(real_path):
        real_eigenvalues = np.load(real_path)
    if diffusion_path is not None and os.path.isfile(diffusion_path):
        diffusion_eigenvalues = np.load(diffusion_path)
    if gmm_path is not None and os.path.isfile(gmm_path):
        gmm_eigenvalues = np.load(gmm_path)

    print("Real eigenvalues shape: ", real_eigenvalues.shape)
    print("Diffusion eigenvalues shape: ", diffusion_eigenvalues.shape)
    print("GMM eigenvalues shape: ", gmm_eigenvalues.shape)

    bins = 100
    density = True
    plt.figure(figsize=(10, 6))
    plt.hist(
        real_eigenvalues,
        bins=bins,
        density=density,
        alpha=0.5,
        color="tab:blue",
        label="Real",
    )
    plt.hist(
        diffusion_eigenvalues,
        bins=bins,
        density=density,
        alpha=0.5,
        color="tab:orange",
        label="Diffusion",
    )
    plt.hist(
        gmm_eigenvalues,
        bins=bins,
        density=density,
        alpha=0.5,
        color="tab:green",
        label="GMM",
    )

    # Plot the spectra
    plt.title("Spectra of Gram matrices")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    print("Saved histogram to ", os.path.join(save_dir, save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, default=None)
    parser.add_argument("--gmm_path", type=str, default=None)
    parser.add_argument("--diffusion_path", type=str, default=None)
    args = parser.parse_args()
    gmm_path = args.gmm_path
    diffusion_path = args.diffusion_path
    real_path = args.real_path

    plot_spectra_from_npy(
        real_path=real_path,
        gmm_path=gmm_path,
        diffusion_path=diffusion_path,
        save_dir="figs",
        save_name="all_spectra.png",
    )
