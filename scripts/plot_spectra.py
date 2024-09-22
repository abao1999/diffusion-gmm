import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_spectra_from_npy(
    real_npy_filepath: str = "gram_spectrum.npy",
    diffusion_npy_filepath: str = "diffusion_gram_spectrum.npy",
    gmm_npy_filepath: str = "gmm_gram_spectrum.npy",
    save_dir: str = "figs",
    save_name: str = "all_spectra.png",
):
    """
    Plot spectra of Gram matrices from all three modes considered:
        1. Real images, 2. Diffusion generated images, 3. GMM generated images
    """
    # Load the eigenvalues computed from the Gram matrices
    real_eigenvalues = np.load(real_npy_filepath)
    diffusion_eigenvalues = np.load(diffusion_npy_filepath)
    gmm_eigenvalues = np.load(gmm_npy_filepath)

    print("Real eigenvalues shape: ", real_eigenvalues.shape)
    print("Diffusion eigenvalues shape: ", diffusion_eigenvalues.shape)
    print("GMM eigenvalues shape: ", gmm_eigenvalues.shape)

    # assert real_eigenvalues.shape == diffusion_eigenvalues.shape == gmm_eigenvalues.shape, "Eigenvalues shapes do not match"

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
    data_dir = "results"
    dataset_name = "cifar10"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_class_name", type=str, default=None, help="Target class for real data"
    )
    args = parser.parse_args()
    target_class_name = args.target_class_name

    path_suffix = "gram_spectrum.npy"
    if target_class_name is not None:
        path_suffix = f"{target_class_name}_{path_suffix}"

    plot_spectra_from_npy(
        real_npy_filepath=os.path.join(data_dir, f"{dataset_name}_{path_suffix}"),
        diffusion_npy_filepath=os.path.join(
            data_dir, f"ddpm_{dataset_name}_{path_suffix}"
        ),
        gmm_npy_filepath=os.path.join(data_dir, f"gmm_{dataset_name}_{path_suffix}"),
        save_dir="figs",
        save_name=f"{target_class_name or 'all'}_spectra.png",
    )
