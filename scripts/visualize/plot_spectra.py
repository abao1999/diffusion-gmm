import argparse
import os
import warnings
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["ggplot", "custom_style.mplstyle"])


def plot_spectra_from_npy(
    real_path: Optional[str] = None,
    diffusion_path: Optional[str] = None,
    gmm_path: Optional[str] = None,
    dataset_name: str = "Imagenet64",
    class_name: Optional[str] = None,
    save_dir: str = "figs",
    save_name: str = "all_spectra",
):
    """
    Plot spectra of Gram matrices from all three modes considered:
        1. Real images, 2. Diffusion generated images, 3. GMM generated images
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")

    plot_real = False
    plot_diffusion = False
    plot_gmm = False
    all_eigenvalues = []
    if real_path is not None and os.path.isfile(real_path):
        real_eigenvalues = np.load(real_path)
        print("Real eigenvalues shape: ", real_eigenvalues.shape)
        all_eigenvalues.append(real_eigenvalues)
        plot_real = True

    if diffusion_path is not None and os.path.isfile(diffusion_path):
        diffusion_eigenvalues = np.load(diffusion_path)
        print("Diffusion eigenvalues shape: ", diffusion_eigenvalues.shape)
        all_eigenvalues.append(diffusion_eigenvalues)
        plot_diffusion = True
    if gmm_path is not None and os.path.isfile(gmm_path):
        gmm_eigenvalues = np.load(gmm_path)
        print("GMM eigenvalues shape: ", gmm_eigenvalues.shape)
        all_eigenvalues.append(gmm_eigenvalues)
        plot_gmm = True
    # Determine the bins based on the combined data
    if len(all_eigenvalues) == 0:
        raise ValueError("No eigenvalues to plot")
    bins = np.histogram_bin_edges(np.concatenate(all_eigenvalues), bins=100)
    print("Bins shape: ", bins.shape)

    plt.figure(figsize=(4, 3))

    if plot_real:
        plt.hist(
            real_eigenvalues,
            bins=bins,  # type: ignore
            density=True,
            alpha=0.5,
            label="Real",
        )

    if plot_diffusion:
        plt.hist(
            diffusion_eigenvalues,
            bins=bins,  # type: ignore
            density=True,
            alpha=0.5,
            label="Diffusion",
        )

    if plot_gmm:
        plt.hist(
            gmm_eigenvalues,
            bins=bins,  # type: ignore
            density=True,
            alpha=0.5,
            label="GMM",
        )
    # Plot the spectra
    if class_name is not None:
        class_name = str(class_name).replace("_", " ")
        print("Class name: ", class_name)
        plt.title(f"{dataset_name} ({class_name})")
    else:
        plt.title(dataset_name)
    plt.xlabel("Eigenvalue")
    plt.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    plt.ylabel(r"Density ($\log$ scale)")
    plt.yscale("log")
    plt.grid(False)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    print("Saved histogram to ", save_path)


def plot_spectra_from_multiple_npy(
    diffusion_paths: Dict[str, str],
    gmm_paths: Dict[str, str],
    n_bins: int = 100,
    num_cols: int = 4,
    save_dir: str = "figs",
    save_name: str = "all_spectra_combined",
    scale_factor: Optional[float] = None,
):
    """
    Plot spectra of Gram matrices from multiple diffusion and GMM paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.png")

    num_classes = len(diffusion_paths)
    # Determine the number of rows and columns for the grid
    num_rows = (num_classes + num_cols - 1) // num_cols  # Ceiling division

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 3 * num_rows)
    )
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for ax, (class_name, diff_path) in zip(axes, diffusion_paths.items()):
        if os.path.isfile(diff_path):
            diffusion_eigenvalues: np.ndarray = np.load(diff_path)

        gmm_path = gmm_paths[class_name]
        if os.path.isfile(gmm_path):
            gmm_eigenvalues: np.ndarray = np.load(gmm_path)

        if diffusion_eigenvalues.shape != gmm_eigenvalues.shape:
            warnings.warn(
                f"Diffusion and GMM eigenvalues shapes do not match for class {class_name} \n"
                f"Diffusion shape: {diffusion_eigenvalues.shape}, GMM shape: {gmm_eigenvalues.shape}"
            )

        if scale_factor is not None:
            diffusion_eigenvalues = diffusion_eigenvalues**scale_factor  # type: ignore
            gmm_eigenvalues = gmm_eigenvalues**scale_factor  # type: ignore
            xlabel = r"Scaled Eigenvalue $\lambda^{{{}}}$".format(scale_factor)
        else:
            xlabel = r"Eigenvalue $\lambda$"

        # Combine all eigenvalues for bin calculation
        combined_eigenvalues = np.concatenate([diffusion_eigenvalues, gmm_eigenvalues])
        bins = np.histogram_bin_edges(combined_eigenvalues, bins=n_bins)

        # Plot diffusion eigenvalues
        ax.hist(
            diffusion_eigenvalues,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Diffusion",
        )

        # Plot GMM eigenvalues
        ax.hist(
            gmm_eigenvalues,
            bins=bins,
            density=True,
            alpha=0.5,
            label="GMM",
        )

        ax.set_title(class_name.replace("_", " "))
        ax.set_ylabel(r"Density ($\log$ scale)")
        ax.set_yscale("log")
        ax.grid(False)
        ax.legend()
        ax.set_xlabel(xlabel)

    # Hide any unused subplots
    for ax in axes[len(diffusion_paths) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print("Saved histogram to ", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, default=None)
    parser.add_argument("--gmm_path", type=str, default=None)
    parser.add_argument("--diffusion_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="figs")
    parser.add_argument("--save_name", type=str, default="all_spectra")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="Imagenet64")
    args = parser.parse_args()
    gmm_path = args.gmm_path
    diffusion_path = args.diffusion_path
    real_path = args.real_path

    plot_spectra_from_npy(
        real_path=real_path,
        gmm_path=gmm_path,
        diffusion_path=diffusion_path,
        dataset_name=args.dataset_name,
        class_name=args.class_name,
        save_dir=args.save_dir,
        save_name=args.save_name,
    )

    # data_dir = "results/gram_spectrum"
    # dataset_name = "Imagenet64"
    # class_list = [
    #     "baseball",
    #     "church",
    #     "english springer",
    #     "french horn",
    #     "garbage truck",
    #     "goldfinch",
    #     "kimono",
    #     "salamandra",
    #     "tabby",
    #     "tench",
    # ]
    # class_names_diffusion = [
    #     f"{dataset_name}_{class_name}_edm_gram_spectrum.npy"
    #     for class_name in class_list
    # ]
    # class_names_gmm = [
    #     f"{dataset_name}_{class_name}_gmm_gram_spectrum.npy"
    #     for class_name in class_list
    # ]
    # all_diffusion_paths = {
    #     class_name: os.path.join(data_dir, path)
    #     for class_name, path in zip(class_list, class_names_diffusion)
    # }
    # all_gmm_paths = {
    #     class_name: os.path.join(data_dir, path)
    #     for class_name, path in zip(class_list, class_names_gmm)
    # }
    # plot_spectra_from_multiple_npy(
    #     diffusion_paths=all_diffusion_paths,
    #     gmm_paths=all_gmm_paths,
    #     n_bins=100,
    #     num_cols=2,
    #     save_dir="final_plots",
    #     save_name="all_spectra_combined",
    #     scale_factor=None,
    # )
