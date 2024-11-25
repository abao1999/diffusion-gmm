import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Define the style
plt.rcParams.update(
    {
        # Font and text size
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        # Axes style
        "axes.linewidth": 0.75,
        "axes.grid": False,
        "grid.color": "gray",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.prop_cycle": cycler(
            "color",
            [
                "#377eb8",
                "#ff7f0e",
                "#4daf4a",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ],
        ),
        # Ticks
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.major.size": 4,
        "ytick.minor.size": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Figure layout
        "figure.figsize": (3.25, 2.5),  # Inches (adjust based on your needs)
        "figure.dpi": 300,
        "figure.autolayout": True,
        # Legend
        "legend.loc": "upper right",
        "legend.frameon": False,
        # Savefig options
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.transparent": True,
    }
)


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
    bins = np.histogram_bin_edges(np.concatenate(all_eigenvalues), bins=150)
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
