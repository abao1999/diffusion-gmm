import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")


def plot_spectra_from_multiple_npy(
    spectra_paths: Dict[str, str],
    n_bins: int = 100,
    num_cols: int = 4,
    save_dir: str = "figs",
    save_name: str = "cov_spectra_combined",
    scale_factor: Optional[float] = None,
    log_scale_x: bool = False,
):
    """
    Plot spectra of Gram matrices from multiple diffusion and GMM paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.png")

    num_classes = len(spectra_paths)
    # Determine the number of rows and columns for the grid
    num_rows = (num_classes + num_cols - 1) // num_cols  # Ceiling division

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(3 * num_cols, 2 * num_rows)
    )
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for idx, (ax, (class_name, spectra_path)) in enumerate(
        zip(axes, spectra_paths.items())
    ):
        if os.path.isfile(spectra_path):
            spectra_eigenvalues: np.ndarray = np.load(spectra_path)
        else:
            raise ValueError(f"File {spectra_path} does not exist")
        # not_small = np.where(spectra_eigenvalues > 1e-6)[0]
        # print(f"Number of not small eigenvalues: {len(not_small)}")
        # spectra_eigenvalues = spectra_eigenvalues[not_small]
        print(f"spectra_eigenvalues shape: {spectra_eigenvalues.shape}")
        print(f"Loaded {class_name} with shape {spectra_eigenvalues.shape}")
        if scale_factor is not None:
            spectra_eigenvalues = spectra_eigenvalues**scale_factor  # type: ignore
            xlabel = r"Scaled Eigenvalue $\lambda^{{{}}}$".format(scale_factor)  # type: ignore
        else:
            xlabel = r"Eigenvalue $\lambda$"

        # # Combine all eigenvalues for bin calculation
        # combined_eigenvalues = np.concatenate([spectra_eigenvalues])
        # bins = np.histogram_bin_edges(combined_eigenvalues, bins=n_bins)

        # Plot spectra eigenvalues
        ax.hist(
            spectra_eigenvalues,
            bins=n_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.8,
            label="Diffusion",
        )

        # Plot GMM eigenvalues
        ax.set_title(class_name.replace("_", " ").title(), fontweight="bold")
        if idx % num_cols == 0:  # First plot in each row
            ax.set_ylabel(r"Density (log scale)", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(False)
        # ax.legend()
        if log_scale_x:
            ax.set_xscale("log")
            ax.set_xlabel(f"{xlabel} (log scale)")
        else:
            ax.set_xlabel(xlabel)

    # Hide any unused subplots
    for ax in axes[len(spectra_paths) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print("Saved histogram to ", save_path)


def compute_and_save_spectra(cov_path: str, class_name: str, save_dir: str):
    cov = np.load(cov_path)
    print(
        f"Loaded covariance matrix for {class_name} with shape {cov.shape} from path {cov_path}"
    )
    eigenvalues = np.linalg.eigvalsh(cov)
    print(f"Computed eigenvalues for {class_name} with shape {eigenvalues.shape}")
    save_path = os.path.join(save_dir, f"{class_name}_eigenvalues.npy")
    print(f"Saving eigenvalues to {save_path}")
    np.save(save_path, eigenvalues)


if __name__ == "__main__":
    # dataset_name = "edm_imagenet64_all"
    dataset_name = "representations"
    stats_dir = os.path.join(DATA_DIR, "computed_stats", dataset_name)
    spectra_dir = os.path.join(stats_dir, "spectra")
    eigenvalues_paths_dict = {}
    for filename in os.listdir(spectra_dir):
        if filename.endswith("eigenvalues.npy"):
            class_name = filename.split("_eigenvalues.npy")[0]
            eigenvalues_paths_dict[class_name] = os.path.join(spectra_dir, filename)
            print(f"{class_name}: {filename}")

    eigenvalues_paths_dict = dict(sorted(eigenvalues_paths_dict.items()))
    plot_spectra_from_multiple_npy(
        spectra_paths=eigenvalues_paths_dict,
        n_bins=100,
        num_cols=4,
        save_dir="final_plots",
        save_name="representations_eigenvalues_spectra_combined",
        scale_factor=None,
        log_scale_x=False,
    )
    exit()

    cov_paths_dict = {}
    for filename in os.listdir(stats_dir):
        if filename.endswith("covariance.npy"):
            class_name = filename.split("_covariance.npy")[0]
            cov_paths_dict[class_name] = os.path.join(stats_dir, filename)
            print(f"{class_name}: {filename}")

    save_dir = os.path.join(stats_dir, "spectra")
    os.makedirs(save_dir, exist_ok=True)
    for class_name, cov_path in cov_paths_dict.items():
        print(f"{class_name}: {cov_path}")
        compute_and_save_spectra(cov_path, class_name, save_dir)
