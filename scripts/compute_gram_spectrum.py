import argparse
import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from diffusion_gmm.utils import setup_dataset

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")

plt.style.use(["ggplot", "custom_style.mplstyle"])


def randomized_svd_batch(
    A: np.ndarray, rank: int, num_iter: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a randomized SVD on a batch of matrices A.

    Args:
        A (np.ndarray): Input batch of matrices of shape (batch_size, m, n).
        rank (int): Target rank for the approximation.
        num_iter (int): Number of power iterations (improves accuracy).

    Returns:
        U (np.ndarray): Left singular vectors for each matrix.
        S (np.ndarray): Singular values for each matrix.
        Vt (np.ndarray): Right singular vectors (transposed) for each matrix.
    """

    start_time = time.time()  # Start the timer

    batch_size, m, n = A.shape
    random_matrix = np.random.randn(batch_size, n, rank)
    print(f"random matrix shape: {random_matrix.shape}")
    Y = np.einsum("bij,bjk->bik", A, random_matrix)
    Q, _ = np.linalg.qr(Y)

    B = np.einsum("bij,bjk->bik", Q.transpose(0, 2, 1), A)

    print("Computing SVD")
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

    U = np.einsum("bij,bjk->bik", Q, U_hat)

    end_time = time.time()  # End the timer
    print(f"randomized_svd_batch execution time: {end_time - start_time:.4f} seconds")

    return U, S, Vt


def make_plot(
    data_dir: str,
    data_split: str,
    class_names: List[str],
    n_samples_per_class: int,
    plot_save_dir: str,
    plot_title: str = "Gram Spectrum",
    density: bool = False,
):
    for class_name in class_names:
        plt.figure(figsize=(4, 3))
        data_path = os.path.join(
            data_dir,
            data_split,
            f"{class_name}_n{n_samples_per_class}_gramian_eigenvalues.npy",
        )
        gmm_path = os.path.join(
            data_dir,
            f"gmm_{data_split}",
            f"{class_name}_n{n_samples_per_class}_gramian_eigenvalues.npy",
        )
        data_gramian_eigs = np.load(data_path).flatten()
        gmm_gramian_eigs = np.load(gmm_path).flatten()
        min_eig = min(data_gramian_eigs.min(), gmm_gramian_eigs.min())
        if min_eig < 1e-4:
            min_eig = 1e-4
        max_eig = max(data_gramian_eigs.max(), gmm_gramian_eigs.max())
        print(f"min_eig: {min_eig}, max_eig: {max_eig}")
        bins = np.logspace(np.log10(min_eig), np.log10(max_eig), num=100)
        plt.hist(
            data_gramian_eigs,
            bins=bins,  # type: ignore
            density=density,
            histtype="stepfilled",
            alpha=0.5,
            label="Diffusion",
        )
        plt.hist(
            gmm_gramian_eigs,
            bins=bins,  # type: ignore
            density=density,
            histtype="stepfilled",
            alpha=0.5,
            label="GMM",
        )
        plt.title(f"{plot_title}")
        plt.xlabel("Eigenvalue (log scale)")
        if density:
            plt.ylabel("Density (log scale)")
        else:
            plt.ylabel("Count (log scale)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plot_save_path = os.path.join(
            plot_save_dir,
            f"{class_name}_gram_spectrum{'_density' if density else ''}.pdf",
        )
        plt.savefig(plot_save_path)
        print(f"Saved histogram to {plot_save_path}")


def plot_spectra_from_multiple_npy(
    data_dir: str,
    spectra_fnames: dict[str, str],
    n_bins: int = 100,
    num_cols: int = 4,
    save_dir: str = "figs",
    save_name: str = "gramian_spectra_combined",
    density: bool = True,
):
    """
    Plot spectra of Gram matrices from multiple diffusion and GMM paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")

    num_classes = len(spectra_fnames)
    # Determine the number of rows and columns for the grid
    num_rows = (num_classes + num_cols - 1) // num_cols  # Ceiling division

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 3 * num_rows)
    )
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for idx, (ax, (name, spectra_fname)) in enumerate(
        zip(axes, spectra_fnames.items())
    ):
        data_path = os.path.join(
            data_dir,
            "edm_imagenet64_all",
            spectra_fname,
        )
        gmm_path = os.path.join(
            data_dir,
            "gmm_edm_imagenet64_all",
            spectra_fname,
        )
        data_gramian_eigs = np.load(data_path).flatten()
        gmm_gramian_eigs = np.load(gmm_path).flatten()
        min_eig = min(data_gramian_eigs.min(), gmm_gramian_eigs.min())
        if min_eig < 1e-4:
            min_eig = 1e-4
        max_eig = max(data_gramian_eigs.max(), gmm_gramian_eigs.max())
        print(f"min_eig: {min_eig}, max_eig: {max_eig}")
        bins = np.logspace(np.log10(min_eig), np.log10(max_eig), num=n_bins)
        ax.hist(
            data_gramian_eigs,
            bins=bins,
            density=density,
            histtype="stepfilled",
            alpha=0.5,
            label="Diffusion",
        )
        ax.hist(
            gmm_gramian_eigs,
            bins=bins,
            density=density,
            histtype="stepfilled",
            alpha=0.5,
            label="GMM",
        )
        ax.legend()
        # Plot GMM eigenvalues
        ax.set_title(name.replace("_", " ").title(), fontweight="bold")
        if idx % num_cols == 0:  # First plot in each row
            if density:
                ax.set_ylabel("Density (log scale)", fontweight="bold")
            else:
                ax.set_ylabel("Count (log scale)", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(False)
        ax.set_xscale("log")
        ax.set_xlabel("Eigenvalue (log scale)")

    # Hide any unused subplots
    for ax in axes[len(spectra_fnames) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print("Saved histogram to ", save_path)


def compute_and_save_gramian_eigs(
    X_matrix: np.ndarray, save_dir: str, save_name: str
) -> np.ndarray:
    gramian = X_matrix @ X_matrix.T
    print(f"shape of gramian: {gramian.shape}")

    gramian_eigs = np.linalg.eigvalsh(gramian)
    print(f"shape of gramian eigenvalues: {gramian_eigs.shape}")
    save_path = os.path.join(
        save_dir,
        f"{save_name}_gramian_eigenvalues.npy",
    )
    np.save(save_path, gramian_eigs)
    print(f"Saved gramian eigenvalues to {save_path}")
    return gramian_eigs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_classes", type=str, nargs="+", required=True)
    parser.add_argument("--data_split", type=str, default="representations")
    parser.add_argument("--n_samples_per_class", type=int, default=1024)
    parser.add_argument(
        "--plot_save_dir", type=str, default="final_plots/gram_spectrum"
    )
    parser.add_argument("--plot_name", type=str, default=None)
    parser.add_argument(
        "--data_save_dir", type=str, default=os.path.join(DATA_DIR, "gram_spectrum")
    )
    parser.add_argument("--use_mixture", action="store_true")
    args = parser.parse_args()

    plot_save_dir = os.path.join(args.plot_save_dir, args.data_split)
    os.makedirs(plot_save_dir, exist_ok=True)

    print(f"mixture: {args.use_mixture}")
    # make_plot(
    #     args.data_save_dir,
    #     args.data_split,
    #     args.target_classes,
    #     args.n_samples_per_class,
    #     plot_save_dir,
    #     density=True,
    # )
    # exit()
    # make_plot(
    #     args.data_save_dir,
    #     args.data_split,
    #     [f"{len(args.target_classes)}classes_mixture"],
    #     args.n_samples_per_class,
    #     plot_save_dir,
    #     density=True,
    #     plot_title=f"{len(args.target_classes)} Classes",
    # )
    # exit()
    # spectra_fnames = {
    #     "4 Classes": "4classes_mixture_n2048_gramian_eigenvalues.npy",
    #     "10 Classes": "10classes_mixture_n512_gramian_eigenvalues.npy",
    #     "20 Classes": "20classes_mixture_n512_gramian_eigenvalues.npy",
    # }
    # plot_spectra_from_multiple_npy(
    #     data_dir=args.data_save_dir,
    #     spectra_fnames=spectra_fnames,
    #     n_bins=100,
    #     num_cols=3,
    #     save_dir=plot_save_dir,
    #     density=True,
    # )
    # exit()

    save_dir = os.path.join(args.data_save_dir, args.data_split)
    os.makedirs(save_dir, exist_ok=True)

    class_list = args.target_classes
    data_dir = os.path.join(DATA_DIR, args.data_split)
    print(f"data_dir: {data_dir}")

    print(f"num classes: {len(class_list)}")
    print("Setting up dataset...")
    dataset, is_npy_dataset = setup_dataset(data_dir)
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None
    print("Dataset setup complete.")

    # NOTE: has to be ImageFolder to have samples attribute.
    image_paths, targets = zip(*dataset.samples)
    targets = np.array(targets)
    # targets = get_targets(dataset) # for case when dataset is not ImageFolder
    class_to_idx = dataset.class_to_idx
    indices_by_class = {
        cls: np.where(targets == class_to_idx[cls])[0].tolist() for cls in class_list
    }
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    X_matrix = []
    for class_name, indices in tqdm(
        indices_by_class.items(), desc="Processing classes"
    ):
        print(f"Class {class_name} has {len(indices)} samples")

        selected_indices = indices[: args.n_samples_per_class]
        print(f"Selected {len(selected_indices)} samples for class {class_name}")
        class_subset = Subset(dataset, selected_indices)
        dataloader = DataLoader(
            class_subset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
        )

        for samples, _ in tqdm(dataloader, desc="Gathering data"):
            batch_size = samples.shape[0]
            X_matrix.append(samples.cpu().numpy().reshape(batch_size, -1))

        if args.use_mixture:
            continue

        X_matrix = np.concatenate(X_matrix, axis=0)
        print(f"shape of X_matrix: {X_matrix.shape}")
        print(f"Computing gramian eigenvalues for {class_name}")
        _ = compute_and_save_gramian_eigs(
            X_matrix, save_dir, f"{class_name}_n{args.n_samples_per_class}"
        )
        X_matrix = []

    if args.use_mixture:
        n_classes = len(class_list)
        X_matrix = np.concatenate(X_matrix, axis=0)
        print(f"shape of X_matrix: {X_matrix.shape}")
        print("Computing gramian eigenvalues for mixture")
        print(f"shape of final X_matrix: {X_matrix.shape}")
        _ = compute_and_save_gramian_eigs(
            X_matrix,
            save_dir,
            f"{n_classes}classes_mixture_n{args.n_samples_per_class}",
        )
