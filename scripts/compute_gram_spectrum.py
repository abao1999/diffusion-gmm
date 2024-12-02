import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")

logger = logging.getLogger(__name__)


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


def compute_gram_spectrum(
    sample_paths: Dict[str, List[str]],
    class_name: str,
    is_npy_dataset: bool = False,
    rank: int = 100,
    num_iter: int = 2,
    save_name_prefix: str = "",
    save_dir: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    if is_npy_dataset:
        all_samples = np.stack(
            [np.load(path).flatten() for path in sample_paths[class_name]]
        )
    else:
        transform = transforms.ToTensor()
        all_samples = torch.stack(
            [
                transform(Image.open(path)).flatten()
                for path in sample_paths[class_name]
            ],
            dim=0,
        ).numpy()
    print(f"all_samples shape: {all_samples.shape}")

    gram_matrices = np.array(
        [sample[:, None] @ sample[None, :] for sample in all_samples]
    )

    print(f"gram matrices shape: {gram_matrices.shape}")
    # Perform randomized SVD on all Gram matrices at once
    U_stacked, S_stacked, Vt_stacked = randomized_svd_batch(
        gram_matrices, rank=rank, num_iter=num_iter
    )
    print(f"U_stacked shape: {U_stacked.shape}, S_stacked shape: {S_stacked.shape}")

    if save_dir is not None:
        save_name_eigenvalues = (
            f"{save_name_prefix}_{class_name}_gram_spectrum_eigenvalues.npy"
        )
        save_name_eigenvectors = (
            f"{save_name_prefix}_{class_name}_gram_spectrum_eigenvectors.npy"
        )
        np.save(os.path.join(save_dir, save_name_eigenvalues), S_stacked)
        np.save(os.path.join(save_dir, save_name_eigenvectors), U_stacked)

    # Reshape and extend eigenvalues and eigenvectors
    combined_eigenvalues = S_stacked.reshape(-1)
    return combined_eigenvalues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="edm_imagenet64_all")
    parser.add_argument("--is_npy_dataset", action="store_true")
    parser.add_argument(
        "--target_classes", type=str, nargs="+", default=["english_springer"]
    )
    parser.add_argument("--num_samples_per_class", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--num_iter", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="results/gram_spectrum")
    parser.add_argument(
        "--plot_save_dir", type=str, default="final_plots/gram_spectrum"
    )
    args = parser.parse_args()

    data_dir = os.path.join(DATA_DIR, args.split)

    sample_paths = {
        class_name: [
            os.path.join(data_dir, class_name, img)
            for img in os.listdir(os.path.join(data_dir, class_name))
        ][: args.num_samples_per_class]
        for class_name in args.target_classes
    }

    all_eigenvalues = {}
    for class_name in args.target_classes:
        logger.info(
            f"Computing gram spectrum for class {class_name} using {args.num_samples_per_class} samples"
        )
        eigenvalues = compute_gram_spectrum(
            sample_paths,
            class_name,
            is_npy_dataset=args.is_npy_dataset,
            rank=args.rank,
            num_iter=args.num_iter,
            save_name_prefix=args.split,
            save_dir=args.save_dir,
        )
        all_eigenvalues[class_name] = eigenvalues

    for class_name, eigenvalues in all_eigenvalues.items():
        plt.figure(figsize=(4, 4))
        plt.hist(eigenvalues**0.1, bins=100, density=True, alpha=0.5)
        plt.yscale("log")
        plt.ylabel("Density (log scale)")
        plt.xlabel(r"Eigenvalue ($\lambda^{{{0.1}}}$)")
        plt.title(class_name.replace("_", " ").title())
        plt.savefig(os.path.join(args.plot_save_dir, f"{args.split}_{class_name}.png"))
        plt.close()
        print(f"Saved plot for {class_name} to {args.plot_save_dir}")
