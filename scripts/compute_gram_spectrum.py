import argparse
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from diffusion_gmm.base import DataPrefetcher
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


def compute_gramian_eigenvalues(
    dataloader: DataLoader | DataPrefetcher,
    device: str = "cpu",
    p: int = 2,
) -> np.ndarray:
    """
    Dataloader is for a specific class.
    Compute the eigenvalues of all images for a specific class.
    """
    gramian_eigs = []

    for samples, _ in tqdm(dataloader, desc="Computing gramian eigenvalues"):
        samples = samples.to(device)
        gramian = samples[:, :, None] @ samples[:, None, :]
        print(f"shape of batch of gramians: {gramian.shape}")

        # Use PyTorch to compute eigenvalues for each matrix in the batch
        eigs = np.linalg.eigvalsh(gramian)

        print(f"shape of batch of eigenvalues: {eigs.shape}")
        gramian_eigs.extend(eigs.tolist())

    return np.array(gramian_eigs)


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
    args = parser.parse_args()

    plot_save_dir = os.path.join(args.plot_save_dir, args.data_split)
    os.makedirs(plot_save_dir, exist_ok=True)

    save_dir = os.path.join(args.data_save_dir, args.data_split)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(4, 3))
    path = os.path.join(save_dir, "english_springer_gramian_eigenvalues.npy")
    gramian_eigs = np.load(path).flatten()
    plt.hist(gramian_eigs, bins=100, density=True)
    plt.title("Gram Spectrum")
    # plt.xlabel(r"Eigenvalue ($\lambda^{{0.1}}$)")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density (log scale)")
    plt.yscale("log")
    plot_save_path = os.path.join(plot_save_dir, "gram_spectrum.png")
    plt.savefig(plot_save_path)
    print(f"Saved histogram to {plot_save_path}")
    exit()

    class_list = args.target_classes
    data_dir = os.path.join(DATA_DIR, args.data_split)

    print("Setting up dataset...")
    dataset, is_npy_dataset = setup_dataset(data_dir)
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None
    print("Dataset setup complete.")

    # NOTE: has to be ImageFolder to have samples attribute.
    image_paths, targets = zip(*dataset.samples)
    targets = np.array(targets)
    # targets = get_targets(dataset) # for case when dataset is not ImageFolder
    class_to_idx = dataset.class_to_idx
    class_to_indices = {
        cls: np.where(targets == class_to_idx[cls])[0].tolist() for cls in class_list
    }
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    for class_name, indices in tqdm(
        class_to_indices.items(), desc="Processing classes"
    ):
        print(f"Class {class_name} has {len(indices)} samples")

        selected_indices = indices[: args.n_samples_per_class]
        print(f"Selected {len(selected_indices)} samples for class {class_name}")
        class_subset = Subset(dataset, selected_indices)
        dataloader = DataLoader(
            class_subset,
            batch_size=64,
            shuffle=False,  # this needs to be False to match image paths
            num_workers=4,
        )

        print(f"Computing gramian eigenvalues for {class_name}")
        gramian_eigs = compute_gramian_eigenvalues(dataloader, device="cpu")
        save_path = os.path.join(save_dir, f"{class_name}_gramian_eigenvalues.npy")
        np.save(save_path, gramian_eigs)
        print("Gramian eigenvalues shape:", gramian_eigs.shape)
