import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from diffusion_gmm.base import DataPrefetcher
from diffusion_gmm.utils import get_targets, setup_dataset

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")


def build_dataloader_for_class(
    dataset: Dataset,
    target_class: str,
    batch_size: int = 64,
    num_workers: int = 4,
    num_samples: int = 1000,
    make_prefetcher: bool = False,
    device: str = "cpu",
) -> DataLoader | DataPrefetcher:
    """
    Build a DataLoader for a specific class from the dataset.
    """
    targets = get_targets(dataset)  # type: ignore
    class_to_idx = dataset.class_to_idx  # type: ignore
    if target_class not in class_to_idx:
        raise ValueError(f"Class {target_class} not found in dataset.")

    target_class_idx = class_to_idx[target_class]
    indices = []
    for idx, class_idx in enumerate(targets):
        if len(indices) == num_samples:
            break
        if class_idx == target_class_idx:
            indices.append(idx)
    if len(indices) < num_samples:
        warnings.warn(
            f"Warning: Requested {num_samples} samples, but only {len(indices)} available for class '{target_class}'. Using available samples."
        )

    selected_subset = Subset(dataset, indices)

    dataloader = DataLoader(
        selected_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    if make_prefetcher:
        dataloader = DataPrefetcher(dataloader, device)

    return dataloader


def compute_norm_of_images(
    dataloader: DataLoader | DataPrefetcher,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute the norm of all images for a specific class.
    """
    image_norms = []

    for images, _ in tqdm(dataloader, desc="Computing norms"):
        images = images.to(device)
        norms = torch.norm(images.view(images.size(0), -1), p=2, dim=1)
        image_norms.extend(norms.tolist())

    return np.array(image_norms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, required=True)
    # parser.add_argument("--target_class", type=str, required=True)
    parser.add_argument("--target_classes", type=str, nargs="+", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    data_dir = os.path.join(DATA_DIR, args.data_split)
    dataset, is_npy_dataset = setup_dataset(data_dir)
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None

    fig = plt.figure(figsize=(6, 6))
    for target_class in args.target_classes:
        dataloader = build_dataloader_for_class(
            dataset, target_class, num_samples=args.num_samples
        )
        image_norms = compute_norm_of_images(dataloader)
        print(f"Average norm of {target_class} images': {np.mean(image_norms)}")

        plt.hist(image_norms, bins=100, density=False, label=target_class, alpha=0.5)
        plt.title(f"Norms of {args.dataset_name} images")
        plt.xlabel("Norm")
        plt.xlim(0, 120)
        plt.ylabel("Count")

    plt.legend()
    plt.savefig(f"plots/{args.data_split}_norms.png", dpi=300)
    plt.close()
