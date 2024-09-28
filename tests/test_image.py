"""
Plot a sample image from an ImageFolder dataset
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")
FIGS_DIR = "tests/figs"


def plot_imagefolder_sample(transform: transforms.Compose, dataset_name: str):
    """
    Plot a sample image from an ImageFolder dataset
    """
    img_dir = os.path.join(DATA_DIR, dataset_name)
    print(f"Loading image from {img_dir}")
    data = datasets.ImageFolder(root=img_dir, transform=transform)

    # Create a DataLoader to load the images
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    # Get a single image from the dataset
    sample = next(iter(dataloader))
    images, label = sample
    print(f"Label: {label}")

    image = images[0]  # First image in the batch
    print("Image shape:", image.shape)

    mean = image.mean()
    std = image.std()
    print("Image mean:", mean)
    print("Image std:", std)

    # Convert the image to a numpy array and plot it
    image_np = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.axis("off")
    save_path = os.path.join(FIGS_DIR, f"{dataset_name}.png")
    print(f"Saving image to {save_path}")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a sample image from an ImageFolder dataset"
    )
    parser.add_argument("dataset_name", type=str, help="Name of data subdirectory")
    args = parser.parse_args()

    dataset_name = args.dataset_name

    os.makedirs(FIGS_DIR, exist_ok=True)
    # set random seed
    rseed = 999
    np.random.seed(rseed)
    torch.manual_seed(rseed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    plot_imagefolder_sample(transform, dataset_name=dataset_name)
