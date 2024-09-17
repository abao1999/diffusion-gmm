from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


def custom_dataset(
    root: str, transform: Callable[[Image.Image], torch.Tensor]
) -> Dataset:
    """
    Custom loader for Dataset, for loading images from a directory
    Currently, the custom loader is defined to be pil_loader, but this can be swapped out
    """

    def pil_loader(folder_path):
        return Image.open(folder_path).convert("RGB")

    # custom Dataset class to load images from a directory without requiring class subdirectories
    data = datasets.DatasetFolder(
        root=root,
        loader=pil_loader,
        extensions=("png", "jpg", "jpeg"),
        transform=transform,
    )
    return data


def get_imagenet_data(
    root: str, transform: Callable[[Image.Image], torch.Tensor]
) -> Dataset:
    """
    Load ImageNet data (or a subset, like ImageNet's validation set)
    """
    data = datasets.ImageNet(root=root, transform=transform)
    return data


def get_mnist_data(
    root: str, transform: Callable[[Image.Image], torch.Tensor]
) -> Dataset:
    """
    Load MNIST data
    """
    data = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return data


def get_cifar10_data(
    root: str, transform: Callable[[Image.Image], torch.Tensor]
) -> Dataset:
    """
    Load CIFAR10 data
    """
    data = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return data
