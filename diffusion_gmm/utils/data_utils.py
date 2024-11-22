import copy
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets as torchvision_datasets
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)


def set_seed(rseed: int):
    """
    Set the seed for the random number generator for torch, cuda, and cudnn
    """
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

    torch.manual_seed(rseed)
    # If using CUDA, you should also set the seed for CUDA for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rseed)
        torch.cuda.manual_seed_all(rseed)  # if you have multiple GPUs

    # For deterministic behavior on GPU (reproducibility), use the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


def get_targets(dataset: DatasetFolder) -> np.ndarray:
    # For datasets like CIFAR10, use targets attribute
    if hasattr(dataset, "targets"):
        targets = dataset.targets  # type: ignore
    # For ImageFolder, reconstruct targets from samples
    elif hasattr(dataset, "samples"):
        targets = [class_idx for _, class_idx in dataset.samples]  # type: ignore
    else:
        raise AttributeError("Dataset doesn't have 'targets' or 'samples' attribute")
    return np.array(targets)


def setup_dataset(
    data_dir: str,
    dataset_name: Optional[str] = None,
) -> Tuple[DatasetFolder, bool]:
    # Check if the directory contains any .npy files
    is_npy_dataset = any(
        fname.endswith(".npy")
        for root, dirs, files in os.walk(data_dir)
        for fname in files
    )
    dataset_cls = DatasetFolder if is_npy_dataset else ImageFolder
    if dataset_name is not None:
        dataset_cls = getattr(torchvision_datasets, dataset_name)
    loader = npy_loader if is_npy_dataset else default_loader
    dataset = dataset_cls(
        root=data_dir,
        loader=loader,
        is_valid_file=lambda path: path.endswith(".npy") if is_npy_dataset else True,
    )
    return dataset, is_npy_dataset


def split_dataset_balanced(
    dataset: DatasetFolder,
    class_list: List[str],
    max_allowed_samples_per_class: Optional[int] = None,
    train_split: float = 0.8,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[Subset, Optional[Subset]]:
    """
    Get the indices for the training and test subsets
    """
    targets = get_targets(dataset)
    class_to_idx = dataset.class_to_idx
    class_to_indices = {
        cls: np.where(targets == class_to_idx[cls])[0].tolist() for cls in class_list
    }

    if rng is not None:
        for indices in class_to_indices.values():
            rng.shuffle(indices)

    n_samples_per_class = min(len(indices) for indices in class_to_indices.values())
    if max_allowed_samples_per_class is not None:
        n_samples_per_class = min(n_samples_per_class, max_allowed_samples_per_class)

    if verbose:
        logger.info("number of samples per class to use: %d", n_samples_per_class)

    train_size_per_class = int(n_samples_per_class * train_split)
    balanced_train_inds = []
    balanced_test_inds = []
    for indices in class_to_indices.values():
        balanced_train_inds.extend(indices[:train_size_per_class])
        balanced_test_inds.extend(indices[train_size_per_class:n_samples_per_class])

    train_subset = Subset(dataset, balanced_train_inds)

    if train_split < 1.0:
        # default behavior is to clone the datasets so train_subset and test_subset have their own instance
        dataset_copy = copy.deepcopy(dataset)
        test_subset = Subset(dataset_copy, balanced_test_inds)
    else:
        test_subset = None
    return train_subset, test_subset


def make_balanced_subsets(
    class_list: List[str],
    data_dir: str,
    max_allowed_samples_per_class: Optional[int],
    train_split: float = 0.8,
    use_augmentations: bool = False,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[Subset, Optional[Subset]]:
    dataset, is_npy_dataset = setup_dataset(data_dir)
    train_subset, test_subset = split_dataset_balanced(
        dataset,
        class_list=class_list,
        max_allowed_samples_per_class=max_allowed_samples_per_class,
        train_split=train_split,
        rng=rng,
        verbose=verbose,
    )

    if is_npy_dataset:
        return train_subset, test_subset

    if test_subset is not None:
        # Define the basic transformation for the test dataset
        test_transform = transforms.ToTensor()
        test_subset.dataset.transform = test_transform  # type: ignore

    # Define the data augmentation pipeline for the train dataset
    if use_augmentations:
        # set up augmentations (for train subset only)
        sample, _ = train_subset.dataset[0]
        if isinstance(sample, torch.Tensor):
            # If the sample is already a tensor (e.g., from npy files)
            img_shape = sample.shape
        else:
            # If the sample is a PIL image, convert it to a tensor first
            sample_tensor = transforms.ToTensor()(sample)
            img_shape = sample_tensor.shape

        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(img_shape[1:], scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),  # Ensure the data is converted to a tensor
            ]
        )
    else:
        train_transform = transforms.ToTensor()

    train_subset.dataset.transform = train_transform  # type: ignore
    return train_subset, test_subset


def validate_subsets(train_subset: Subset, test_subset: Subset):
    """
    Verify that there are no shared indices between the train and test subsets
    """
    # train_inds = [i for i, _ in train_subset]
    # test_inds = [i for i, _ in test_subset]
    train_inds = train_subset.indices
    test_inds = test_subset.indices
    if set(train_inds) & set(test_inds):
        raise ValueError("Overlap detected between train and test indices.")
