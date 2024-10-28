import os
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset


@dataclass
class ImageExperiment:
    data_dir: str
    dataset_name: Optional[str] = None
    batch_size: int = 64
    custom_transform: Optional[Callable] = None
    verbose: bool = False
    load_npy: bool = False
    rseed: int = 99

    def __post_init__(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        self.rng = np.random.default_rng(self.rseed)

        if self.custom_transform is None:
            self.custom_transform = transforms.Compose([transforms.ToTensor()])

        self.data = self._get_dataset()
        self.classes = self.data.classes  # type: ignore
        self.class_to_idx: Dict[str, int] = self.data.class_to_idx  # type: ignore
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.img_shape = self._get_sample_shape()

        if self.verbose:
            print("Data directory: ", self.data_dir)
            print("Using custom transform: ", self.custom_transform)
            print("Transform: ", self.custom_transform)

    def _get_dataset(self) -> VisionDataset:
        if self.dataset_name is None:
            if self.load_npy:

                def npy_loader(path):
                    sample = torch.from_numpy(np.load(path))
                    return sample

                data = DatasetFolder(
                    root=self.data_dir,
                    loader=npy_loader,
                    extensions=(".npy",),
                )
            else:
                data = ImageFolder(
                    root=self.data_dir,
                    transform=self.custom_transform,
                    target_transform=None,
                )

        elif self.dataset_name == "cifar10":
            # Load the real CIFAR10 train split from torchvision
            data = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.custom_transform,
            )
        elif self.dataset_name == "imagenet":
            data = datasets.ImageNet(
                root=self.data_dir,
                split="train",
                download=True,
                transform=self.custom_transform,
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

        return data

    def _get_sample_shape(self) -> Tuple[int, int, int]:
        first_sample = next(iter(DataLoader(self.data)))[0].squeeze()
        shape = tuple(first_sample.shape)
        print("Image shape: ", shape)
        assert len(shape) == 3, "Images should have 3 dimensions (C, H, W)"
        if not shape[0] == 3:
            warnings.warn(
                f"Images should have 3 color channels. Found {shape[0]} instead."
            )
        return shape

    def _build_dataloader(
        self, num_samples: int, target_class: Optional[Union[str, List[str]]] = None
    ) -> DataLoader:
        num_tot_samples = len(self.data)
        if target_class is None:
            if num_samples > num_tot_samples:
                indices = list(range(num_tot_samples))
            else:
                indices = self.rng.choice(
                    num_tot_samples, num_samples, replace=False
                ).tolist()
        else:
            # Ensure target_class is a list
            if isinstance(target_class, str):
                target_class = [target_class]

            # Convert target_class from string to integer index
            target_class_indices = [self.class_to_idx[cls] for cls in target_class]

            # For datasets like CIFAR10, use targets attribute
            if hasattr(self.data, "targets"):
                targets = self.data.targets  # type: ignore
            # For ImageFolder, reconstruct targets from samples
            elif hasattr(self.data, "samples"):
                targets = [class_idx for _, class_idx in self.data.samples]  # type: ignore
            else:
                raise AttributeError(
                    "Dataset doesn't have 'targets' or 'samples' attribute"
                )

            # TODO: change this to use Subset on the dataset, to subset the data into selected classes
            # and use weighted subset sampler
            indices = []
            class_counts = {cls: 0 for cls in target_class_indices}
            max_samples_per_class = num_samples // len(target_class_indices)

            for idx, class_idx in enumerate(targets):
                if all(
                    count == max_samples_per_class for count in class_counts.values()
                ):
                    break
                if (
                    class_idx in target_class_indices
                    and class_counts[class_idx] < max_samples_per_class
                ):
                    indices.append(idx)
                    class_counts[class_idx] += 1

        if self.verbose:
            print(f"Sampling from {len(indices)} valid samples")

        custom_sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(
            self.data, batch_size=self.batch_size, shuffle=False, sampler=custom_sampler
        )
        return dataloader
