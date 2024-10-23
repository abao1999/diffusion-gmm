import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torchvision.transforms as transforms
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.datasets import ImageFolder, VisionDataset
from tqdm.auto import tqdm

from diffusion_gmm.utils import save_and_plot_samples


@dataclass
class ImageGMM(GaussianMixture):
    """
    Gaussian Mixture Model (GMM) for image data
    """

    n_components: int
    datapath: str
    dataset_name: Optional[str] = None
    batch_size: int = 32
    custom_transform: Optional[transforms.Compose] = None
    verbose: bool = False

    def __post_init__(self):
        super().__init__(n_components=self.n_components)
        if self.custom_transform is None:
            self.custom_transform = transforms.Compose([transforms.ToTensor()])
        self.data = self._get_dataset()
        self.classes = self.data.classes  # type: ignore
        self.class_to_idx: Dict[str, int] = self.data.class_to_idx  # type: ignore
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.img_shape = self._get_sample_shape()

    def _get_dataset(self) -> VisionDataset:
        if self.dataset_name is None:
            data = ImageFolder(
                root=self.datapath,
                transform=self.custom_transform,
                target_transform=None,
            )
        elif self.dataset_name == "cifar10":
            # Load the real CIFAR10 train split from torchvision
            data = datasets.CIFAR10(
                root=self.datapath,
                train=True,
                download=True,
                transform=self.custom_transform,
            )
        elif self.dataset_name == "imagenet":
            data = datasets.ImageNet(
                root=self.datapath,
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
        self, num_samples: int, target_class: Optional[str] = None
    ) -> DataLoader:
        num_tot_samples = len(self.data)
        if target_class is None:
            if num_samples > num_tot_samples:
                indices = list(range(num_tot_samples))
            else:
                indices = np.random.choice(
                    num_tot_samples, num_samples, replace=False
                ).tolist()
        else:
            # Convert target_class from string to integer index
            target_class_idx = self.class_to_idx[target_class]

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

            indices = []
            for idx, class_idx in enumerate(targets):
                if len(indices) == num_samples:
                    break
                if class_idx == target_class_idx:
                    indices.append(idx)

        if self.verbose:
            print(f"Sampling from {len(indices)} valid samples")

        custom_sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(
            self.data, batch_size=self.batch_size, shuffle=False, sampler=custom_sampler
        )
        return dataloader

    def fit(self, num_samples: int, target_class: Optional[str] = None) -> None:
        """
        Fit a Gaussian Mixture Model (GMM) to the image data, after a data processing step to flatten the images
        """
        dataloader = self._build_dataloader(num_samples, target_class)
        all_images = np.concatenate([images.numpy() for images, _ in tqdm(dataloader)])
        all_images = all_images.reshape(-1, np.prod(self.img_shape))
        print("Fitting GMM to ", all_images.shape, " samples")
        super().fit(all_images)

    def save_samples(
        self,
        n_samples: int,
        save_dir: str,
        plot_kwargs: dict = {},
    ) -> np.ndarray:
        """
        Generate samples from the fitted GMM and save them as images with self.img_shape
        """
        os.makedirs(save_dir, exist_ok=True)
        # NOTE: this also returns component_labels
        samples, _ = super().sample(n_samples)
        _, n_features = samples.shape
        assert (
            np.prod(self.img_shape) == n_features
        ), "Mismatch between the number of features and the image shape"

        samples = samples.reshape(-1, *self.img_shape)
        if self.verbose:
            print("Samples shape: ", samples.shape)
            print(f"Saving {n_samples} samples from the fitted GMM to {save_dir}...")

        save_and_plot_samples(
            samples,
            save_dir,
            **plot_kwargs,
        )

        return samples
