import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

from diffusion_gmm.utils.data_utils import get_targets

logger = logging.getLogger(__name__)


@dataclass
class ImageGMM(GaussianMixture):
    """
    Gaussian Mixture Model (GMM) for image data
    """

    n_components: int
    dataset: DatasetFolder
    class_list: List[str]
    covariance_type: str = "full"
    verbose: bool = False
    rseed: int = 99

    def __post_init__(self):
        super().__init__(n_components=self.n_components)

        sample, _ = self.dataset[0]
        if isinstance(sample, torch.Tensor):
            self.img_shape = sample.shape
        else:
            raise ValueError("Sample is not a tensor")

        self.rng = np.random.default_rng(self.rseed)

        available_classes = self.dataset.classes
        n_classes = len(self.class_list)
        if n_classes != self.n_components:
            warnings.warn(
                "Number of classes in class_list does not match number of components"
            )
        if n_classes > len(available_classes):
            raise ValueError(
                "Number of classes in class_list is greater than number of classes in dataset"
            )

        targets = get_targets(self.dataset)
        class_to_idx = self.dataset.class_to_idx
        self.class_to_indices = {
            cls: np.where(targets == class_to_idx[cls])[0].tolist()
            for cls in self.class_list
        }

    def fit(
        self,
        num_samples_per_class: int,
        use_dataloader: bool = False,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fit a Gaussian Mixture Model (GMM) to the image data, after a data processing step to flatten the images
        """
        selected_inds = []
        for cls in self.class_list:
            selected_inds.extend(self.class_to_indices[cls][:num_samples_per_class])

        subset = Subset(self.dataset, selected_inds)

        if use_dataloader:
            dataloader = DataLoader(
                subset,
                **(dataloader_kwargs or {}),
            )
            all_images = np.concatenate(
                [images.numpy() for images, _ in tqdm(dataloader)]
            )
        else:
            all_images = np.concatenate(
                [self.dataset[i][0].numpy() for i in selected_inds]
            )

        all_images = all_images.reshape(-1, np.prod(self.img_shape))
        logger.info(f"Fitting GMM to samples of shape {all_images.shape}")
        super().fit(all_images)

    def compute_mean_and_covariance(
        self,
        num_samples_per_class: int,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute the mean and covariance matrix from the dataset by flattening the images from the dataloader
        """
        class_stats = {}
        for class_name in self.class_list:
            class_stats[class_name] = {
                "mean": None,
                "covariance": None,
            }
            sel_inds = self.class_to_indices[class_name][:num_samples_per_class]
            all_pixels = np.array(
                [self.dataset[i][0].numpy().reshape(-1) for i in sel_inds]
            )

            class_stats[class_name]["mean"] = np.mean(all_pixels, axis=0)
            class_stats[class_name]["covariance"] = np.cov(all_pixels, rowvar=False)

        return class_stats

    def save_samples(
        self,
        n_samples: int,
        save_dir: str,
    ) -> None:
        """
        Generate samples from the fitted GMM and save them as images with self.img_shape
        """
        os.makedirs(save_dir, exist_ok=True)
        # NOTE: this also returns component_labels
        samples, _ = super().sample(n_samples)
        assert (
            np.prod(self.img_shape) == samples.shape[1]
        ), "Mismatch between the number of features and the image shape"

        samples = samples.reshape(-1, *self.img_shape)
        if self.verbose:
            logger.info(
                f"Saving {n_samples} samples, shape {samples.shape}, from the fitted GMM to {save_dir}..."
            )

        # Save samples in batches
        batch_size = 128
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_samples = samples[start_idx:end_idx]
            save_path = os.path.join(save_dir, f"batch_{start_idx:06d}.npz")
            np.savez(save_path, *batch_samples)

    def sample_from_computed_stats(
        self,
        class_stats: Dict[str, Dict[str, np.ndarray]],
        n_samples_per_class: int,
        save_dir: str,
    ) -> None:
        """
        Generate Gaussian samples with the computed mean and covariance matrices and save them as images with self.img_shape
        """
        if not set(class_stats.keys()).issubset(self.class_list):
            raise ValueError(
                "Class list in class_stats is not a subset of the class list"
            )
        for class_name in self.class_list:
            mean = class_stats[class_name]["mean"]
            covariance = class_stats[class_name]["covariance"]
            samples = self.rng.multivariate_normal(
                mean, covariance, size=n_samples_per_class
            )

            assert (
                np.prod(self.img_shape) == samples[0].shape
            ), "New shape does not match the sample shape"

            samples = samples.reshape(-1, *self.img_shape)
            logger.info(
                f"Saving {n_samples_per_class} samples, shape {samples.shape}, computed from class statistics, to {save_dir}..."
            )
            class_save_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_save_dir, exist_ok=True)

            # Save samples in batches
            batch_size = 128
            for start_idx in range(0, n_samples_per_class, batch_size):
                end_idx = min(start_idx + batch_size, n_samples_per_class)
                batch_samples = samples[start_idx:end_idx]
                batch_save_paths = [
                    os.path.join(class_save_dir, f"sample_{i:06d}.npy")
                    for i in range(start_idx, end_idx)
                ]
                for img, save_path in zip(batch_samples, batch_save_paths):
                    np.save(save_path, img)
