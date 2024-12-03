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
    classes: List[str] | str
    covariance_type: str = "full"
    verbose: bool = False
    rseed: int = 99

    def __post_init__(self):
        super().__init__(n_components=self.n_components)

        if isinstance(self.classes, str):
            self.classes = [self.classes]

        sample, _ = self.dataset[0]
        if isinstance(sample, torch.Tensor):
            self.img_shape = sample.shape
        else:
            raise ValueError("Sample is not a tensor")

        self.rng = np.random.default_rng(self.rseed)

        available_classes = self.dataset.classes
        n_classes = len(self.classes)
        if n_classes != self.n_components:
            warnings.warn(
                "Number of classes in classes does not match number of components"
            )
        if n_classes > len(available_classes):
            raise ValueError(
                "Number of classes in classes is greater than number of classes in dataset"
            )

        targets = get_targets(self.dataset)
        class_to_idx = self.dataset.class_to_idx
        self.class_to_indices = {
            cls: np.where(targets == class_to_idx[cls])[0].tolist()
            for cls in self.classes
        }

    def fit(
        self,
        num_samples_per_class: int,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fit a Gaussian Mixture Model (GMM) to the image data, after a data processing step to flatten the images
        """
        selected_inds = []
        for cls in self.classes:
            selected_inds.extend(self.class_to_indices[cls][:num_samples_per_class])

        subset = Subset(self.dataset, selected_inds)

        dataloader = DataLoader(
            subset,
            **(dataloader_kwargs or {}),
        )
        all_images = np.concatenate([images.numpy() for images, _ in tqdm(dataloader)])

        all_images = all_images.reshape(-1, np.prod(self.img_shape))
        logger.info(f"Fitting GMM to samples of shape {all_images.shape}")
        super().fit(all_images)

    def compute_mean_and_covariance(
        self,
        num_samples_per_class: int,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute the mean and covariance matrix from the dataset by flattening the images from the dataloader
        """
        class_stats = {}
        selected_inds = []
        for class_name in self.classes:
            logger.info(f"Computing mean and covariance for class {class_name}")
            class_stats[class_name] = {
                "mean": None,
                "covariance": None,
            }
            selected_inds.extend(
                self.class_to_indices[class_name][:num_samples_per_class]
            )

            subset = Subset(self.dataset, selected_inds)

            dataloader = DataLoader(
                subset,
                batch_size=64,
                **(dataloader_kwargs or {}),
            )
            all_pixels = np.concatenate(
                [images.numpy() for images, _ in tqdm(dataloader)]
            )
            all_pixels = all_pixels.reshape(-1, np.prod(self.img_shape))
            logger.info(f"all_pixels.shape: {all_pixels.shape}")
            mean_all_pixels = np.mean(all_pixels, axis=0)
            cov_all_pixels = np.cov(all_pixels, rowvar=False)
            logger.info(f"cov_all_pixels shape: {cov_all_pixels.shape}")
            logger.info(f"mean_all_pixels shape: {mean_all_pixels.shape}")
            class_stats[class_name]["mean"] = mean_all_pixels
            class_stats[class_name]["covariance"] = cov_all_pixels

        return class_stats

    def sample_and_save(
        self,
        n_samples: int,
        save_dir: str,
        sample_idx: int = 0,
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
        for start_idx in range(sample_idx, n_samples + sample_idx, batch_size):
            end_idx = min(start_idx + batch_size, n_samples + sample_idx)
            batch_samples = samples[start_idx:end_idx]
            save_path = os.path.join(
                save_dir, f"batch_{start_idx:06d}_{end_idx:06d}.npz"
            )
            np.savez(save_path, *batch_samples)

    def sample_from_computed_stats(
        self,
        class_stats: Dict[str, Dict[str, np.ndarray]],
        n_samples_per_class: int,
        save_dir: str,
        batch_size: Optional[int] = None,
        sample_idx: int = 0,
    ) -> None:
        """
        Generate Gaussian samples with the computed mean and covariance matrices and save them as images with self.img_shape
        """
        if not set(class_stats.keys()).issubset(self.classes):
            raise ValueError(
                "Class list in class_stats is not a subset of the class list"
            )
        if batch_size is None:
            batch_size = n_samples_per_class
        for class_name in self.classes:
            mean = class_stats[class_name]["mean"]
            covariance = class_stats[class_name]["covariance"]

            os.makedirs(save_dir, exist_ok=True)

            # Sample in batches
            for start_idx in tqdm(
                range(sample_idx, n_samples_per_class + sample_idx, batch_size),
                desc=f"Sampling from class {class_name}",
            ):
                end_idx = min(start_idx + batch_size, n_samples_per_class + sample_idx)
                batch_samples = self.rng.multivariate_normal(
                    mean, covariance, size=(end_idx - start_idx)
                )

                assert (
                    np.prod(self.img_shape) == batch_samples[0].shape
                ), "New shape does not match the sample shape"

                batch_samples = batch_samples.reshape(-1, *self.img_shape)
                logger.info(
                    f"Saving {end_idx - start_idx} samples, shape {batch_samples.shape}, computed from class {class_name}, to {save_dir}..."
                )

                # Save samples in batches
                batch_save_paths = [
                    os.path.join(save_dir, f"sample_{i:06d}.npy")
                    for i in range(start_idx, end_idx)
                ]
                for img, save_path in zip(batch_samples, batch_save_paths):  # type: ignore
                    np.save(save_path, img)
