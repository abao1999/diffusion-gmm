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


@dataclass
class ImageGMM(GaussianMixture):
    """
    Gaussian Mixture Model (GMM) for image data
    Args:
        n_components: The number of mixture components.
        data_dir: The path to the folder to be loaded as imagefolder.
        dataset_name: (Optional) The name of the dataset, to use custom VisionDataset class for loading. For example, Imagenet or CIFAR10.
        covariance_type: The type of covariance matrix to use.
            'full': each component has its own general covariance matrix.
            'tied': all components share the same general covariance matrix.
            'diag': each component has its own diagonal covariance matrix.
            'spherical': each component has its own single variance.
        batch_size: batch size fo dataloader (loading images)
        custom_transform: (Optional) A custom transform to apply to the data.

    ----------------------------------------------------------------
    Attributes:
        From GaussianMixture:
        weights_ : array-like of shape (n_components,)
            The weights of each mixture components.

        means_ : array-like of shape (n_components, n_features)
            The mean of each mixture component.

        covariances_ : array-like
            The covariance of each mixture component.
            The shape depends on `covariance_type`::

                (n_components,)                        if 'spherical',
                (n_features, n_features)               if 'tied',
                (n_components, n_features)             if 'diag',
                (n_components, n_features, n_features) if 'full'

        precisions_ : array-like
            The precision matrices for each component in the mixture. A precision
            matrix is the inverse of a covariance matrix. A covariance matrix is
            symmetric positive definite so the mixture of Gaussian can be
            equivalently parameterized by the precision matrices. Storing the
            precision matrices instead of the covariance matrices makes it more
            efficient to compute the log-likelihood of new samples at test time.
            The shape depends on `covariance_type` in the same way as `covariances_`.

        precisions_cholesky_ : array-like
            The cholesky decomposition of the precision matrices of each mixture
            component.

        converged_ : bool
            True when convergence of the best fit of EM was reached, False otherwise.

        n_iter_ : int
            Number of step used by the best fit of EM to reach the convergence.

        lower_bound_ : float
            Lower bound value on the log-likelihood (of the training data with
            respect to the model) of the best fit of EM.

        n_features_in_ : int
            Number of features seen during :term:`fit`.

        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.

    """

    n_components: int
    data_dir: str
    dataset_name: Optional[str] = None
    covariance_type: str = "full"
    batch_size: int = 32
    custom_transform: Optional[transforms.Compose] = None
    verbose: bool = False
    rseed: int = 99

    def __post_init__(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        super().__init__(n_components=self.n_components)

        self.rng = np.random.default_rng(self.rseed)
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
        self, num_samples: int, target_class: Optional[str] = None, num_workers: int = 4
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
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=custom_sampler,
            num_workers=num_workers,
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

        os.makedirs(save_dir, exist_ok=True)
        print("Saving samples to ", save_dir)

        for i, img in enumerate(samples):
            save_path = os.path.join(save_dir, f"sample_{i}.npy")
            np.save(save_path, img)

        return samples

    # Function to compute mean and covariance matrix from dataset
    def compute_mean_and_covariance(
        self,
        num_samples: int,
        target_class: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and covariance matrix from the dataset by flattening the images from the dataloader
        """
        all_pixels = []
        dataloader = self._build_dataloader(num_samples, target_class)
        # Accumulate all image pixel values
        for idx, (images, _) in tqdm(enumerate(dataloader)):
            # Flatten images to 2D: (number of images, number of pixels per image)
            flattened_images = images.view(images.size(0), -1).numpy()
            all_pixels.append(flattened_images)

        # Concatenate all pixel values
        all_pixels = np.concatenate(all_pixels, axis=0)

        # Compute the mean and covariance
        mean = np.mean(all_pixels, axis=0)
        # get the pixel-wise covariance matrix
        covariance = np.cov(all_pixels, rowvar=False)

        return mean, covariance

    def save_samples_single_class(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        n_samples: int,
        save_dir: str,
    ) -> np.ndarray:
        """
        Generate samples from a single class by simply creating synthetic Gaussian data with same mean and covariance as images in that class
        Saves samples as images with self.img_shape
        Args:
            mean: Mean of the Gaussian distribution
            covariance: Covariance matrix of the Gaussian distribution
            n_samples: Number of samples to generate
            save_dir: Directory to save the samples after converting to images
            plot_kwargs: Keyword arguments for saving and plotting the images
        Returns:
            samples: Generated samples from the fitted GMM
        """

        print("mean shape: ", mean.shape)
        print("covariance shape: ", covariance.shape)
        print("Generating samples...")
        samples = np.random.multivariate_normal(mean, covariance, size=n_samples)

        sample_shape = samples[0].shape

        assert (
            np.prod(self.img_shape) == sample_shape
        ), "New shape does not match the sample shape"

        samples = samples.reshape(-1, *self.img_shape)

        if self.verbose:
            print("Samples shape: ", samples.shape)
            print(f"Saving {n_samples} samples from the fitted GMM to {save_dir}...")

        os.makedirs(save_dir, exist_ok=True)
        print("Saving samples to ", save_dir)

        for i, img in enumerate(samples):
            save_path = os.path.join(save_dir, f"sample_{i}.npy")
            np.save(save_path, img)

        return samples
