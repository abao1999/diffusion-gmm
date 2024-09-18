import os
from typing import Tuple, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusion_gmm.utils import save_and_plot_samples


class ImageGMM:
    """
    Class to fit a Gaussian Mixture Model (GMM) to image data
    This is a wrapper around the scikit-learn GaussianMixture class
    """

    def __init__(
        self,
        dataloader: DataLoader,
        img_shape: Tuple[int, int, int],
        n_components: int = 5,
        verbose: bool = False,
    ):
        self.dataloader = dataloader
        self.img_shape = img_shape
        self.verbose = verbose
        self.gmm = GaussianMixture(n_components=n_components, covariance_type="full")
        self.mean = None  # to be set after fitting
        self.covariance = None  # to be set after fitting

    def fit(
        self,
    ) -> None:
        """
        Fit a Gaussian Mixture Model (GMM)
        """
        all_images = []
        # Accumulate all image pixel values
        for idx, (images, _) in tqdm(enumerate(self.dataloader)):
            if idx == 0 and self.img_shape is None:
                self.img_shape = images.shape[1:]
                if self.verbose:
                    print("Image shape: ", self.img_shape)
            images = images.numpy()
            all_images.append(images)

        # Concatenate all pixel values
        all_images = np.concatenate(all_images, axis=0)
                # Flatten images to 2D: (number of images, number of pixels per image)
        all_images = all_images.reshape(-1, np.prod(self.img_shape))
        print("all_images shape: ", all_images.shape)


        self.gmm.fit(all_images)  # NOTE: fit on flattened cifar10 images

    # Function to compute mean and covariance matrix from dataset
    def set_mean_and_covariance(
        self,
        num_images: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and covariance matrix from the dataset by flattening the images from the dataloader
        """
        all_pixels = []

        if self.verbose:
            print(f"Computing mean and covariance from {num_images} images...")

        # Accumulate all image pixel values
        for idx, (images, _) in tqdm(enumerate(self.dataloader)):
            if idx == 0 and self.img_shape is None:
                self.img_shape = images.shape[1:]
                if self.verbose:
                    print("Image shape: ", self.img_shape)
            # if (idx * images.shape[0]) >= num_images:
            #     break
            # Flatten images to 2D: (number of images, number of pixels per image)
            flattened_images = images.view(images.size(0), -1).numpy()
            all_pixels.append(flattened_images)

        # Concatenate all pixel values
        all_pixels = np.concatenate(all_pixels, axis=0)

        # Compute the mean and covariance
        mean = np.mean(all_pixels, axis=0)
        # get the pixel-wise covariance matrix
        covariance = np.cov(all_pixels, rowvar=False)

        self.mean = mean
        self.covariance = covariance

        return mean, covariance

    def save_samples_single_class(
        self,
        n_samples: int,
        save_dir: str,
        plot_kwargs: dict = {},
    ):
        mean, covariance = self.mean, self.covariance
        if mean is None or covariance is None:
            mean, covariance = self.set_mean_and_covariance(n_samples)
        # Generate synthetic data from the mean and covariance to fit the GMM
        samples = np.random.multivariate_normal(
            mean, covariance, size=n_samples
        )

        sample_shape = samples[0].shape
        assert (
            np.prod(self.img_shape) == sample_shape
        ), "New shape does not match the sample shape"

        samples = samples.reshape(-1, *self.img_shape)

        if self.verbose:
            print("Samples shape: ", samples.shape)
            print(f"Saving {n_samples} samples from the fitted GMM to {save_dir}...")

        # Save and plot the samples
        save_and_plot_samples(
            samples,
            save_dir,
            **plot_kwargs,
        )

        return samples
        


    def save_samples(
        self,
        n_samples: int,
        save_dir: str,
        plot_kwargs: dict = {},
    ):
        """
        Generate samples from the fitted GMM and save them as images with self.img_shape
        Args:
            gmm: Fitted Gaussian Mixture Model, which generates (flattened) samples
            n_samples: Number of samples to generate
            save_dir: Directory to save the samples after converting to images
        """
        os.makedirs(save_dir, exist_ok=True)

        # Generate samples from the fitted GMM
        samples, _ = self.gmm.sample(n_samples)
        
        # flattened_sample_shape = samples_flattened[0].shape
        sample_shape = samples[0].shape
        assert (
            np.prod(self.img_shape) == sample_shape
        ), "New shape does not match the sample shape"

        samples = samples.reshape(-1, *self.img_shape)

        if self.verbose:
            print("Samples shape: ", samples.shape)
            print(f"Saving {n_samples} samples from the fitted GMM to {save_dir}...")

        # Save and plot the samples
        save_and_plot_samples(
            samples,
            save_dir,
            **plot_kwargs,
        )

        return samples
