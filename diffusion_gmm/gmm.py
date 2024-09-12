from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
import numpy as np
import os
from typing import Optional, Tuple
from PIL import Image
from tqdm.auto import tqdm

from diffusion_gmm.utils import save_images_grid


class ImageGMM():
    """
    Class to fit a Gaussian Mixture Model (GMM) to image data
    This is a wrapper around the scikit-learn GaussianMixture class
    """
    def __init__(
        self, 
        dataloader: DataLoader,
        img_shape: Optional[Tuple[int, int, int]] = None,
        n_components: int = 5, 
        verbose: bool = False,
    ):
        self.dataloader = dataloader
        self.img_shape = img_shape
        self.n_components = n_components
        self.verbose = verbose
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.mean = None # to be set after fitting
        self.covariance = None # to be set after fitting

    # Function to compute mean and covariance matrix from dataset
    def _compute_mean_and_covariance(
        self, 
        num_images: int = 1024,
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
        covariance = np.cov(all_pixels, rowvar=False)

        self.mean = mean
        self.covariance = covariance
        
        return mean, covariance

    def fit(
        self,
        n_samples_compute_stats: int = 1024,
        n_samples_fit: int = 1024,
    ) -> GaussianMixture:
        """
        Fit a Gaussian Mixture Model (GMM)
        Args:
            n_samples_compute_stats: Number of samples to compute mean and covariance
            n_samples_fit: Number of samples to fit the GMM
        """
        # Computed mean and covariance of samples from image loader, to use for fitting gmm
        mean, covariance = self._compute_mean_and_covariance(n_samples_compute_stats)
        print("mean: ", mean)
        print("covariance: ", covariance)
        print("mean mean: ", np.mean(mean))
        print("mean covariance: ", np.mean(covariance))
        
        if self.verbose:
            print(f"Fitting GMM with {self.n_components} components and {n_samples_fit} samples...")
            print("Mean shape:", mean.shape)
            print("Covariance shape:", covariance.shape) 

        # Generate synthetic data from the mean and covariance to fit the GMM
        synthetic_data = np.random.multivariate_normal(mean, covariance, size=n_samples_fit)
        self.gmm.fit(synthetic_data)

    def __call__(
        self,
        n_samples_compute_stats: int = 1024,
        n_samples_fit: int = 1024,
        run_name: str = 'gmm',
    ) -> GaussianMixture:
        self.fit(n_samples_compute_stats, n_samples_fit)
        self.save_samples(
            n_samples=100, #n_samples_gmm, 
            save_fig_dir='figs',
            save_grid_shape=(10, 10),
            save_name=run_name,
        )

    def save_samples(
        self,
        n_samples: int = 1024, 
        save_fig_dir: str = 'figs',
        save_grid_shape: Optional[Tuple[int, int]] = None,
        save_name: str = 'gmm',
    ):
        """
        Generate samples from the fitted GMM and save them as images with self.img_shape
        Args:
            gmm: Fitted Gaussian Mixture Model, which generates (flattened) samples
            n_samples: Number of samples to generate
            save_fig_dir: Directory to save the samples after converting to images
            save_name: Name of the file to save the samples
        """
        os.makedirs(save_fig_dir, exist_ok=True)
        
        samples_flattened, _ = self.gmm.sample(n_samples)
        flattened_sample_shape = samples_flattened[0].shape
        assert np.prod(self.img_shape) == flattened_sample_shape, "New shape does not match the sample shape"

        if self.verbose:
            print("Sample shape: ", flattened_sample_shape)
            print(f"Saving {n_samples} samples from the fitted GMM to {save_fig_dir}...")

        samples = samples_flattened.reshape(-1, *self.img_shape)

        # Convert the samples to images and save
        if save_grid_shape is not None:
            save_images_grid(
                samples, 
                os.path.join(save_fig_dir, f"{save_name}_sample_grid.png"), 
                grid_shape=save_grid_shape,
            )
        else:
            # processed_samples = np.clip(samples, -1, 1)
            # processed_samples = ((processed_samples + 1) / 2) * 255
            # processed_samples = processed_samples.astype(np.uint8)
            for i, img in tqdm(enumerate(samples)):
                img = np.transpose(img, (1, 2, 0))  # Reorder dimensions to HWC
                img = (img * 255).astype(np.uint8)  # Convert to uint8
                img = Image.fromarray(img)
                img.save(os.path.join(save_fig_dir, f"{save_name}_sample_{i}.png"))
        
        return samples