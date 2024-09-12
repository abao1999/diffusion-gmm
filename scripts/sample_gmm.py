import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

from diffusion_gmm.gmm import ImageGMM

from diffusion_gmm.utils import save_images_grid, plot_pixel_intensity_hist

import matplotlib.pyplot as plt


WORK_DIR = os.getenv('WORK')
DATA_DIR = os.path.join(WORK_DIR, 'vision_datasets')


if __name__ == '__main__':

    use_generated_data = False
    batch_size = 1

    # for gmm fitting
    n_for_stats = 1024
    n_for_fit = 1024

    # for generating samples
    dataset_name = 'cifar10'
    cifar10_shape = (3, 32, 32)
    n_samples_generate = 1024

    # Define the transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the dataset
    if use_generated_data:
        # load generated images
        image_dir = os.path.join(DATA_DIR, f"diffusion_{dataset_name}")
        dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    else:
        # load real images
        image_dir = os.path.join(DATA_DIR, dataset_name)
        dataset = datasets.CIFAR10(
            root=os.path.join(DATA_DIR, 'cifar10'), 
            train=False, 
            download=True, 
            transform=transform
        )

        # # Limit the number of samples
        # num_samples = 1000
        # dataset.data = dataset.data[:num_samples]
        # dataset.targets = dataset.targets[:num_samples]

    print("Image directory:", image_dir)

    n_tot_images = len(dataset)
    if n_for_stats > n_tot_images:
        print(f"Warning: Only {n_tot_images} images found in the dataset. Using all available images.")
        n_for_stats = n_tot_images

    custom_sampler = SubsetRandomSampler(range(n_for_stats))
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        sampler=custom_sampler # if custom_sampler else None
    )

    gmm = ImageGMM(
        dataloader=dataloader,
        img_shape=cifar10_shape, 
        n_components=10, 
        verbose=True,
    )

    # gmm(n_samples_compute_stats=1024, n_samples_fit=1024, run_name='gmm_cifar10')

    gmm.fit(n_samples_compute_stats=n_for_stats, n_samples_fit=n_for_fit)
    print("GMM fitted successfully.")

    mean = gmm.mean
    covariance = gmm.covariance
    # plot histogram of per-pixel mean and covariance
    print("mean shape: ", mean.shape)
    print("mean mean: ", np.mean(mean))
    print("covariance shape: ", covariance.shape)
    print("mean covariance: ", np.mean(covariance))

    plt.figure(figsize=(10, 6))
    plt.hist(mean, bins=100)
    plt.title("Mean Pixel Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('figs/mean_pixel_values.png', dpi=300)

    plt.figure(figsize=(10, 6))
    plt.hist(covariance.flatten(), bins=100)
    plt.title("Covariance Pixel Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('figs/covariance_pixel_values.png', dpi=300)

    # Save the samples generated from the fitted GMM
    save_dir = os.path.join(DATA_DIR, 'gmm_cifar10', 'unknown')
    save_name = f"gmm_{dataset_name}"
    print(f"Saving samples generated from the fitted GMM to {save_dir}...")
    samples = gmm.save_samples(
        n_samples=n_samples_generate, 
        save_fig_dir=save_dir,
        save_grid_shape=None,
        save_name=save_name,
    )
    
    print("Samples shape: ", samples.shape)
    print("Saving a 10x10 grid of the first 100 samples to figs directory...")
    save_images_grid(
        samples[:100], 
        file_path=os.path.join('figs', f"{save_name}_sample_grid.png"), 
        grid_shape=(10, 10),
    )

    # Plot the histogram of samples generated from the fitted GMM
    print("Plotting histogram of computed pixel statistics...")
    plot_pixel_intensity_hist(samples, bins=100)
