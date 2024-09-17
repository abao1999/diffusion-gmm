import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from diffusion_gmm.gmm import ImageGMM
from diffusion_gmm.utils import (
    default_image_processing_fn,
    plot_pixel_intensity_hist,
)

FIGS_DIR = "figs"
WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")


if __name__ == "__main__":
    # Define the parameters
    parser = argparse.ArgumentParser(
        description="Fit a Gaussian Mixture Model (GMM) to image data"
    )
    parser.add_argument(
        "--use_generated_data",
        action="store_true",
        help="Use generated data instead of real data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for dataloader"
    )
    parser.add_argument(
        "--n_for_stats",
        type=int,
        default=1024,
        help="Number of images to use for computing statistics",
    )
    parser.add_argument(
        "--n_for_fit",
        type=int,
        default=1024,
        help="Number of images to use for fitting the GMM",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="cifar10", help="Name of the dataset"
    )
    parser.add_argument(
        "--n_samples_generate",
        type=int,
        default=1024,
        help="Number of samples to generate from the fitted GMM",
    )
    args = parser.parse_args()

    use_generated_data = args.use_generated_data
    batch_size = args.batch_size

    # for gmm fitting
    n_for_stats = args.n_for_stats
    n_for_fit = args.n_for_fit

    # for generating samples
    dataset_name = args.dataset_name
    if dataset_name != "cifar10":
        raise NotImplementedError("Only CIFAR10 dataset is supported for now.")

    cifar10_shape = (3, 32, 32)
    n_samples_generate = 1024

    # Define the transformation to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the dataset
    if use_generated_data:
        # load generated images
        image_dir = os.path.join(DATA_DIR, f"diffusion_{dataset_name}")
        dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    else:
        # load real images
        image_dir = os.path.join(DATA_DIR, dataset_name)
        dataset = datasets.CIFAR10(
            root=os.path.join(DATA_DIR, "cifar10"),
            train=False,
            download=True,
            transform=transform,
        )

    print("Image directory:", image_dir)

    n_tot_images = len(dataset)
    if n_for_stats > n_tot_images:
        print(
            f"Warning: Only {n_tot_images} images found in the dataset. Using all available images."
        )
        n_for_stats = n_tot_images

    custom_sampler = SubsetRandomSampler(range(n_for_stats))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=custom_sampler,  # if custom_sampler else None
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

    mean = np.array(gmm.mean)
    covariance = np.array(gmm.covariance)
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
    plt.savefig("figs/mean_pixel_values.png", dpi=300)

    plt.figure(figsize=(10, 6))
    plt.hist(covariance.flatten(), bins=100)
    plt.title("Covariance Pixel Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("figs/covariance_pixel_values.png", dpi=300)

    # Save the samples generated from the fitted GMM
    save_dir = os.path.join(DATA_DIR, "gmm_cifar10", "unknown")
    save_name = f"gmm_{dataset_name}"
    print(f"Saving samples generated from the fitted GMM to {save_dir}...")

    samples = gmm.save_samples(
        n_samples=n_samples_generate,
        save_dir=save_dir,
        plot_kwargs={
            "save_grid_dir": FIGS_DIR,
            "save_grid_shape": (10, 10),
            "process_fn": default_image_processing_fn,
        },
    )

    # count number of negative values in samples
    num_neg_values = np.sum(samples < 0)
    print(f"Number of negative values in samples: {num_neg_values}")

    # Plot the histogram of samples generated from the fitted GMM
    print("Plotting histogram of computed pixel statistics...")
    plot_pixel_intensity_hist(samples, bins=100)
