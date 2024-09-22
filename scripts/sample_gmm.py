import argparse
import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from diffusion_gmm.gmm import ImageGMM

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
        "--n_components", type=int, default=1, help="Number of components"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for dataloader"
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
    parser.add_argument(
        "--num_images",
        type=int,
        default=1024,
        help="Number of images to process from the dataset",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Target class for real data (default: None)",
    )

    args = parser.parse_args()

    use_generated_data = args.use_generated_data
    n_components = args.n_components
    batch_size = args.batch_size
    n_samples_generate = args.n_samples_generate
    num_images = args.num_images
    target_class = args.target_class

    print("Target class: ", target_class)

    # for generating samples
    dataset_name = args.dataset_name
    if dataset_name != "cifar10":
        raise NotImplementedError("Only CIFAR10 dataset is supported for now.")

    cifar10_shape = (3, 32, 32)

    # Define the transformation to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the dataset
    if use_generated_data:
        # load generated images
        image_dir = os.path.join(DATA_DIR, f"diffusion_{dataset_name}")
        data = datasets.ImageFolder(root=image_dir, transform=transform)
        raise NotImplementedError
        # TODO: need to align labels (targets) when we don't have all the target classes
    else:
        # Load the real CIFAR10 data from torchvision
        data = datasets.CIFAR10(
            root=os.path.join(DATA_DIR, "cifar10"),
            train=True,  # TODO: change to False
            download=True,
            transform=transform,
            target_transform=None
            if target_class is None
            else lambda y: y == target_class,
        )

    classes = data.classes
    print(f"Classes: {classes}")

    # filter the target_class dataset to include only at most num_images images
    if target_class is not None:
        # Create a sampler that only selects images from the target class
        indices = [i for i, (_, label) in enumerate(DataLoader(data)) if label]
        print(f"Number of images of class {target_class}: {len(indices)}")
        sel_indices = indices[:num_images] if len(indices) >= num_images else indices
        custom_sampler = SubsetRandomSampler(sel_indices)
    else:
        # choose num_images random indices from the dataset
        num_tot_samples = len(data)
        sel_indices = list(np.random.choice(num_tot_samples, num_images, replace=False))
        custom_sampler = SubsetRandomSampler(sel_indices)

    dataloader = DataLoader(
        data, batch_size=batch_size, shuffle=False, sampler=custom_sampler
    )

    gmm = ImageGMM(
        dataloader=dataloader,
        img_shape=cifar10_shape,
        n_components=n_components,  # NOTE: make sure this is number of classes when fitting on entire dataset
        verbose=True,
    )

    ## We currently don't actually use the full GMM machinery, because we sample from only a single class for now
    # gmm.fit()
    # print("GMM fitted successfully.")

    # Save the samples generated from the fitted GMM
    if target_class is not None:
        target_class_name = classes[target_class]
        print(f"Target class name: {target_class_name}")
        save_dir = os.path.join(DATA_DIR, "gmm_cifar10", target_class_name)
    else:
        save_dir = os.path.join(DATA_DIR, "gmm_cifar10", "unknown")
    save_name = f"gmm_{dataset_name}"
    print(f"Saving samples generated from the fitted GMM to {save_dir}...")

    samples = gmm.save_samples_single_class(
        n_samples=n_samples_generate,
        save_dir=save_dir,
        plot_kwargs={
            "save_grid_dir": FIGS_DIR,
            "save_grid_shape": (10, 10),
            "process_fn": None,
        },
    )

    # count number of negative values in samples
    num_neg_values = np.sum(samples < 0)
    print(f"Number of negative values in samples: {num_neg_values}")

    # # Plot the histogram of samples generated from the fitted GMM
    # print("Plotting histogram of computed pixel statistics...")
    # plot_pixel_intensity_hist(samples, bins=100)
