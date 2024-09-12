import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


WORK_DIR = os.getenv('WORK')
DATA_DIR = os.path.join(WORK_DIR, 'vision_datasets')
FIGS_DIR = "tests/figs"


def main():
    dataset = 'diffusion_cifar10'
    split = 'unknown'
    img_dir = os.path.join(DATA_DIR, dataset, split)
    print("Image directory:", img_dir)
    # get list of all paths in image directory
    img_paths = sorted(Path(img_dir).glob('*.png'), key=lambda x: int(x.stem.split('_')[-1]))
    print("Number of images:", len(img_paths))
    print("First five image paths:", img_paths[:5])
    # load first png image from image directory
    test_img = plt.imread(img_paths[0])
    # plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(test_img)
    plt.axis('off')
    plt.savefig(os.path.join(FIGS_DIR, 'test_image.png'), bbox_inches='tight', pad_inches=0, dpi=300)

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(
        root=os.path.join(DATA_DIR, 'cifar10'), 
        train=False, 
        download=True, 
        transform=transform
    )
    return dataset

def plot_cifar10_sample(transform):
    # Load the CIFAR-10 dataset
    cifar10_real_dir = os.path.join(DATA_DIR, 'cifar10')
    # cifar10_dataset = datasets.ImageFolder(root=cifar10_real_dir, transform=transform)
    cifar10_dataset = datasets.CIFAR10(root=cifar10_real_dir, transform=transform)

    # Create a DataLoader to load the images
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

    # Get a single image from the dataset
    sample = next(iter(cifar10_loader))
    images, label = sample

    image = images[0] # First image in the batch
    print("Image shape:", image.shape)

    mean = image.mean()
    std = image.std()
    print("Image mean:", mean)
    print("Image std:", std)

    # Convert the image to a numpy array and plot it
    image_np = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.axis('off')
    plt.savefig(os.path.join(FIGS_DIR, 'cifar10.png'), bbox_inches='tight', pad_inches=0, dpi=300)

def plot_imagefolder_sample(transform, dataset_name='diffusion_cifar10', split='unknown'):
    # Load the CIFAR-10 dataset
    img_dir = os.path.join(DATA_DIR, dataset_name)
    # cifar10_dataset = datasets.ImageFolder(root=cifar10_real_dir, transform=transform)
    data = datasets.ImageFolder(root=img_dir, transform=transform)

    # Create a DataLoader to load the images
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    # Get a single image from the dataset
    sample = next(iter(dataloader))
    images, label = sample

    image = images[0] # First image in the batch
    print("Image shape:", image.shape)

    mean = image.mean()
    std = image.std()
    print("Image mean:", mean)
    print("Image std:", std)

    # Convert the image to a numpy array and plot it
    image_np = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.axis('off')
    plt.savefig(os.path.join(FIGS_DIR, f"{dataset_name}.png"), bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == '__main__':
    os.makedirs(FIGS_DIR, exist_ok=True)
    # set random seed
    rseed = 999
    np.random.seed(rseed)
    torch.manual_seed(rseed)

    # Define transformations for the CIFAR10 data
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # plot_cifar10_sample(transform)
    # plot_imagefolder_sample(transform, dataset_name='diffusion_cifar10', split='unknown')
    plot_imagefolder_sample(transform, dataset_name='gmm_cifar10')