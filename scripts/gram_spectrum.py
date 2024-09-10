import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image

from diffusion_gmm.utils import compute_gram_matrix, get_gram_spectrum

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os


WORK_DIR = os.getenv('WORK')
DATA_DIR = os.path.join(WORK_DIR, 'vision_datasets')


def main():
    # Load a pre-trained model, e.g., VGG16
    model = models.vgg16(pretrained=True).features.eval()

    # Define a hook to extract features from a specific layer (e.g., after layer 10)
    features = []

    def hook(module, input, output):
        features.append(output)

    # Attach the hook to a specific layer
    model[10].register_forward_hook(hook)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # # Define transformations for the MNIST data
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # Load ImageNet data (or a subset, like ImageNet's validation set)
    # imagenet_data = datasets.ImageNet(root='path/to/imagenet', transform=transform)
    # data = datasets.MNIST(root=os.path.join(DATA_DIR, 'mnist'), train=False, download=True, transform=transform)

    # data = datasets.CIFAR10(
    #     root=os.path.join(DATA_DIR, 'cifar10'), 
    #     train=False, 
    #     download=True, 
    #     transform=transform
    # )

    data = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'generated_cifar10'),
        transform=transform
    )

    # # Custom loader function to load images
    # def pil_loader(path):
    #     return Image.open(path).convert("RGB")

    # # custom DatasetFolder class to load images from a directory without requiring class subdirectories
    # data = datasets.DatasetFolder(
    #     root=os.path.join(DATA_DIR, 'generated_cifar10'), 
    #     loader=pil_loader, 
    #     extensions=('png', 'jpg', 'jpeg'), 
    #     transform=transform
    # )

    # keep batch_size=1 because we are interested in the Gram matrix of a single image
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    # Accumulate eigenvalues from all images
    all_eigenvalues = []

    for i, (images, _) in tqdm(enumerate(dataloader)):
        features.clear()  # Clear previous features
        with torch.no_grad():
            model(images)  # Forward pass through the model
        
        if features:
            # print("number of features: ", len(features))
            gram_matrix = compute_gram_matrix(features[0].squeeze())
            spectrum = get_gram_spectrum(gram_matrix)
            all_eigenvalues.extend(spectrum.cpu().numpy())
        
        # Limit to the first _ images for demonstration purposes
        if i == 999:
            break

    # Convert accumulated eigenvalues to a numpy array
    all_eigenvalues = np.array(all_eigenvalues)
    print("all_eigenvalues shape: ", all_eigenvalues.shape)
    # save computed eigenvalues to a file
    np.save(f"gram_spectrum.npy", all_eigenvalues)

    # Plotting the histogram of density versus eigenvalue
    plt.figure(figsize=(10, 6))
    plt.hist(all_eigenvalues, bins='fd', density=True, alpha=0.75, color='tab:blue')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.title('Eigenvalues of Gram Matrices')
    plt.grid(True)
    plt.savefig('gram_spectrum.png', dpi=300)

# plot histogram from npy file
def plot_gram_spectrum(filepath: str = 'gram_spectrum.npy'):
    print(f"Loading eigenvalues from file {filepath}")
    all_eigenvalues = np.load(filepath)
    plt.figure(figsize=(10, 6))
    plt.hist(all_eigenvalues, bins=100, density=True, alpha=0.75, color='tab:blue')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.title('Eigenvalues of Gram Matrices')
    plt.grid(True)
    plt.savefig('gram_spectrum.png', dpi=300)

if __name__ == "__main__":
    # main()
    plot_gram_spectrum()