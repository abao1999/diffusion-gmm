"""
Test the ImageNet dataset
"""

import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from diffusion_gmm.utils import default_image_processing_fn, save_images_grid

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")
FIGS_DIR = "tests/figs"

# Define the classes you are interested in (these are indices, replace with your target class indices)
# You can also use class names and map them to the indices as needed.
target_classes = [0]  # Replace with your specific class indices from ImageNet

# Define the transform for your dataset
# transform = transforms.Compose(
#     [
#         transforms.Resize(32),
#         transforms.CenterCrop(32),
#         transforms.ToTensor(),
#     ]
# )

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Load the ImageNet dataset (ensure you've got the dataset in the correct path or download it)
root = os.path.join(DATA_DIR, "imagenet")
print(f"Loading Imagenet data from {root}")
data = datasets.ImageFolder(root=root, transform=transform)

# Create a subset of the dataset with only the desired classes
subset_dataset = Subset(data, list(np.arange(0, 25)))

# Create a DataLoader to load the subset
dataloader = DataLoader(subset_dataset, batch_size=16, shuffle=False)

# Iterate through the DataLoader
images_lst = []
for images, labels in dataloader:
    # Your training code here
    images = default_image_processing_fn(images)
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels: {labels}")
    print("dtype: ", images.dtype)
    images_lst.extend(images)

images_np = np.array(images_lst)
save_images_grid(
    images_np, file_path="tests/figs/imagenet_salamander.png", grid_shape=(5, 5)
)
