import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")
FIGS_DIR = "tests/figs"


# Define the classes you are interested in (these are indices, replace with your target class indices)
# You can also use class names and map them to the indices as needed.
target_classes = [0]  # Replace with your specific class indices from ImageNet

# Define the transform for your dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load the ImageNet dataset (ensure you've got the dataset in the correct path or download it)
root = os.path.join(DATA_DIR, "imagenet")
print("root: ", root)
imagenet_dataset = datasets.ImageNet(root=root, split='val', transform=transform)

print("Number of images in the dataset:", len(imagenet_dataset))
# Create a subset of indices that belong to the target classes
def get_class_indices(dataset, target_classes):
    indices = []
    for idx, (img, label) in tqdm(enumerate(dataset)):
        if label in target_classes:
            indices.append(idx)
        if idx == 1000:
            break
    return indices

# indices = [i for i, (_, label) in enumerate(DataLoader(imagenet_dataset)) if label == 25]
# print(indices)

# Get the indices of the desired classes
class_indices = get_class_indices(imagenet_dataset, target_classes)
print("class_indices: ", class_indices)
# Create a subset of the dataset with only the desired classes
subset_dataset = Subset(imagenet_dataset, class_indices)

# Create a DataLoader to load the subset
data_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader
for images, labels in data_loader:
    # Your training code here
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels: {labels}")



# import os

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torch.utils.data import (  # SubsetRandomSampler, RandomSampler
#     DataLoader,
#     SubsetRandomSampler,
# )
# WORK_DIR = os.getenv("WORK", "")
# DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")
# FIGS_DIR = "tests/figs"

# if __name__ == "__main__":
#     os.makedirs(FIGS_DIR, exist_ok=True)
#     target_class = 25
#     num_images = 10
#     # set random seed
#     rseed = 999
#     np.random.seed(rseed)
#     torch.manual_seed(rseed)

#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )

#     root = os.path.join(DATA_DIR, "imagenet")
#     print("root: ", root)
#     data = datasets.ImageNet(
#         root=root, 
#         split="val", 
#         transform=transform,
#         target_transform=None
#         if target_class is None
#         else lambda y: y == target_class,
#     )
#     if target_class is not None:
#         # Create a sampler that only selects images from the target class
#         indices = [i for i, (_, label) in enumerate(DataLoader(data)) if label]
#         print(f"Number of images of class {target_class}: {len(indices)}")
#         sel_indices = indices[:num_images] if len(indices) >= num_images else indices
#         custom_sampler = SubsetRandomSampler(sel_indices)
#     else:
#         # custom_sampler = RandomSampler(data, replacement=False, num_samples=num_images)
#         # # custom_sampler = SubsetRandomSampler(range(num_images))

#         # choose num_images random indices from the dataset
#         num_tot_samples = len(data)
#         print("len data: ", num_tot_samples)
#         sel_indices = list(np.random.choice(num_tot_samples, num_images, replace=False))
#         custom_sampler = SubsetRandomSampler(sel_indices)

#     dataloader = DataLoader(
#         data, batch_size=1, shuffle=False, sampler=custom_sampler
#     )

#     image = next(iter(dataloader))[0][0]  # First image in the batch
#     # Convert the image to a numpy array and plot it
#     image_np = image.permute(1, 2, 0).numpy()
#     plt.figure(figsize=(5, 5))
#     plt.imshow(image_np)
#     plt.axis("off")
#     save_path = os.path.join(FIGS_DIR, f"imagenet_salamander.png")
#     plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)