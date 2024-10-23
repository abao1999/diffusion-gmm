"""
labels (see ILSVRC2017 development kit):
https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57

Also, see: https://patrykchrabaszcz.github.io/Imagenet32/

Using images downloaded from downscaled section: https://www.image-net.org/download-images.php
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets", "imagenet64")


def unpickle(file):
    with open(file, "rb") as f:
        d = pickle.load(f)  # encoding="latin1")
    return d


def load_databatch(data_file, img_size=64):
    print(data_file)
    d = unpickle(data_file)
    x = d["data"]
    y = d["labels"]
    # mean_image = d["mean"]

    print("Shapes")
    print(x.shape)
    print(len(y))

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]

    x = x.reshape((x.shape[0], img_size, img_size, 3))

    return x, np.array(y)


# Load and visualize a batch of images
def load_batch(file):
    data = unpickle(file)
    images = data["data"]
    labels = data["labels"]
    images = images.reshape(len(images), 3, 64, 64).transpose(
        0, 2, 3, 1
    )  # Convert to HWC format
    return images, labels


if __name__ == "__main__":
    data_split_path = os.path.join(DATA_DIR, "val_data")

    # images, labels = load_databatch(data_split_path)
    images, labels = load_batch(data_split_path)

    print(images.shape)
    print(len(labels))

    # Display an image
    sample_idx = 10
    img = Image.fromarray(images[sample_idx])
    # Save the first image to disk (if needed)
    img.save("example_image.png")
    print(labels[sample_idx])

    # Loop through images and print their labels
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.savefig(f"sample_{i}.png", dpi=300)

        if i == 3:  # Display the first 4 images
            break
