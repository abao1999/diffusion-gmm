import torch
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the Gram matrix
def compute_gram_matrix(features: np.ndarray) -> np.ndarray:
    c, h, w = features.shape
    features = features.view(c, h * w)
    gram_matrix = torch.mm(features, features.t())
    return gram_matrix

# Function to get eigenvalues of the Gram matrix
def get_gram_spectrum(gram_matrix):
    # Compute the eigenvalues (complex values) of the Gram matrix
    eigenvalues = torch.linalg.eigvals(gram_matrix)
    # Return the real part of the eigenvalues
    return eigenvalues.real


import PIL.Image
def save_sample(sample, i, file_path, custom_figsize=None):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])

    print(f"Saving image at step {i} to {file_path}")

    if custom_figsize:
        plt.figure(figsize=custom_figsize)
        plt.imshow(image_pil)
        plt.savefig(file_path)
    else:
        image_pil.save(file_path)

def save_images_grid(images, file_path, grid_shape=(20,20)):
    n, _, w, h = images.shape
    n_rows = w * grid_shape[0]
    n_cols = h * grid_shape[1]

    assert n >= grid_shape[0] * grid_shape[1], "you have fewer images than grid spaces!"

    new_im = PIL.Image.new('RGB', (n_rows, n_cols))

    idx = 0
    for i in range(0, n_rows, w):
        for j in range(0, n_cols, h):
            image_processed = images[idx:idx+1].cpu().permute(0, 2, 3, 1)
            image_processed = (image_processed + 1.0) * 127.5
            image_processed = image_processed.numpy().astype(np.uint8)
            im = PIL.Image.fromarray(image_processed[0])
            # paste the image at location i,j:
            new_im.paste(im, (i, j))
            idx += 1

    new_im.save(file_path)
