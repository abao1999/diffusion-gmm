import argparse
import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

from diffusion_gmm.utils import (
    compute_gram_matrix,
    get_gram_spectrum,
    plot_from_npy,
    plot_gram_spectrum,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")


def main(
    mode: str,
    cnn_model_id: str = "vgg16",
    hook_layer: int = 10,
    num_images: int = 1024,
    verbose: bool = False,
    save_dir: str = "results",
    save_name: str = "gram_spectrum.npy",
) -> None:
    """
    Compute the Gram spectrum of a pre-trained CNN model on CIFAR10 data
    TODO: Drop-in support for other datasets and models
    """

    os.makedirs(save_dir, exist_ok=True)

    # Load a pre-trained model, e.g., VGG16 (models.vgg16)
    model = getattr(models, cnn_model_id)(pretrained=True).features.eval()

    # Define a hook to extract features from a specific layer (e.g., after layer 10)
    features = []

    def hook(module, input, output):
        features.append(output)

    # Attach the hook to a specific layer
    model[hook_layer].register_forward_hook(hook)

    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    if verbose:
        print("Applying transformation: ", transform)

    if mode == "diffusion":
        # Load the generated CIFAR10 data from the diffusion model
        data = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, "diffusion_cifar10"), transform=transform
        )

    elif mode == "gmm":
        # Load the generated CIFAR10 data from the GMM model
        data = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, "gmm_cifar10"), transform=transform
        )

    elif mode == "real":
        # Load the real CIFAR10 data from torchvision
        data = datasets.CIFAR10(
            root=os.path.join(DATA_DIR, "cifar10"),
            train=False,
            download=True,
            transform=transform,
        )

    else:
        raise ValueError(f"Invalid mode: {mode}")

    custom_sampler = RandomSampler(data, replacement=False, num_samples=num_images)
    # custom_sampler = SubsetRandomSampler(range(num_images))
    dataloader = DataLoader(data, batch_size=64, shuffle=False, sampler=custom_sampler)

    # Accumulate eigenvalues from all images
    all_eigenvalues = []

    for idx, (images, _) in tqdm(enumerate(dataloader)):
        features.clear()  # Clear previous features
        with torch.no_grad():
            model(images)  # Forward pass through the model

        if features:
            # use first features from from hook
            feats = features[0].squeeze().cpu().numpy()
            gram_matrix = compute_gram_matrix(feats)
            spectrum = get_gram_spectrum(gram_matrix)
            all_eigenvalues.extend(spectrum)

    # Convert accumulated eigenvalues to a numpy array
    all_eigenvalues = np.array(all_eigenvalues)
    if verbose:
        print("all_eigenvalues shape: ", all_eigenvalues.shape)
    # save computed eigenvalues to a file
    np.save(os.path.join(save_dir, save_name), all_eigenvalues)
    # np.save(f"gram_spectrum.npy", all_eigenvalues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the Gram spectrum of a pre-trained CNN model on CIFAR10 data"
    )
    parser.add_argument(
        "mode", type=str, help="Mode to run in: real, diffusion, or gmm"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing npy file"
    )
    parser.add_argument(
        "--cnn_model", type=str, default="vgg16", help="Pre-trained CNN model ID"
    )
    parser.add_argument(
        "--hook_layer", type=int, default=10, help="Layer to extract features from"
    )
    parser.add_argument(
        "--num_images", type=int, default=1024, help="Number of images to process"
    )
    args = parser.parse_args()

    # TODO: this is currently hard-coded
    dataset = "cifar10"

    cnn_model_id = args.cnn_model
    hook_layer = args.hook_layer
    num_images = args.num_images
    mode = args.mode

    # book-keeping for save names
    npy_save_dir = "results"
    figs_save_dir = "figs"

    npy_save_name = f"{dataset}_gram_spectrum.npy"
    fig_save_name = f"{dataset}_gram_spectrum.png"

    if mode == "diffusion":
        diffusion_model = "ddpm"
        npy_save_name = f"{diffusion_model}_{npy_save_name}"
        fig_save_name = f"{diffusion_model}_{fig_save_name}"
    elif mode == "gmm":
        npy_save_name = f"gmm_{npy_save_name}"
        fig_save_name = f"gmm_{fig_save_name}"
    elif mode == "real":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    npy_path = os.path.join(npy_save_dir, npy_save_name)

    # if npy file doesnt already exist, compute and save it
    if not os.path.exists(npy_path) or args.overwrite:
        print("Computing and saving Gram spectrum...")
        main(
            mode,
            cnn_model_id=cnn_model_id,
            hook_layer=hook_layer,
            num_images=num_images,
            verbose=True,
            save_dir=npy_save_dir,
            save_name=npy_save_name,
        )
    else:
        print(f"Loading Gram spectrum from file: {npy_path}")

    plot_from_npy(
        filepath=npy_path,
        plot_fn=plot_gram_spectrum,
        verbose=True,
        plot_kwargs={
            "bins": 100,
            "density": True,
            "save_dir": figs_save_dir,
            "save_name": fig_save_name,
        },
    )
