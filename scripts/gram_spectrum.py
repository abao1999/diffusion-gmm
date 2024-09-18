import argparse
import os
from typing import Optional

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import (  # SubsetRandomSampler, RandomSampler
    DataLoader,
    SubsetRandomSampler,
)
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
    cnn_model_id: Optional[str] = None,
    hook_layer: int = 10,
    num_images: int = 1024,
    batch_size: int = 64,
    target_class: Optional[int] = None,
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
    if cnn_model_id is not None:
        try:
            model = getattr(models, cnn_model_id)(pretrained=True).features.eval()
        except AttributeError:
            raise ValueError(f"Invalid CNN model ID: {cnn_model_id}")

        print(f"Using CNN model: {cnn_model_id}")
        # Define a hook to extract features from a specific layer (e.g., after layer 10)
        features = []

        def hook(module, input, output):
            features.append(output)

        # Attach the hook to a specific layer
        model[hook_layer].register_forward_hook(hook)
        print("Hook attached to layer: ", hook_layer)

    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    if verbose:
        print("CNN model ID: ", cnn_model_id)
        print("Target class: ", target_class)
        print("Transform: ", transform)
        print("Num images: ", num_images)

    if mode == "diffusion":
        # Load the generated CIFAR10 data from the diffusion model
        data = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, "diffusion_cifar10"), transform=transform
        )

    elif mode == "gmm":
        # # Load the generated CIFAR10 data from the GMM model
        # data = datasets.ImageFolder(
        #     root=os.path.join(DATA_DIR, "gmm_cifar10"), transform=transform
        # )
        # Load the generated CIFAR10 data from the GMM model stored as .npy files
        class NumpyDataset(torch.utils.data.Dataset):
            def __init__(self, root, transform=None):
                self.root_dir = root
                self.transform = transform
                self.file_list = [f for f in os.listdir(root) if f.endswith('.npy')]

            def __len__(self):
                return len(self.file_list)

            def __getitem__(self, idx):
                file_path = os.path.join(self.root_dir, self.file_list[idx])
                image = np.load(file_path)
                if self.transform:
                    image = self.transform(image)
                return image, 9  # Dummy label

        data = NumpyDataset(root=os.path.join(DATA_DIR, "gmm_cifar10", "unknown"), transform=None)

    elif mode == "real":
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

    else:
        raise ValueError(f"Invalid mode: {mode}")

    if target_class is not None:
        # Create a sampler that only selects images from the target class
        indices = [i for i, (_, label) in enumerate(data) if label]
        print(f"Number of images of class {target_class}: {len(indices)}")
        sel_indices = indices[:num_images] if len(indices) >= num_images else indices
        custom_sampler = SubsetRandomSampler(sel_indices)
    else:
        # custom_sampler = RandomSampler(data, replacement=False, num_samples=num_images)
        # # custom_sampler = SubsetRandomSampler(range(num_images))

        # choose num_images random indices from the dataset
        print("len data: ", len(data))
        sel_indices = np.random.choice(len(data), num_images, replace=False)
        print("sel indices shape: ", sel_indices.shape)
        custom_sampler = SubsetRandomSampler(list(sel_indices))

    dataloader = DataLoader(
        data, batch_size=batch_size, shuffle=False, sampler=custom_sampler
    )
    # Accumulate eigenvalues from all images
    all_eigenvalues = []

    for idx, (images, _) in tqdm(enumerate(dataloader)):
        if cnn_model_id is not None:
            features.clear()  # Clear previous features
            with torch.no_grad():
                model(images)  # Forward pass through the model
            # use first features from from hook
            feats = features[0].squeeze().cpu().numpy()
            gram_matrix = compute_gram_matrix(feats)
            spectrum = get_gram_spectrum(gram_matrix)
            all_eigenvalues.extend(spectrum)

        else:
            data = images.squeeze().cpu().numpy()
            print("data shape: ", data.shape)
            gram_matrix = compute_gram_matrix(data)
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
        "--cnn_model", type=str, default=None, help="Pre-trained CNN model ID"
    )
    parser.add_argument(
        "--hook_layer", type=int, default=10, help="Layer to extract features from"
    )
    parser.add_argument(
        "--num_images", type=int, default=1024, help="Number of images to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for dataloader"
    )
    parser.add_argument(
        "--target_class", type=int, default=None, help="Target class for real data"
    )
    args = parser.parse_args()
    print("Arguments: ", args)

    # TODO: this is currently hard-coded
    dataset = "cifar10"

    cnn_model_id = args.cnn_model
    hook_layer = args.hook_layer
    num_images = args.num_images
    mode = args.mode
    target_class = args.target_class

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
    if True:  # not os.path.exists(npy_path) or args.overwrite:
        print("Computing and saving Gram spectrum...")
        main(
            mode,
            cnn_model_id=cnn_model_id,
            hook_layer=hook_layer,
            num_images=num_images,
            target_class=target_class,
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
