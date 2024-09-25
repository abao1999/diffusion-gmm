import argparse
import os
from typing import List, Optional

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

from diffusion_gmm.dataset import NumpyDataset
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
    target_class_name: str = None,
    verbose: bool = False,
    save_dir: str = "results",
    save_name: str = "gram_spectrum.npy",
) -> None:
    """
    Compute the Gram spectrum of a pre-trained CNN model on Imagenet data
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
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )

    if verbose:
        print("CNN model ID: ", cnn_model_id)
        print("Target class: ", target_class_name)
        print("Transform: ", transform)
        print("Num images: ", num_images)

    if mode == "diffusion":
        # Load the generated Imagenet data from the diffusion model
        root = os.path.join(DATA_DIR, "diffusion_imagenet")
        print(f"Loading Imagenet data from {root}")
        data = datasets.ImageFolder(
            root=root, transform=transform
        )

    elif mode == "gmm":
        # # Load the generated Imagenet data from the GMM model stored as .npy files

        root = os.path.join(DATA_DIR, "gmm_imagenet")
        if target_class_name is not None:
            root = os.path.join(root, target_class_name)
            # check if root exists
            if not os.path.exists(root):
                raise FileNotFoundError(
                    f"Target class folder {target_class_name} not found"
                )
            print(f"Loading target class {target_class_name} from {root}")
        else:
            root = os.path.join(root, "unknown")
            print(f"Loading all target classes from {root}")

        # def npy_loader(path):
        #     sample = torch.from_numpy(np.load(path))
        #     return sample

        # data = datasets.DatasetFolder(
        #     root=root,
        #     loader=npy_loader,
        #     extensions=[".npy"],  # type: ignore
        # )
        data = NumpyDataset(root=root, transform=None)

    elif mode == "real":
        # Load the real ImageNet data
        root = os.path.join(DATA_DIR, "imagenet")
        print(f"Loading Imagenet data from {root}")
        data = datasets.ImageFolder(
            root=root, transform=transform
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if target_class_name is not None:
        # Create a sampler that only selects images from the target class
        indices = [i for i, (_, label) in enumerate(DataLoader(data)) if label is not None]
        print(f"Number of images of class {target_class_name}: {len(indices)}")
        sel_indices = indices[:num_images] if len(indices) >= num_images else indices
        custom_sampler = SubsetRandomSampler(sel_indices)
    else:
        # choose num_images random indices from the dataset
        num_tot_samples = len(data)
        print("len data: ", num_tot_samples)
        sel_indices = list(np.random.choice(num_tot_samples, num_images, replace=False))
        custom_sampler = SubsetRandomSampler(sel_indices)

    dataloader = DataLoader(
        data, batch_size=batch_size, shuffle=False, sampler=custom_sampler
    )
    # Accumulate eigenvalues from all images
    all_eigenvalues: List[float] = []

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
    all_eigenvalues = np.array(all_eigenvalues)  # type: ignore
    if verbose:
        print("all_eigenvalues shape: ", all_eigenvalues.shape)  # type: ignore

    np.save(os.path.join(save_dir, save_name), all_eigenvalues)
    # np.save(f"gram_spectrum.npy", all_eigenvalues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the Gram spectrum of a pre-trained CNN model on Imagenet data"
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
        "--target_class_name", type=str, default=None, help="Target class name for real data"
    )
    args = parser.parse_args()
    print("Arguments: ", args)

    # TODO: this is currently hard-coded
    dataset = "imagenet"

    cnn_model_id = args.cnn_model
    hook_layer = args.hook_layer
    num_images = args.num_images
    mode = args.mode
    target_class_name = args.target_class_name

    # book-keeping for save names
    npy_save_dir = "results"
    figs_save_dir = "figs"

    # save computed eigenvalues to a file
    npy_save_name = f"{dataset}_gram_spectrum.npy"
    fig_save_name = f"{dataset}_gram_spectrum.png"

    if target_class_name is not None:
        print(f"Target class name: {target_class_name}")
        npy_save_name = f"{dataset}_{target_class_name}_gram_spectrum.npy"
        fig_save_name = f"{dataset}_{target_class_name}_gram_spectrum.png"

    if mode == "diffusion":
        diffusion_model = "diffusion_DiT"
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
            target_class_name=target_class_name,
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
