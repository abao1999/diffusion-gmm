import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import DatasetFolder


def set_seed(rseed: int):
    """
    Set the seed for the random number generator for torch, cuda, and cudnn
    """
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

    torch.manual_seed(rseed)
    # If using CUDA, you should also set the seed for CUDA for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rseed)
        torch.cuda.manual_seed_all(rseed)  # if you have multiple GPUs

    # For deterministic behavior on GPU (reproducibility), use the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_targets(dataset: DatasetFolder) -> List[int]:
    # For datasets like CIFAR10, use targets attribute
    if hasattr(dataset, "targets"):
        targets = dataset.targets  # type: ignore
    # For ImageFolder, reconstruct targets from samples
    elif hasattr(dataset, "samples"):
        targets = [class_idx for _, class_idx in dataset.samples]  # type: ignore
    else:
        raise AttributeError("Dataset doesn't have 'targets' or 'samples' attribute")
    return targets


# plot histogram from npy file
def plot_gram_spectrum(
    all_eigenvalues: np.ndarray,
    verbose: bool = False,
    bins: Union[int, str] = 100,  # 'fd' for Freedman-Diaconis rule
    density: bool = True,
    save_dir: str = "figs",
    save_name: str = "gram_spectrum.png",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        print()
    plt.figure(figsize=(10, 6))
    plt.hist(all_eigenvalues, bins=bins, density=density, alpha=0.75, color="tab:blue")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title("Eigenvalues of Gram Matrices")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    print("Saved histogram to ", os.path.join(save_dir, save_name))


# plot histogram from npy file
def plot_from_npy(
    filepath: str = "gram_spectrum.npy",
    plot_fn: Callable[[np.ndarray], None] = plot_gram_spectrum,
    verbose: bool = False,
    plot_kwargs: dict = {},
):
    all_data = np.load(filepath)
    if verbose:
        print(f"Loaded data from file {filepath}")
        print("data shape: ", all_data.shape)
    plot_fn(all_data, **plot_kwargs)


def save_sample(
    sample: torch.Tensor,
    i: int,
    file_path: str,
    custom_figsize: Optional[Tuple[int, int]] = None,
) -> None:
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])

    print(f"Saving image at step {i} to {file_path}")

    if custom_figsize:
        plt.figure(figsize=custom_figsize)
        plt.imshow(image_pil)
        plt.savefig(file_path)
    else:
        image_pil.save(file_path)


def save_images_grid(
    images: np.ndarray,
    file_path: str,
    grid_shape: Tuple[int, int] = (10, 10),
) -> None:
    """
    Save a grid of images to a file
    Args:
        images: Numpy array of images
        file_path: Path to save the image
        grid_shape: Shape of the grid
        process_fn: Function to process the images before saving
    """
    assert images.dtype == np.uint8, "Images must be in uint8 format."

    num_images = grid_shape[0] * grid_shape[1]
    assert num_images <= images.shape[0], "Fewer images than grid spaces!"

    print(f"Populating grid of images with the first {num_images} images")
    images = images[:num_images]
    _, _, w, h = images.shape
    n_rows = w * grid_shape[0]
    n_cols = h * grid_shape[1]

    # create a new RGB image to paste the images onto
    new_im = Image.new("RGB", (n_rows, n_cols))

    idx = 0
    for i in range(0, n_rows, w):
        for j in range(0, n_cols, h):
            img = images[idx : idx + 1].squeeze()
            # Reorder dimensions to (h, w, c)
            img = np.transpose(img, (1, 2, 0))
            # convert to Image PIL type
            im = Image.fromarray(img)
            # paste the image at location i,j:
            new_im.paste(im, (i, j))
            idx += 1

    new_im.save(file_path)


def save_and_plot_samples(
    samples: np.ndarray,
    save_dir: str,
    save_grid_dir: Optional[str] = None,
    save_grid_shape: Tuple[int, int] = (10, 10),
    process_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    overwrite: bool = False,
) -> None:
    """
    Save and plot samples to data directory, and optionally saves grid of subset of samples
    Args:
        samples: Generated samples from the model, converted to numpy array
        save_dir: Directory to save the samples as png images for loading as a torchvision ImageFolder

    Optional Args:
        process_fn: Callable post-processing function from np array Image PIL type (e.g. for scaling, clipping, etc.)
                    NOTE: the dtype must be made into np.uint8 before saving as image
                    If None, default behavior is to save unprocessed samples into npy files
        save_grid: Boolean flag, whether to save grid of subset of samples
        save_grid_shape: Tuple of grid shape for the grid of samples
        save_grid_dir: Directory to save the grid of samples

    """
    os.makedirs(save_dir, exist_ok=True)
    print("Saving samples to ", save_dir)
    if process_fn is not None:
        # apply postprocessing to samples
        samples = process_fn(samples)
        assert (
            samples.dtype == np.uint8
        ), "Samples must be converted to np.uint8 before saving as image."

        # save image grid of subset of samples
        if save_grid_dir is not None:
            os.makedirs(save_grid_dir, exist_ok=True)
            parent_name = os.path.basename(os.path.dirname(save_dir))
            curr_name = os.path.basename(save_dir)
            save_name = f"{parent_name}_{curr_name}"
            grid_save_path = os.path.join(save_grid_dir, f"{save_name}_sample_grid.png")
            print("Saving grid of samples to ", grid_save_path)
            save_images_grid(
                samples,
                file_path=grid_save_path,
                grid_shape=save_grid_shape,
            )

        last_sample_index = -1

        if not overwrite:
            print("Checking for existing samples in save_dir...")
            # find the last sample index in save_dir
            sample_files = [
                f
                for f in os.listdir(save_dir)
                if os.path.isfile(os.path.join(save_dir, f))
            ]
            sample_indices = [
                int(f.split("_")[-1].split(".")[0])
                for f in sample_files
                if "sample" in f
            ]
            if sample_indices:
                last_sample_index = max(sample_indices)
            print("Last sample index: ", last_sample_index)

        # save all samples as png images in save_dir
        for i, img in enumerate(samples):
            # Reorder dimensions to (h, w, c)
            img = np.transpose(img, (1, 2, 0))
            # convert to Image PIL type
            img = Image.fromarray(img)
            sample_idx = last_sample_index + i + 1
            save_path = os.path.join(save_dir, f"sample_{sample_idx}.png")
            img.save(save_path)

    else:
        warnings.warn(
            "No image post-processing function provided. Saving samples as numpy files. This may not be the desired behavior."
        )
        # Default behavior: save as npy files
        for i, img in enumerate(samples):
            save_path = os.path.join(save_dir, f"sample_{i}.npy")
            np.save(save_path, img)


def default_image_processing_fn(
    samples: np.ndarray, verbose: bool = True
) -> np.ndarray:
    """
    Default processing function for samples
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    min_val, max_val = samples.min(), samples.max()
    if verbose:
        print("min_val, max_val: ", min_val, max_val)
    samples = np.clip(samples, min_val, max_val)
    samples = ((samples - min_val) / (max_val - min_val)) * 255
    samples = samples.astype(np.uint8)
    # samples = torch.clamp(samples, min_val, max_val)
    # samples = samples.to(torch.uint8)
    return samples


def plot_training_history(
    train_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    test_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    accuracy_history_all_runs: List[List[Tuple[float, float, int]]],
    save_dir: str = "plots",
    save_name: str = "loss_accuracy.png",
    title: str = "Binary Linear Classifier",
    plot_individual_runs: bool = False,
) -> None:
    fig, ax1 = plt.subplots()

    os.makedirs(save_dir, exist_ok=True)

    train_losses = [[item[1] for item in x] for x in train_loss_history_all_runs]
    test_losses = [[item[1] for item in x] for x in test_loss_history_all_runs]
    accuracies = [[item[1] for item in x] for x in accuracy_history_all_runs]
    # prop_train = np.linspace(0.1, 1.0, len(train_losses[0]))
    prop_train_schedule = [[item[0] for item in x] for x in train_loss_history_all_runs]
    n_runs = len(prop_train_schedule)
    assert (
        n_runs == len(train_losses) == len(test_losses) == len(accuracies)
    ), "Number of runs must match number of train, test, and accuracy lists"
    print("n_runs: ", n_runs)

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_test_losses = np.mean(test_losses, axis=0)
    mean_accuracies = np.mean(accuracies, axis=0)

    # Calculate standard deviation for train losses
    std_train_losses = np.std(train_losses, axis=0)
    std_test_losses = np.std(test_losses, axis=0)
    std_accuracies = np.std(accuracies, axis=0)

    ax1.set_xlabel("Proportion of Training Data")
    ax1.set_ylabel("Loss", color="tab:blue")

    if plot_individual_runs:
        for i in range(n_runs):
            ax1.plot(
                prop_train_schedule[i],
                train_losses[i],
                color="tab:blue",
                alpha=0.2,
            )
            ax1.plot(
                prop_train_schedule[i],
                test_losses[i],
                color="tab:orange",
                alpha=0.2,
            )
    # Plot average train loss
    ax1.plot(
        prop_train_schedule[0],  # Use the first schedule as they should all be the same
        mean_train_losses,
        label="Avg Train Loss",
        color="tab:blue",
        marker=".",
    )
    # Plot standard deviation envelope for train losses
    ax1.fill_between(
        prop_train_schedule[0],
        mean_train_losses - std_train_losses,
        mean_train_losses + std_train_losses,
        color="tab:blue",
        alpha=0.1,
    )
    # plot average test loss
    ax1.plot(
        prop_train_schedule[0],
        mean_test_losses,
        label="Avg Test Loss",
        color="tab:orange",
        marker=".",
    )
    # plot standard deviation envelope for test losses
    ax1.fill_between(
        prop_train_schedule[0],
        mean_test_losses - std_test_losses,
        mean_test_losses + std_test_losses,
        color="tab:orange",
        alpha=0.1,
    )

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:green")
    if plot_individual_runs:
        for i in range(n_runs):
            ax2.plot(
                prop_train_schedule[i],
                accuracies[i],
                color="tab:green",
                alpha=0.2,
            )
    # plot average accuracy
    ax2.plot(
        prop_train_schedule[0],
        mean_accuracies,
        label="Avg Accuracy",
        color="tab:green",
        marker=".",
    )
    # plot standard deviation envelope for accuracies
    ax2.fill_between(
        prop_train_schedule[0],
        mean_accuracies - std_accuracies,
        mean_accuracies + std_accuracies,
        color="tab:green",
        alpha=0.1,
    )
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.close()


def plot_accuracies(
    results: Dict[str, List[List[Tuple[float, float, int]]]],
    save_dir: str = "plots",
    save_name: str = "loss_accuracy.png",
    title: str = "Binary Linear Classifier",
) -> None:
    fig, ax1 = plt.subplots()

    os.makedirs(save_dir, exist_ok=True)

    for run_name, accuracy_history in results.items():
        accuracies = [[item[1] for item in x] for x in accuracy_history]
        prop_train_schedule = [[item[0] for item in x] for x in accuracy_history]
        n_runs = len(prop_train_schedule)
        assert n_runs == len(
            accuracies
        ), "Number of runs must match number of accuracies"
        print("n_runs: ", n_runs)

        mean_accuracies = np.mean(accuracies, axis=0)

        # Calculate standard deviation for accuracies
        std_accuracies = np.std(accuracies, axis=0)

        ax1.plot(
            prop_train_schedule[0],
            mean_accuracies,
            label=run_name,
            marker=".",
        )
        ax1.fill_between(
            prop_train_schedule[0],
            mean_accuracies - std_accuracies,
            mean_accuracies + std_accuracies,
            alpha=0.1,
        )

    ax1.set_xlabel("Proportion of Training Data")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    plt.title(title)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.close()
