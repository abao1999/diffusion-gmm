import argparse
import os
from typing import Optional, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusion_gmm.base import DataPrefetcher
from diffusion_gmm.utils import build_dataloader_for_class, setup_dataset

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")

plt.style.use(["ggplot", "custom_style.mplstyle"])


def compute_norm_of_images(
    dataloader: Union[DataLoader, DataPrefetcher],
    device: str = "cpu",
    p: int = 2,
) -> np.ndarray:
    """
    Dataloader is for a specific class.
    Compute the norm of all images for a specific class.
    """
    image_norms = []

    for images, _ in tqdm(dataloader, desc="Computing norms"):
        images = images.to(device)
        batch_size = images.size(0)
        norms = torch.norm(images.view(batch_size, -1), p=p, dim=1)
        image_norms.extend(norms.tolist())

    return np.array(image_norms)


def plot_norms(
    data_dir,
    class_list: list[str],
    plot_save_dir: str,
    p: int,
    plot_separately: bool = False,
    plot_name: Optional[str] = None,
    use_log_scale: bool = False,
):
    os.makedirs(plot_save_dir, exist_ok=True)
    run_name = "-".join(class_list)
    norm_npy_paths = [
        os.path.join(data_dir, f"{target_class}_{args.p}-norms.npy")
        for target_class in class_list
    ]
    print(f"existing saved norm filepaths: {norm_npy_paths}")
    if plot_separately:
        # make a plot with a subplot for each class histogram of norms
        n_cols = 5
        n_rows = len(class_list) // n_cols + (1 if len(class_list) % n_cols > 0 else 0)
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 6 * n_rows)
        )
        axes = axes.flatten()  # Flatten the axes array for easy 1D indexing
        for i, (target_class, norm_npy_path) in enumerate(
            zip(class_list, norm_npy_paths)
        ):
            image_norms = np.load(norm_npy_path)
            axes[i].hist(image_norms, bins=50, density=False, label=target_class)
            axes[i].set_title(target_class.replace("_", " ").capitalize())
            axes[i].set_xlabel("Norm")
            axes[i].set_ylabel("Count" if not use_log_scale else "Count (log scale)")
            if use_log_scale:
                axes[i].set_yscale("log")
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plot_save_dir,
                f"{run_name if plot_name is None else plot_name}_{p}-norms_separate.png",
            ),
            dpi=300,
        )
        plt.close()
    else:
        # Choose a colormap and extract 10 colors from it
        colormap = cm.get_cmap(
            "tab10", 10
        )  # 'tab10' is a colormap with 10 distinct colors
        colors = [mcolors.rgb2hex(colormap(i)) for i in range(colormap.N)]

        print("colors:", colors)
        plt.figure(figsize=(4, 3))
        for i, target_class in enumerate(class_list):
            image_norms = np.load(
                os.path.join(data_dir, f"{target_class}_{p}-norms.npy")
            )
            print("class:", target_class)
            print("Image norms shape:", image_norms.shape)
            print("min, max:", np.min(image_norms), np.max(image_norms))
            plt.hist(
                image_norms,
                bins=50,
                density=False,
                label=target_class,
                color=colors[i],
                alpha=0.3,
            )
            # # Add an arrow at the minimum value of image_norms with lower zorder
            # min_value = np.min(image_norms)
            # plt.annotate(
            #     "",
            #     xy=(min_value, 0),
            #     xytext=(min_value, -5),  # Adjust the y position for the arrow tail
            #     arrowprops=dict(
            #         facecolor=colors[i], shrink=0.05, width=2, headwidth=4, alpha=0.5
            #     ),
            #     zorder=1,  # Lower zorder to ensure axis markings are on top
            # )

        # # Add a single legend entry for the minimum value arrows
        # plt.scatter([], [], color="black", marker="|", s=100, label="Min value")

        p_label = r"\infty" if p == float("inf") else str(p)
        plt.title(r"$\ell_{" + p_label + r"}$ Norm")
        plt.xlabel(r"$\ell_{" + p_label + r"}$ Norm")
        plt.ylabel("Count" if not use_log_scale else "Count (log scale)")
        if use_log_scale:
            plt.yscale("log")
        # plt.legend()
        # plt.legend(loc="upper left")
        plt.savefig(
            os.path.join(
                plot_save_dir,
                f"{run_name if plot_name is None else plot_name}_{p}-norms.pdf",
            ),
            dpi=300,
        )
        plt.close()


def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_classes", type=str, nargs="+", required=True)
    parser.add_argument("--data_split", type=str, default="edm_imagenet64_all")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--p", type=int_or_float, default=2)
    parser.add_argument("--plot_save_dir", type=str, default="final_plots/norms")
    parser.add_argument("--plot_name", type=str, default=None)
    parser.add_argument(
        "--data_save_dir", type=str, default=os.path.join(DATA_DIR, "norms")
    )
    args = parser.parse_args()

    plot_save_dir = os.path.join(args.plot_save_dir, args.data_split)
    os.makedirs(args.plot_save_dir, exist_ok=True)

    save_dir = os.path.join(args.data_save_dir, args.data_split)
    os.makedirs(save_dir, exist_ok=True)

    if args.target_classes == ["all"]:
        class_list = [
            "baseball",
            "church",
            "english_springer",
            "french_horn",
            "garbage_truck",
            "goldfinch",
            "kimono",
            "salamandra",
            "tabby",
            "tench",
        ]
    else:
        class_list = args.target_classes
    # run_name = "-".join(class_list)
    n_classes = len(class_list)
    run_name = f"{n_classes}_classes"

    plot_norms(
        save_dir,
        class_list,
        plot_save_dir,
        args.p,
        plot_separately=False,
        plot_name=run_name,
        use_log_scale=False,
    )
    exit()

    print(f"Computing norms for {run_name}...")
    data_dir = os.path.join(DATA_DIR, args.data_split)

    print("Setting up dataset...")
    dataset, is_npy_dataset = setup_dataset(data_dir)
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None
    print("Dataset setup complete.")

    for target_class in args.target_classes:
        print(f"Computing norms for {target_class}")
        dataloader = build_dataloader_for_class(
            dataset, target_class, num_samples=args.num_samples
        )
        image_norms = compute_norm_of_images(dataloader, p=args.p)
        save_path = os.path.join(save_dir, f"{target_class}_{args.p}-norms.npy")
        np.save(save_path, image_norms)
        print("Image norms shape:", image_norms.shape)
        print(
            f"Average {args.p}-norm of {target_class} images': {np.mean(image_norms)}"
        )

    plot_norms(
        save_dir,
        class_list,
        plot_save_dir,
        args.p,
        plot_separately=False,
        plot_name=args.plot_name,
        use_log_scale=False,
    )
