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
    compute_pixelwise: bool = False,
) -> np.ndarray:
    """
    Dataloader is for a specific class.
    Compute the norm of all images for a specific class.
    """
    image_norms = []

    for images, _ in tqdm(dataloader, desc="Computing norms"):
        images = images.to(device)
        batch_size = images.size(0)
        if compute_pixelwise:
            # comptue norm across channel dimension, then flatten
            norms = torch.norm(images, p=p, dim=1).flatten()
        else:
            norms = torch.norm(images.view(batch_size, -1), p=p, dim=1)
        image_norms.extend(norms.tolist())
    res = np.array(image_norms)
    print("shape of computed norms:", res.shape)
    return res


def plot_norms(
    data_dir,
    filename_suffix: str,
    class_list: list[str],
    plot_save_dir: str,
    p: int,
    plot_separately: bool = False,
    plot_name: Optional[str] = None,
    use_log_scale: bool = False,
    cmap: str = "tab10",
):
    os.makedirs(plot_save_dir, exist_ok=True)
    run_name = "-".join(class_list)
    norm_npy_paths = [
        os.path.join(data_dir, f"{target_class}{filename_suffix}.npy")
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
            print("loading from: ", norm_npy_path)
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
                f"{run_name if plot_name is None else plot_name}{filename_suffix}_separate.pdf",
            ),
            dpi=300,
        )
        plt.close()
    else:
        # Choose a colormap and extract 10 colors from it
        colormap = cm.get_cmap(cmap, len(class_list))
        colors = [mcolors.rgb2hex(colormap(i)) for i in range(colormap.N)]

        print("colors:", colors)
        plt.figure(figsize=(4, 3))
        for i, (target_class, norm_npy_path) in enumerate(
            zip(class_list, norm_npy_paths)
        ):
            print("loading from: ", norm_npy_path)
            image_norms = np.load(norm_npy_path)
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
                f"{run_name if plot_name is None else plot_name}{filename_suffix}.pdf",
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
    parser.add_argument("--compute_pixelwise", action="store_true")
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
        # class_list = [
        #     "baseball",
        #     "church",
        #     "english_springer",
        #     "french_horn",
        #     "garbage_truck",
        #     "goldfinch",
        #     "kimono",
        #     "salamandra",
        #     "tabby",
        #     "tench",
        # ]
        class_list = [
            "baseball",
            "cauliflower",
            "church",
            "coral_reef",
            "english_springer",
            "french_horn",
            "garbage_truck",
            "goldfinch",
            "kimono",
            "mountain_bike",
            "patas_monkey",
            "pizza",
            "planetarium",
            "polaroid",
            "racer",
            "salamandra",
            "tabby",
            "tench",
            "trimaran",
            "volcano",
        ]
    else:
        class_list = args.target_classes
    # run_name = "-".join(class_list)
    n_classes = len(class_list)
    run_name = f"{n_classes}_classes"
    filename_suffix = f"_p{args.p}_n{args.num_samples}_norms{'_pixelwise' if args.compute_pixelwise else ''}"

    # plot_norms(
    #     save_dir,
    #     filename_suffix,
    #     class_list,
    #     plot_save_dir,
    #     args.p,
    #     plot_separately=False,
    #     plot_name=run_name,
    #     use_log_scale=False,
    #     cmap="tab20",
    # )
    # exit()

    print(f"Computing norms for {run_name}...")
    data_dir = os.path.join(DATA_DIR, args.data_split)

    print("Setting up dataset...")
    dataset, is_npy_dataset = setup_dataset(data_dir)
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None
    print("Dataset setup complete.")

    for target_class in class_list:
        print(f"Computing norms for {target_class}")
        dataloader = build_dataloader_for_class(
            dataset, target_class, num_samples=args.num_samples
        )
        image_norms = compute_norm_of_images(
            dataloader, p=args.p, compute_pixelwise=args.compute_pixelwise
        )
        save_path = os.path.join(
            save_dir,
            f"{target_class}{filename_suffix}.npy",
        )
        print("Saving computed norms to:", save_path)
        np.save(save_path, image_norms)
        print("Image norms shape:", image_norms.shape)
        print(
            f"Average {'pixelwise ' if args.compute_pixelwise else ''}{args.p}-norm of {target_class} images': {np.mean(image_norms)}"
        )

    plot_norms(
        save_dir,
        filename_suffix,
        class_list,
        plot_save_dir,
        args.p,
        plot_separately=False,
        plot_name=run_name,
        use_log_scale=False,
        cmap="tab20",
    )
