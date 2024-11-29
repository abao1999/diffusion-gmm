import argparse
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
from PIL import Image

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")


def create_image_grid(
    image_paths: Dict[str, List[str]],
    save_path: str,
    title: str = "Imagenet 64x64",
    save_as_image: bool = True,
) -> None:
    """
    Creates a grid of images with categories on the y-axis and title.

    :param image_paths: Dictionary mapping class names to lists of image file paths.
    :param title: Title of the plot.
    """
    num_classes = len(image_paths)
    num_images_per_class = len(next(iter(image_paths.values())))

    if save_as_image:
        sample = Image.open(next(iter(image_paths.values()))[0])
        w, h = sample.size
        # Calculate the number of blocks needed per class
        block_size = 3
        n_blocks_per_row = 2
        # Create a new RGB image to paste the images onto
        new_im = Image.new(
            "RGB",
            (
                block_size * w * n_blocks_per_row,
                block_size * h * num_classes // n_blocks_per_row,
            ),
        )

        for class_idx, (class_name, paths) in enumerate(image_paths.items()):
            for img_idx, img_path in enumerate(paths[: block_size**2]):
                img = Image.open(img_path)
                # Calculate position to paste the image in a 3x3 block
                block_row = img_idx // block_size
                block_col = img_idx % block_size
                # Calculate the position based on the class index and block position
                i = ((class_idx // n_blocks_per_row) * block_size + block_row) * h
                j = ((class_idx % n_blocks_per_row) * block_size + block_col) * w
                new_im.paste(img, (j, i))
        print(f"Saving grid image to {save_path}")
        new_im.save(save_path)
    else:
        fig, axes = plt.subplots(
            num_classes,
            num_images_per_class,
            figsize=(num_images_per_class, num_classes),
        )
        # plt.suptitle(title, fontsize=16)

        for row_idx, (class_name, paths) in enumerate(image_paths.items()):
            for col_idx, ax in enumerate(axes[row_idx]):
                img = Image.open(paths[col_idx])
                ax.imshow(img)
                ax.axis("off")
                # if col_idx == 0:
                #     ax.set_ylabel(
                #         class_name, rotation=0, labelpad=1, fontsize=12, va="center"
                #     )

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, required=True)
    parser.add_argument("--target_classes", type=str, nargs="+", required=True)
    parser.add_argument("--num_samples_per_class", type=int, default=9)
    parser.add_argument("--save_dir", type=str, default="final_plots")
    args = parser.parse_args()

    data_dir = os.path.join(DATA_DIR, args.data_split)

    class_names = "-".join(args.target_classes)
    image_paths = {
        class_name: random.sample(
            [
                os.path.join(data_dir, class_name, img)
                for img in os.listdir(os.path.join(data_dir, class_name))
            ],
            args.num_samples_per_class,
        )
        for class_name in args.target_classes
    }
    print(image_paths)
    save_path = os.path.join(args.save_dir, f"{args.data_split}_{class_names}_grid.pdf")
    create_image_grid(image_paths, save_path)
