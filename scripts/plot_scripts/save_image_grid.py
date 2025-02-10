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
    print(num_classes, num_images_per_class)
    if save_as_image:
        sample = Image.open(next(iter(image_paths.values()))[0])
        w, h = sample.size
        # Calculate the number of blocks needed per class
        block_size = (1, 6)
        n_blocks_per_row = 1
        # Calculate the total number of rows needed
        total_rows = (
            num_classes * block_size[0] + n_blocks_per_row - 1
        ) // n_blocks_per_row
        # Create a new RGB image to paste the images onto
        new_im = Image.new(
            "RGB",
            (
                block_size[1] * w * n_blocks_per_row,
                total_rows * h,
            ),
        )

        for class_idx, (class_name, paths) in enumerate(image_paths.items()):
            print(class_name)
            for img_idx, img_path in enumerate(paths[: block_size[0] * block_size[1]]):
                img = Image.open(img_path)
                # Calculate position to paste the image in a block
                block_row = img_idx // block_size[1]
                block_col = img_idx % block_size[1]
                # Calculate the position based on the class index and block position
                i = ((class_idx // n_blocks_per_row) * block_size[0] + block_row) * h
                j = ((class_idx % n_blocks_per_row) * block_size[1] + block_col) * w
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


# def create_image_grid_single_class(
#     image_paths: Dict[str, List[str]],
#     save_path: str,
#     title: str = "Imagenet 64x64",
#     save_as_image: bool = True,
# ) -> None:
#     sample = Image.open(next(iter(image_paths.values()))[0])
#     w, h = sample.size
#     new_im = Image.new("RGB", (w, h))
#     new_im.save(save_path)
#     n_rows = 1
#     n_cols = 1
#     new_img = Image.new("RGB", (w, h))
#     for img_path in image_paths[class_name]:
#         img = Image.open(img_path)
#         new_img.paste(img, (0, 0))
#     new_img.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, required=True)
    parser.add_argument("--target_classes", type=str, nargs="+", required=True)
    parser.add_argument("--num_samples_per_class", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="final_plots")
    args = parser.parse_args()

    if args.target_classes == ["all"]:
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
    data_dir = os.path.join(DATA_DIR, args.data_split)

    class_names = "-".join(class_list)
    image_paths = {
        class_name: random.sample(
            [
                os.path.join(data_dir, class_name, img)
                for img in os.listdir(os.path.join(data_dir, class_name))
            ],
            args.num_samples_per_class,
        )
        for class_name in class_list
    }
    print(image_paths)
    # save_path = os.path.join(args.save_dir, f"{args.data_split}_{class_names}_grid.pdf")
    save_path = os.path.join(args.save_dir, f"sampling_{class_names}_grid.pdf")
    create_image_grid(image_paths, save_path)
