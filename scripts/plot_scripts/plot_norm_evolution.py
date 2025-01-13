import os
import pickle
from typing import Any, Dict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")
plt.style.use(["ggplot", "custom_style.mplstyle"])


def plot_norm_evolution(
    snapshot_dict: Dict[float, Dict[str, Any]],
    snapshot_image_dir: str,
    save_dir: str,
    bins: int = 40,
) -> None:
    timesteps = list(snapshot_dict.keys())
    print("timesteps", timesteps)
    timestep_indices = np.array(np.arange(0, 256, 32).tolist() + [255])
    print(f"len(timestep_indices): {len(timestep_indices)}")
    print(f"len(timesteps): {len(timesteps)}")

    colors = cm.get_cmap(
        "cividis", len(timesteps)
    )  # Using a sequential blue color map for a distinct and easy-on-the-eyes gradient
    os.makedirs(save_dir, exist_ok=True)

    img_sample_idx = 2048  # Initialize with a float value
    x_paths = {}
    for key in ["x_cur", "denoised", "d_cur"]:
        x_paths[key] = [
            os.path.join(
                snapshot_image_dir,
                key,
                f"step_{timestep:03d}",
                f"{img_sample_idx:06d}.png",
            )
            for timestep in timestep_indices
        ]
    x_cur_paths, x_denoised_paths, x_d_cur_paths = (
        x_paths["x_cur"],
        x_paths["denoised"],
        x_paths["d_cur"],
    )
    print(f"x_cur_paths: {x_cur_paths}")
    print(f"x_denoised_paths: {x_denoised_paths}")
    for path in x_cur_paths:
        if not os.path.exists(path):
            raise ValueError(f"The file {path} does not exist.")
    for path in x_denoised_paths:
        if not os.path.exists(path):
            raise ValueError(f"The file {path} does not exist.")
    for path in x_d_cur_paths:
        if not os.path.exists(path):
            raise ValueError(f"The file {path} does not exist.")

    # Load images from x_cur_paths and concatenate them into a single row
    images_x_cur = [Image.open(path) for path in x_cur_paths]
    images_x_denoised = [Image.open(path) for path in x_denoised_paths]
    images_x_d_cur = [Image.open(path) for path in x_d_cur_paths]
    widths, heights = zip(*(i.size for i in images_x_cur))

    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image with the total width and max height
    combined_image_x_cur = Image.new("RGB", (total_width, max_height))
    combined_image_x_denoised = Image.new("RGB", (total_width, max_height))
    combined_image_x_d_cur = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in images_x_cur:
        combined_image_x_cur.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in images_x_denoised:
        combined_image_x_denoised.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in images_x_d_cur:
        combined_image_x_d_cur.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # noise_schedule = np.load(os.path.join(snapshot_image_dir, "t_steps.npy"))

    fig = plt.figure(figsize=(12, 16))  # Adjusted width for two columns
    n_subplots = 8
    gs = gridspec.GridSpec(
        n_subplots,
        2,  # Two columns
        height_ratios=[2, 1, 1, 1, 2, 1, 1, 1],  # Adjust height ratios
        width_ratios=[2, 1],  # Make the second column half the width of the first
    )

    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    axs.extend([fig.add_subplot(gs[i, 1]) for i in range(4)])

    print(f"len(axs): {len(axs)}")

    for i in range(len(timesteps)):
        axs[0].scatter(
            timestep_indices[i],
            timesteps[i],
            color=colors(i),
            s=50,
            alpha=1.0,
            edgecolor="black",
            label=rf"$t_{{{timestep_indices[i]}}} = {timesteps[i]:.3f}$",
        )
    axs[0].set_title(r"Noise Schedule $\mathbf{\sigma(t_i) = t_i}$", fontsize=12)
    # axs[0].set_ylabel(
    #     r"$\mathbf{\sigma(t_i) = t_i}$ (log scale)",
    #     fontsize=12,
    # )
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Sampling Step (i)", fontsize=10)

    # Add a new subplot for the legend of axs[0]
    axs[4].legend(
        *axs[0].get_legend_handles_labels(),
        frameon=True,
        loc="center",
        ncol=3,
    )
    axs[4].axis("off")  # Hide the axis

    # Intermediate Images
    axs[1].imshow(combined_image_x_cur)
    axs[1].axis("off")  # Hide the axis
    axs[1].set_title(r"Intermediate Images $\mathbf{x^{(i)}}$", fontsize=12)

    # Output of Denoiser
    axs[2].imshow(combined_image_x_denoised)
    axs[2].axis("off")  # Hide the y-axis for the denoised output image
    axs[2].set_xlabel("Sampling Step", fontsize=10)
    axs[2].set_title(
        r"Output of Denoiser $\mathbf{D_{\theta}}$ After Euler Step",
        fontsize=12,
    )

    # Instantaneous Change
    axs[3].imshow(combined_image_x_d_cur)
    axs[3].axis("off")  # Hide the axis
    axs[3].set_xlabel("Sampling Step", fontsize=10)
    axs[3].set_title(
        r"Instantaneous Change $\mathbf{d_i}$",
        fontsize=12,
    )

    # Norm of x_i
    for i, (timestep, data) in enumerate(snapshot_dict.items()):
        axs[5].hist(
            data["norms"],
            bins=bins,
            color=colors(i),
            alpha=0.9,
            histtype="stepfilled",
            edgecolor="black",
            label=f"Timestep {timestep}",
        )
    axs[5].set_title(r"Norm of $\mathbf{x^{(i)}}$", fontsize=12)
    axs[5].set_ylabel("Count", fontsize=10, fontweight="bold")
    axs[5].set_xlabel(r"$\ell_2$ Norm (log scale)", fontsize=10)
    axs[5].set_xscale("log")

    # Norm of Denoiser output
    for i, (timestep, data) in enumerate(snapshot_dict.items()):
        axs[6].hist(
            data["norms_denoised"],
            bins=bins,
            color=colors(i),
            alpha=0.9,
            edgecolor="black",
            histtype="stepfilled",
            label=f"Timestep {timestep}",
        )
    axs[6].set_title(
        r"Norm of $\mathbf{D_{\theta}}$",
        fontsize=12,
    )
    axs[6].set_ylabel("Count", fontsize=10, fontweight="bold")
    axs[6].set_xlabel(r"$\ell_2$ Norm", fontsize=10)

    # Norm of Instantaneous Change
    for i, (timestep, data) in enumerate(snapshot_dict.items()):
        axs[7].hist(
            data["norms_dxdt"],
            bins=bins,
            color=colors(i),
            alpha=0.9,
            edgecolor="black",
            histtype="stepfilled",
            label=f"Timestep {timestep}",
        )
    # axs[7].set_title(
    #     r"Norm of $\mathbf{\frac{d x}{dt}|_{\hat{t_{i}}}}$",
    #     fontsize=12,
    # )
    axs[7].set_title(
        r"Norm of $\mathbf{d_i}$",
        fontsize=12,
    )
    axs[7].set_ylabel("Count", fontsize=10, fontweight="bold")
    axs[7].set_xlabel(r"$\ell_2$ Norm", fontsize=10)

    # plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

    plt.tight_layout()  # Adjust layout to prevent overlap

    plt.savefig(
        os.path.join(save_dir, "norm_evolution.png"),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    class_name = "goldfinch"
    data_split = "edm_imagenet64_snapshots"
    data_dir = os.path.join(DATA_DIR, data_split, class_name)
    snapshot_dict_paths = [
        os.path.join(data_dir, "snapshot_dict_seeds_0-2047.pkl"),
        os.path.join(data_dir, "snapshot_dict_seeds_2048-10239.pkl"),
    ]
    # snapshot_dict_paths = [
    #     os.path.join(data_dir, "snapshot_dict_seeds_0-1023.pkl"),
    # ]
    # combine snapshot dicts
    snapshot_dict = {}
    for snapshot_dict_path in snapshot_dict_paths:
        curr_snapshot_dict = pickle.load(open(snapshot_dict_path, "rb"))
        for timestep, data in curr_snapshot_dict.items():
            if timestep not in snapshot_dict:
                snapshot_dict[timestep] = data
            else:
                snapshot_dict[timestep]["norms"] = np.concatenate(
                    (snapshot_dict[timestep]["norms"], data["norms"])
                )
                snapshot_dict[timestep]["norms_denoised"] = np.concatenate(
                    (snapshot_dict[timestep]["norms_denoised"], data["norms_denoised"])
                )
                snapshot_dict[timestep]["norms_dxdt"] = np.concatenate(
                    (snapshot_dict[timestep]["norms_dxdt"], data["norms_dxdt"])
                )

    plot_norm_evolution(snapshot_dict, snapshot_image_dir=data_dir, save_dir="figs")
