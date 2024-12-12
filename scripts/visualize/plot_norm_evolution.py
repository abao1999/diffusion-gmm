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
    combined: bool = True,
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
    num_timesteps = len(timesteps)
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

    if combined:
        fig = plt.figure(figsize=(8, 14))  # 16
        n_subplots = 7
        gs = gridspec.GridSpec(
            n_subplots,
            1,
            height_ratios=[2, 1, 1, 1, 1, 1, 1],  # 3
        )  # Adjust height ratios

        axs = [fig.add_subplot(gs[i]) for i in range(n_subplots)]

        # Plot all norms on the first row
        for i in range(len(timesteps)):
            axs[0].scatter(
                timestep_indices[i],
                timesteps[i],
                color=colors(i),
                s=50,
                alpha=1.0,
                edgecolor="black",
                label=rf"$\sigma_{{{timestep_indices[i]}}} = {timesteps[i]:.3f}$",
            )
        axs[0].set_title("Noise Schedule", fontsize=12)
        axs[0].set_ylabel(
            r"$\mathbf{\sigma(t_i) = t_i}$ (log scale)",
            fontsize=12,
        )
        axs[0].set_yscale("log")
        axs[0].set_xlabel("Sampling Step", fontsize=10)
        axs[0].legend(frameon=True, loc="lower left")

        axs[1].imshow(combined_image_x_cur)
        axs[1].axis("off")  # Hide the axis
        axs[1].set_title("Intermediate Images $\mathbf{x_i}$", fontsize=12)

        for i, (timestep, data) in enumerate(snapshot_dict.items()):
            axs[2].hist(
                data["norms"],
                bins=bins,
                color=colors(i),
                alpha=0.9,
                histtype="stepfilled",
                edgecolor="black",
                label=f"Timestep {timestep}",
            )
        axs[2].set_title(r"Norm of $\mathbf{x_i}$", fontsize=12)
        axs[2].set_ylabel("Count", fontsize=10, fontweight="bold")
        axs[2].set_xlabel(r"$\ell_2$ Norm (log scale)", fontsize=10)
        axs[2].set_xscale("log")

        axs[3].imshow(combined_image_x_denoised)
        # axs[3].axis("off")  # Hide the axis
        axs[3].axis("off")  # Hide the y-axis for the denoised output image
        axs[3].set_xlabel("Sampling Step", fontsize=10)
        axs[3].set_title(
            r"Output of Denoiser $\mathbf{D_{\theta}({x_{i+1}, t_{i+1}})}$",
            fontsize=12,
        )

        for i, (timestep, data) in enumerate(snapshot_dict.items()):
            axs[4].hist(
                data["norms_denoised"],
                bins=bins,
                color=colors(i),
                alpha=0.9,
                edgecolor="black",
                histtype="stepfilled",
                label=f"Timestep {timestep}",
            )
        axs[4].set_title(
            r"Norm of $\mathbf{D_{\theta}({x_{i+1}, t_{i+1}})}$",
            fontsize=12,
        )
        axs[4].set_ylabel("Count", fontsize=10, fontweight="bold")
        axs[4].set_xlabel(r"$\ell_2$ Norm", fontsize=10)

        for i, (timestep, data) in enumerate(snapshot_dict.items()):
            axs[6].hist(
                data["norms_dxdt"],
                bins=bins,
                color=colors(i),
                alpha=0.9,
                edgecolor="black",
                histtype="stepfilled",
                label=f"Timestep {timestep}",
            )
        axs[5].imshow(combined_image_x_d_cur)
        axs[5].axis("off")  # Hide the axis
        axs[5].set_xlabel("Sampling Step", fontsize=10)
        axs[5].set_title(
            r"Instantaneous Change $\mathbf{\frac{d x}{dt}|_{\hat{t_{i}}}}$",
            fontsize=12,
        )
        axs[6].set_title(
            r"Norm of $\mathbf{\frac{d x}{dt}|_{\hat{t_{i}}}}$",
            fontsize=12,
        )
        axs[6].set_ylabel("Count", fontsize=10, fontweight="bold")
        axs[6].set_xlabel(r"$\ell_2$ Norm", fontsize=10)

    else:
        fig = plt.figure(figsize=(15, 9))
        # Create a GridSpec with 3 rows and num_timesteps columns
        gs = gridspec.GridSpec(3, num_timesteps, figure=fig)

        # Adjust the first row to have only one column
        ax0 = fig.add_subplot(gs[0, :])  # Span all columns in the first row

        # Scatter plot of timesteps in the first row
        timestep_indices = np.arange(0, 256, 32)
        for i in range(len(timesteps)):
            ax0.scatter(
                timestep_indices[i],
                timesteps[i],
                color=colors(i),
                s=50,
                label=rf"$\sigma_{{{timestep_indices[i]}}} = {timesteps[i]:.4f}$",
            )
        ax0.set_title("Noise Schedule")
        ax0.set_ylabel(r"$\sigma(t_i) = t_i$")
        ax0.set_xlabel("Sampling Step")
        ax0.legend()
        # Keep the last two rows with the current configuration
        for i, (timestep, data) in enumerate(snapshot_dict.items()):
            # First row for norms
            ax1 = fig.add_subplot(gs[1, i])
            ax1.hist(data["norms"], bins=bins, color=colors(i))
            if i == 0:
                ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
            if i == len(timesteps) // 2:
                ax1.set_title("Norm of $x_i$")

            # Second row for norms_denoised
            ax2 = fig.add_subplot(gs[2, i])
            ax2.hist(data["norms_denoised"], bins=bins, color=colors(i))
            if i == 0:
                ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            if i == len(timesteps) // 2:
                ax2.set_title(r"Norm of Denoiser $D_{\theta}({x_{i+1}, t_{i+1}})$")

    plt.tight_layout()  # Adjust layout to prevent overlap

    plt.savefig(os.path.join(save_dir, "norm_evolution.png"), dpi=300)


if __name__ == "__main__":
    class_name = "goldfinch"
    data_split = "edm_imagenet64_snapshots"
    data_dir = os.path.join(DATA_DIR, data_split, class_name)
    snapshot_dict_paths = [
        os.path.join(data_dir, "snapshot_dict_seeds_0-2047.pkl"),
        os.path.join(data_dir, "snapshot_dict_seeds_2048-10239.pkl"),
    ]
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

    plot_norm_evolution(snapshot_dict, snapshot_image_dir=data_dir, save_dir="figs")
