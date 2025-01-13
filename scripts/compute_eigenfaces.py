import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from diffusion_gmm.utils import setup_dataset

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")

plt.style.use(["ggplot", "custom_style.mplstyle"])


def compute_eigenfaces_svd(dataloader, device="cpu"):
    # Collect all images into a single matrix
    images = []
    for batch in dataloader:
        imgs, _ = batch
        imgs = imgs.to(device)
        images.append(imgs.view(imgs.size(0), -1))

    img_shape = imgs.shape[1:]
    print(f"Image shape: {img_shape}")

    # Concatenate all images into a single matrix
    images_matrix = torch.cat(images, dim=0).cpu().numpy()

    # Compute the mean image
    mean_image = np.mean(images_matrix, axis=0)

    # Center the images by subtracting the mean image
    centered_images = images_matrix - mean_image
    print(centered_images.shape)

    # # Perform truncated SVD
    # U, S, Vt = svds(centered_images, k=100, which="LM", return_singular_vectors=True)
    # Perform SVD
    U, S, Vt = np.linalg.svd(centered_images, full_matrices=False)
    assert all(var is not None for var in (U, S, Vt))

    print(U.shape, S.shape, Vt.shape)  # type: ignore

    # Compute the total variance explained
    # total_variance_explained = S**2 / np.sum(S**2)
    mean_image = mean_image.reshape(1, *img_shape)

    return mean_image, U, S, Vt


def compute_eigenfaces(dataloader, device="cpu"):
    # Collect all images into a single matrix
    images = []
    for batch in dataloader:
        # Assuming batch is a tuple (images, labels)
        imgs, _ = batch
        imgs = imgs.to(device)
        # Flatten each image and add to the list
        images.append(imgs.view(imgs.size(0), -1))

    # Concatenate all images into a single matrix
    images_matrix = torch.cat(images, dim=0).cpu().numpy()

    # Compute the mean image
    mean_image = np.mean(images_matrix, axis=0)

    # Center the images by subtracting the mean image
    centered_images = images_matrix - mean_image

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_images, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Return the eigenvectors as eigenfaces
    return eigenvectors


def visualize_eigenfaces(data_dir, class_list, plot_save_dir):
    for class_name in class_list:
        eigenface_path = os.path.join(data_dir, f"{class_name}_eigenfaces.npy")
        projections_path = os.path.join(data_dir, f"{class_name}_projections.npy")
        eigenvalues_path = os.path.join(data_dir, f"{class_name}_eigenvalues.npy")
        mean_image_path = os.path.join(data_dir, f"{class_name}_mean_image.npy")

        eigenfaces = np.load(eigenface_path)
        projections = np.load(projections_path)
        eigenvalues = np.load(eigenvalues_path)
        mean_image = np.load(mean_image_path)
        print(
            f"eigenfaces shape: {eigenfaces.shape}, eigenvalues shape: {eigenvalues.shape}"
        )
        print(f"mean image shape: {mean_image.shape}")
        print(f"projections shape: {projections.shape}")

        n = eigenfaces.shape[0]
        eigenvalues_gram = eigenvalues**2 / n
        explained_variances = eigenvalues_gram / np.sum(eigenvalues_gram)

        img_shape = mean_image.shape[1:]
        print(f"img shape: {img_shape}")
        fig = plt.figure(figsize=(12, 10))  # Adjusted figure size for more rows
        for i in range(12):  # Plot 12 eigenfaces
            plt.subplot(4, 6, i + 1)  # Adjusted to accommodate 4 rows and 6 columns
            # Normalize the eigenface for visualization
            eigenface = eigenfaces[i]
            eigenface_min, eigenface_max = eigenface.min(), eigenface.max()
            eigenface_normalized = (eigenface - eigenface_min) / (
                eigenface_max - eigenface_min
            )

            plt.imshow(eigenface_normalized.reshape(*img_shape).transpose(1, 2, 0))
            plt.title(f"Eigenface {i+1}", fontweight="bold")
            plt.axis("off")

        # Plot mean image
        plt.subplot2grid((4, 6), (2, 0))  # Moved to the third row
        plt.imshow(mean_image.reshape(*img_shape).transpose(1, 2, 0))
        plt.title("Mean Image", fontweight="bold")
        plt.axis("off")

        # Plot eigenvalues as a subplot spanning 2 columns and 1 row in the third row
        plt.subplot2grid((4, 6), (2, 1), colspan=2, rowspan=2)  # Adjusted position
        plt.plot(explained_variances[:12] * 100, marker="o")  # Adjusted to plot 12
        plt.title("Explained Variance (%)", fontweight="bold")
        plt.xlabel("Eigenface Index")
        plt.grid()

        # Plot projections as a subplot spanning 3 columns and 2 rows in the third and fourth rows
        ax = plt.subplot2grid(
            (4, 6), (2, 3), colspan=3, rowspan=2, projection="3d"
        )  # Adjusted colspan and rowspan
        ax.scatter(
            projections[:, 0],
            projections[:, 1],
            projections[:, 2],
            c=projections[:, 0],
            cmap="viridis",
            alpha=0.7,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")  # type: ignore
        ax.set_title("Projections", fontweight="bold")

        plt.savefig(os.path.join(plot_save_dir, f"{class_name}_eigenfaces.pdf"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_classes", type=str, nargs="+", required=True)
    parser.add_argument("--data_split", type=str, default="edm_imagenet64_all")
    parser.add_argument("--n_samples_per_class", type=int, default=1024)
    parser.add_argument("--plot_save_dir", type=str, default="final_plots/eigenfaces")
    parser.add_argument("--plot_name", type=str, default=None)
    parser.add_argument(
        "--data_save_dir", type=str, default=os.path.join(DATA_DIR, "eigenfaces")
    )
    args = parser.parse_args()

    plot_save_dir = os.path.join(args.plot_save_dir, args.data_split)
    os.makedirs(plot_save_dir, exist_ok=True)

    save_dir = os.path.join(args.data_save_dir, args.data_split)
    os.makedirs(save_dir, exist_ok=True)

    class_list = args.target_classes
    data_dir = os.path.join(DATA_DIR, args.data_split)

    # visualize_eigenfaces(save_dir, class_list, plot_save_dir)
    # exit()

    print("Setting up dataset...")
    dataset, is_npy_dataset = setup_dataset(data_dir)
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None
    print("Dataset setup complete.")

    # NOTE: has to be ImageFolder to have samples attribute.
    image_paths, targets = zip(*dataset.samples)
    targets = np.array(targets)
    # targets = get_targets(dataset) # for case when dataset is not ImageFolder
    class_to_idx = dataset.class_to_idx
    indices_by_class = {
        cls: np.where(targets == class_to_idx[cls])[0].tolist() for cls in class_list
    }
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    for class_name, indices in tqdm(
        indices_by_class.items(), desc="Processing classes"
    ):
        print(f"Class {class_name} has {len(indices)} samples")

        selected_indices = indices[: args.n_samples_per_class]
        print(f"Selected {len(selected_indices)} samples for class {class_name}")
        class_subset = Subset(dataset, selected_indices)
        dataloader = DataLoader(
            class_subset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
        )

        print(f"Computing eigenfaces for {class_name}")
        mean_image, U, S, Vt = compute_eigenfaces_svd(dataloader, device="cpu")
        save_path_eigenfaces = os.path.join(save_dir, f"{class_name}_eigenfaces.npy")
        save_path_projections = os.path.join(save_dir, f"{class_name}_projections.npy")
        save_path_eigenvalues = os.path.join(save_dir, f"{class_name}_eigenvalues.npy")
        save_path_mean_image = os.path.join(save_dir, f"{class_name}_mean_image.npy")
        np.save(save_path_eigenfaces, Vt)
        np.save(save_path_projections, U)
        np.save(save_path_eigenvalues, S)
        np.save(save_path_mean_image, mean_image)
