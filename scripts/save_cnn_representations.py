import logging
import os

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from diffusion_gmm.utils import setup_dataset


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    class_list = cfg.cnn.class_list
    logger.info(f"Class list: {class_list}")
    data_dir = cfg.cnn.data_dir
    save_dir = cfg.cnn.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Save directory: {save_dir}")

    # Load the pretrained cnn model
    model_id = cfg.cnn.model_id
    model = getattr(models, model_id)(pretrained=True)
    logger.info(f"Loaded model: {model_id}")

    # Modify the model to output features after the average pooling layer (remove last FC layer)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()  # Set model to evaluation mode

    print(model)
    device = torch.device(cfg.cnn.device)
    model = model.to(device)

    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset, is_npy_dataset = setup_dataset(data_dir)
    if is_npy_dataset:
        raise ValueError("Numpy dataset not supported for cnn feature extraction")
    dataset.transform = transform
    # dataset = datasets.ImageFolder(root=data_dir, transform=transform)

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

    with torch.no_grad():
        for class_name, indices in tqdm(
            indices_by_class.items(), desc="Processing classes"
        ):
            print(f"Class {class_name} has {len(indices)} samples")
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            selected_indices = indices[: cfg.cnn.n_samples_per_class]
            print(f"Selected {len(selected_indices)} samples for class {class_name}")
            class_subset = Subset(dataset, selected_indices)
            batch_size = cfg.cnn.batch_size or 1  # Default to 1 if None
            dataloader = DataLoader(
                class_subset,
                batch_size=batch_size,
                shuffle=False,  # this needs to be False to match image paths
                num_workers=cfg.cnn.num_workers,
            )

            for i, (images, labels) in tqdm(
                enumerate(dataloader), desc=f"Processing batches for class {class_name}"
            ):
                images = images.to(device)
                output = model(images)  # Output shape: (batch_size, 2048, 1, 1)
                output = output.view(
                    output.size(0), -1
                )  # Flatten to (batch_size, 2048)
                if i == 0:
                    print(f"representation shape: {output.shape}")

                # Save features and labels in a folder structure
                batch_image_paths = image_paths[
                    i * batch_size : min((i + 1) * batch_size, len(image_paths))
                ]
                for feature, label, img_path in zip(
                    output.cpu(), labels, batch_image_paths
                ):
                    if class_name != idx_to_class[label.item()]:
                        raise ValueError(
                            f"Class mismatch: {class_name} != {idx_to_class[label.item()]}"
                        )
                    # Use the original image name for the feature file
                    sample_idx = os.path.splitext(os.path.basename(img_path))[0]
                    if cfg.cnn.save_as_pt:
                        feature_path = os.path.join(class_dir, f"{sample_idx}.pt")
                        torch.save(feature, feature_path)
                    else:
                        feature_path = os.path.join(class_dir, f"{sample_idx}.npy")
                        np.save(feature_path, feature)
                # print(
                #     f"Saved {len(batch_image_paths)} feature representations for class {class_name} to {class_dir}"
                # )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
