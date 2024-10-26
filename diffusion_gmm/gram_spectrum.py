import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from tqdm.auto import tqdm


@dataclass
class GramSpectrumExperiment:
    data_dir: str
    dataset_name: Optional[str] = None
    batch_size: int = 64
    custom_transform: Optional[Callable] = None
    verbose: bool = False
    load_npy: bool = False
    rseed: int = 99

    def __post_init__(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        self.rng = np.random.default_rng(self.rseed)

        if self.custom_transform is None:
            self.custom_transform = transforms.Compose([transforms.ToTensor()])

        self.data = self._get_dataset()
        self.classes = self.data.classes  # type: ignore
        self.class_to_idx: Dict[str, int] = self.data.class_to_idx  # type: ignore
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.img_shape = self._get_sample_shape()

        if self.verbose:
            print("Data directory: ", self.data_dir)
            print("Using custom transform: ", self.custom_transform)
            print("Transform: ", self.custom_transform)

    @staticmethod
    def compute_gram_matrix(features: np.ndarray) -> np.ndarray:
        b, c, h, w = features.shape
        features = features.reshape(b, c, h * w)
        gram_matrix = np.matmul(features, features.transpose(0, 2, 1))
        # features = features.view(b, c, h * w)
        # gram_matrix = torch.bmm(features, features.transpose(1, 2))
        return gram_matrix

    @staticmethod
    def get_gram_spectrum(gram_matrix: np.ndarray) -> np.ndarray:
        """
        Get the eigenvalues of the Gram matrix
        """
        eigenvalues = np.linalg.eigvals(gram_matrix)
        return eigenvalues.real.flatten()

    def _get_dataset(self) -> VisionDataset:
        if self.dataset_name is None:
            if self.load_npy:

                def npy_loader(path):
                    sample = torch.from_numpy(np.load(path))
                    return sample

                data = DatasetFolder(
                    root=self.data_dir,
                    loader=npy_loader,
                    extensions=(".npy",),
                )
            else:
                data = ImageFolder(
                    root=self.data_dir,
                    transform=self.custom_transform,
                    target_transform=None,
                )

        elif self.dataset_name == "cifar10":
            # Load the real CIFAR10 train split from torchvision
            data = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.custom_transform,
            )
        elif self.dataset_name == "imagenet":
            data = datasets.ImageNet(
                root=self.data_dir,
                split="train",
                download=True,
                transform=self.custom_transform,
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

        return data

    def _get_sample_shape(self) -> Tuple[int, int, int]:
        first_sample = next(iter(DataLoader(self.data)))[0].squeeze()
        shape = tuple(first_sample.shape)
        print("Image shape: ", shape)
        assert len(shape) == 3, "Images should have 3 dimensions (C, H, W)"
        if not shape[0] == 3:
            warnings.warn(
                f"Images should have 3 color channels. Found {shape[0]} instead."
            )
        return shape

    def _build_dataloader(
        self, num_samples: int, target_class: Optional[str] = None
    ) -> DataLoader:
        num_tot_samples = len(self.data)
        if target_class is None:
            if num_samples > num_tot_samples:
                indices = list(range(num_tot_samples))
            else:
                indices = self.rng.choice(
                    num_tot_samples, num_samples, replace=False
                ).tolist()
        else:
            # Convert target_class from string to integer index
            target_class_idx = self.class_to_idx[target_class]

            # For datasets like CIFAR10, use targets attribute
            if hasattr(self.data, "targets"):
                targets = self.data.targets  # type: ignore
            # For ImageFolder, reconstruct targets from samples
            elif hasattr(self.data, "samples"):
                targets = [class_idx for _, class_idx in self.data.samples]  # type: ignore
            else:
                raise AttributeError(
                    "Dataset doesn't have 'targets' or 'samples' attribute"
                )

            indices = []
            for idx, class_idx in enumerate(targets):
                if len(indices) == num_samples:
                    break
                if class_idx == target_class_idx:
                    indices.append(idx)

        if self.verbose:
            print(f"Sampling from {len(indices)} valid samples")

        custom_sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(
            self.data, batch_size=self.batch_size, shuffle=False, sampler=custom_sampler
        )
        return dataloader

    def run(
        self,
        num_samples: int,
        save_dir: Union[str, Path],
        save_name: str,
        target_class: Optional[str] = None,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        dataloader = self._build_dataloader(num_samples, target_class)
        all_eigenvalues = []
        unique_labels = set()

        for _, (images, labels) in tqdm(enumerate(dataloader)):
            data = images.squeeze().cpu().numpy()
            print("data shape: ", data.shape)
            gram_matrix = self.compute_gram_matrix(data)
            spectrum = self.get_gram_spectrum(gram_matrix)
            all_eigenvalues.extend(spectrum)
            unique_labels.update(labels.cpu().numpy())

        all_eigenvalues = np.array(all_eigenvalues)
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, all_eigenvalues)

        if self.verbose:
            print("Unique labels: ", unique_labels)
            print("All eigenvalues shape: ", all_eigenvalues.shape)
            print("Savied gram spectrum to: ", save_path)


@dataclass
class GramSpectrumCNNExperiment(GramSpectrumExperiment):
    """
    Compute the Gram spectrum of a pre-trained CNN model's features on a dataset
    Common CNN input dimensions are 3x224x224, 3x299x299
    """

    cnn_model_id: str = "vgg16"
    hook_layer: int = 10

    def __post_init__(self):
        super().__post_init__()
        try:
            self.model = getattr(tv_models, self.cnn_model_id)(
                pretrained=True
            ).features.eval()
        except AttributeError:
            raise ValueError(f"Invalid CNN model ID: {self.cnn_model_id}")

        self.features = []

        def hook(module, input, output):
            self.features.append(output)

        # Attach the hook to a specific layer
        self.model[self.hook_layer].register_forward_hook(hook)

        if self.verbose:
            print(f"Using CNN model: {self.cnn_model_id}")
            print("Hook attached to layer: ", self.hook_layer)

    @torch.no_grad()
    def run(
        self,
        num_samples: int,
        save_dir: Union[str, Path],
        save_name: str,
        target_class: Optional[str] = None,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        dataloader = self._build_dataloader(num_samples, target_class)
        all_eigenvalues = []
        unique_labels = set()

        for _, (images, labels) in tqdm(enumerate(dataloader)):
            self.features.clear()  # Clear previous features
            with torch.no_grad():
                self.model(images)  # Forward pass through the model
            # use first features from from hook
            feats = self.features[0].squeeze().cpu().numpy()
            print("feats shape: ", feats.shape)
            gram_matrix = self.compute_gram_matrix(feats)
            spectrum = self.get_gram_spectrum(gram_matrix)
            all_eigenvalues.extend(spectrum)
            unique_labels.update(labels.cpu().numpy())

        all_eigenvalues = np.array(all_eigenvalues)
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, all_eigenvalues)

        if self.verbose:
            print("Unique labels: ", unique_labels)
            print("All eigenvalues shape: ", all_eigenvalues.shape)
            print("Savied gram spectrum to: ", save_path)
