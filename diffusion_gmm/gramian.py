import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchvision.models as tv_models
from tqdm.auto import tqdm

from diffusion_gmm.base import ImageExperiment


@dataclass
class GramSpectrumExperiment(ImageExperiment):
    def __post_init__(self):
        super().__post_init__()
        self.batch_size = 64

    @staticmethod
    def compute_gram_matrix(features: np.ndarray) -> np.ndarray:
        b, c, h, w = features.shape
        features = features.reshape(b, 1, -1)
        gram_matrix = np.matmul(features, features.transpose(0, 2, 1))
        return gram_matrix

    @staticmethod
    def get_gram_spectrum(gram_matrix: np.ndarray) -> np.ndarray:
        """
        Get the eigenvalues of the Gram matrix
        """
        eigenvalues = np.linalg.eigvals(gram_matrix)
        return eigenvalues.real.flatten()

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
            breakpoint()
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
