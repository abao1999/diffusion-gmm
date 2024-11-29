import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm.auto import tqdm

from diffusion_gmm.base import (
    DataPrefetcher,
    MultiClassSubset,
)
from diffusion_gmm.utils import get_targets

logger = logging.getLogger(__name__)


# Define a simple linear multiclass classifier
class LinearMulticlassClassifier(nn.Module):
    """
    Simple linear multiclass classifier
    """

    def __init__(self, num_classes: int, input_dim: int):
        super(LinearMulticlassClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return F.softmax(self.fc(x), dim=1)


# Define a fully connected two-layer network for multiclass classification
class TwoLayerMulticlassClassifier(nn.Module):
    """
    Two-layer fully connected multiclass classifier
    """

    def __init__(self, num_classes: int, input_dim: int, hidden_dim: int):
        super(TwoLayerMulticlassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # Second layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))  # First layer with ReLU activation
        return F.softmax(self.fc2(x), dim=1)  # Second layer with Softmax activation


# Define a simple linear classifier
class LinearBinaryClassifier(nn.Module):
    """
    Simple linear binary classifier
    """

    def __init__(self, num_classes: int, input_dim: int, output_logit: bool = False):
        """
        num_classes must be 2 for binary classification
        if output_logit is True, the output will be logits (unconstrained)
            This is meant to be used with BCEWithLogitsLoss
        Otherwise, the output will be constrained to be between 0 and 1 (probability) and it is recommended to use BCELoss
        """
        if num_classes != 2:
            raise ValueError("num_classes must be 2 for binary classification")
        super(LinearBinaryClassifier, self).__init__()
        self.fc = nn.Linear(
            input_dim, 1, bias=True
        )  # Output 1 for binary classification
        self.output_logit = output_logit

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        if self.output_logit:
            return self.fc(x)
        else:
            return torch.sigmoid(self.fc(x))


# Define a two-layer fully connected binary classifier
class TwoLayerBinaryClassifier(nn.Module):
    """
    Two-layer fully connected binary classifier
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int,
        output_logit: bool = False,
    ):
        """
        num_classes must be 2 for binary classification
        if output_logit is True, the output will be logits (unconstrained)
            This is meant to be used with BCEWithLogitsLoss
        Otherwise, the output will be constrained to be between 0 and 1 (probability) and it is recommended to use BCELoss
        """
        if num_classes != 2:
            raise ValueError("num_classes must be 2 for binary classification")
        super(TwoLayerBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_dim, 1)  # Second layer for binary classification
        self.output_logit = output_logit

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))  # First layer with ReLU activation
        if self.output_logit:
            return self.fc2(x)  # Return logits
        else:
            return torch.sigmoid(self.fc2(x))  # Return probabilities


class ResNet64(nn.Module):
    """
    ResNet model adapted for 64x64 input images
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        pretrained: bool = True,
        output_logit: bool = False,
    ):
        super(ResNet64, self).__init__()
        # Load a pre-defined ResNet model
        self.pretrained = pretrained
        self.resnet = models.resnet18(pretrained=self.pretrained)
        self.output_logit = output_logit

        # No change, but exposing this could be useful
        self.resnet.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Modify the fully connected layer to output the correct number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        if self.output_logit:
            return self.resnet(x)
        else:
            return torch.sigmoid(self.resnet(x))


@dataclass
class ClassifierExperiment:
    """
    Base class for classifier experiments
    """

    input_dim: int
    class_list: List[str]

    # model parameters
    model_class: Type[nn.Module]
    criterion_class: Type[nn.Module]
    optimizer_class: Type[optim.Optimizer] = optim.SGD
    lr_scheduler_class: Optional[Type[lr_scheduler.LRScheduler]] = None

    model_kwargs: Optional[Dict[str, Any]] = None
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None

    # training parameters
    lr: float = 1e-3
    device: Union[torch.device, str] = "cpu"
    verbose: bool = False

    def __post_init__(self):
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device(self.device)
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")
        print("device: ", self.device)

        self.criterion = self.criterion_class()

        self.use_one_hot_enc = False
        if len(self.class_list) == 2:
            # BCELoss and MSELoss require labels to be floats (same type as output)
            # Furthermore, predictions are made by rounding scalar, rather than taking torch.max
            self.use_binary_classifier = True
        else:
            self.use_binary_classifier = False
            if self.criterion_class == nn.MSELoss:
                self.use_one_hot_enc = True

    def make_model_and_optimizer(self):
        self.model = self.model_class(
            num_classes=len(self.class_list),
            input_dim=self.input_dim,
            **(self.model_kwargs or {}),
        ).to(self.device)
        # self.model.to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.lr,  # type: ignore
            **(self.optimizer_kwargs or {}),
        )
        if self.lr_scheduler_class is not None:
            self.scheduler = self.lr_scheduler_class(
                self.optimizer, **(self.scheduler_kwargs or {})
            )
        else:
            self.scheduler = None

    def train(self, dataloader: DataLoader | DataPrefetcher) -> float:
        """
        Train the model for one epoch
        """
        self.model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            if inputs is None or labels is None:
                warnings.warn("Inputs or labels are None")
                logger.warning("Inputs or labels are None")
                continue
            inputs = inputs.to(self.device, non_blocking=True).float()
            if self.use_one_hot_enc:
                labels = F.one_hot(
                    labels.long(), num_classes=len(self.class_list)
                ).float()
            else:
                labels = (
                    labels.to(self.device, non_blocking=True).long()
                    if not self.use_binary_classifier
                    else labels.to(self.device, non_blocking=True).float()
                )
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        n_total = len(dataloader.dataset)  # type: ignore
        avg_loss = running_loss / n_total
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader | DataPrefetcher) -> Tuple[float, float]:
        """
        Evaluate the model on the test set and return the average loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                if inputs is None or labels is None:
                    warnings.warn("Inputs or labels are None")
                    logger.warning("Inputs or labels are None")
                    continue
                inputs = inputs.to(self.device, non_blocking=True).float()
                if self.use_one_hot_enc:
                    labels = F.one_hot(
                        labels.long(), num_classes=len(self.class_list)
                    ).float()
                else:
                    labels = (
                        labels.to(self.device, non_blocking=True).long()
                        if not self.use_binary_classifier
                        else labels.to(self.device, non_blocking=True).float()
                    )
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                if self.use_binary_classifier:
                    preds = torch.round(outputs)  # binary predictions
                else:
                    _, preds = torch.max(outputs, dim=1)

                # Undo one-hot encoding for comparison
                if self.use_one_hot_enc:
                    labels = torch.argmax(labels, dim=1)

                correct += (preds == labels).sum().item()

            n_total = len(dataloader.dataset)  # type: ignore
        avg_loss = running_loss / n_total
        accuracy = correct / n_total
        return avg_loss, accuracy

    def _build_dataloader(
        self,
        subset: Subset,
        n_samples_per_class: int,
        batch_size: int = 64,
        rng: Optional[np.random.Generator] = None,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        make_prefetcher: bool = True,
    ) -> DataLoader | DataPrefetcher:
        # hardcode balance class distribution
        n_samples_to_use = int(n_samples_per_class * len(self.class_list))
        dataset = subset.dataset
        class_to_idx = dataset.class_to_idx  # type: ignore
        # Map class indices to user-specified multi-class classification labels
        class_idx_to_label = {
            class_to_idx[cls]: idx for idx, cls in enumerate(self.class_list)
        }
        if n_samples_to_use < len(subset):
            subset_indices = subset.indices
            targets = get_targets(dataset)  # type: ignore

            # Group indices by class
            class_to_indices = {class_to_idx[cls]: [] for cls in self.class_list}
            for idx in subset_indices:
                class_idx = targets[idx]
                if class_idx in class_to_indices:
                    class_to_indices[class_idx].append(idx)
                else:
                    raise ValueError(
                        f"Class {class_idx} should not be present in dataset_indices."
                    )

            # Shuffle and select a proportion of indices for each class
            prop_to_use = n_samples_to_use / len(subset)
            selected_indices = []
            for class_idx, indices in class_to_indices.items():
                if rng is not None:
                    rng.shuffle(indices)
                subset_size = int(len(indices) * prop_to_use)
                selected_indices.extend(indices[:subset_size])

            selected_subset = MultiClassSubset(
                Subset(dataset, selected_indices),
                class_idx_to_label,
            )
        else:
            selected_subset = MultiClassSubset(subset, class_idx_to_label)

        if self.verbose:
            target_counts = np.bincount([label.item() for _, label in selected_subset])
            logger.info("Built dataloader with %d samples...", len(selected_subset))
            logger.info("Distribution of targets in subset: %s", target_counts)

        dataloader = DataLoader(
            selected_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=str(self.device),
            persistent_workers=persistent_workers,
        )
        if make_prefetcher:
            dataloader = DataPrefetcher(dataloader, self.device)

        return dataloader

    def run(
        self,
        train_subset: Subset,
        test_subset: Subset,
        rng: np.random.Generator,
        n_train_per_class_schedule: Sequence[int],
        n_test_samples_per_class: int,
        num_epochs: int = 20,
        eval_epoch_interval: int = 50,
        early_stopping_patience: int = 100,
        batch_size: int = 64,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        model_save_dir: Optional[str] = None,
        model_save_name: Optional[str] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run the classifier experiment
        """
        if early_stopping_patience < eval_epoch_interval:
            raise ValueError(
                "early_stopping_patience must be greater than or equal to eval_epoch_interval"
            )
        test_loader = self._build_dataloader(
            test_subset,
            batch_size=batch_size,
            **(dataloader_kwargs or {}),
            rng=rng,
            n_samples_per_class=int(n_test_samples_per_class),
        )

        if verbose:
            logger.info(
                f"Built test dataloader with {len(test_loader.dataset)} samples from the test split...",  # type: ignore
            )

        results: List[Dict[str, Any]] = [{}] * len(n_train_per_class_schedule)
        for i, n_train_per_class in tqdm(
            enumerate(n_train_per_class_schedule),
            desc="Training on different numbers of training samples",
        ):
            # reset the model and optimizer and scheduler to the initial state
            self.make_model_and_optimizer()
            n_train_per_class = int(n_train_per_class)
            # build train dataloader
            train_loader = self._build_dataloader(
                train_subset,
                batch_size=batch_size,
                n_samples_per_class=n_train_per_class,
                rng=rng,
                **(dataloader_kwargs or {}),
            )

            if verbose:
                logger.info(f"n_train_per_class {n_train_per_class}")
                logger.info(
                    f"Building train dataloader using {len(train_loader.dataset)} samples from the training split..."  # type: ignore
                )

            best_test_loss = float("inf")
            early_stopping_counter = 0

            for epoch in tqdm(range(num_epochs), desc="Training"):
                train_loss = self.train(train_loader)
                if epoch % eval_epoch_interval == 0 or epoch == num_epochs - 1:
                    test_loss, accuracy = self.evaluate(test_loader)
                    if verbose:
                        logger.info(
                            "Epoch %d, Train Loss: %f, Test Loss: %f, Accuracy: %f%%",
                            epoch,
                            train_loss,
                            test_loss,
                            accuracy * 100,
                        )
                        logger.info(
                            f"learning rate: {self.optimizer.param_groups[0]['lr']}"
                        )

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        results[i] = {
                            "num_train_samples": len(train_loader.dataset),  # type: ignore
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "accuracy": accuracy,
                            "epoch": epoch,
                        }
                        early_stopping_counter = 0  # Reset counter if improvement
                    else:
                        early_stopping_counter += eval_epoch_interval

                    if early_stopping_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break

                if self.scheduler is not None:
                    self.scheduler.step()

            if model_save_dir is not None:
                model_save_name = f"n{n_train_per_class}_c{len(self.class_list)}.pth"
                model_save_path = os.path.join(model_save_dir, model_save_name)
                logger.info(f"Saving model to {model_save_path}")
                os.makedirs(model_save_dir, exist_ok=True)
                torch.save(
                    {
                        "best_test_loss": best_test_loss,
                        "last_epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    model_save_path,
                )

        return results
