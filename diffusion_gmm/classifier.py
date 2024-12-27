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

import wandb
from diffusion_gmm.base import (
    DataPrefetcher,
    MultiClassSubset,
)
from diffusion_gmm.utils import get_targets

logger = logging.getLogger(__name__)


class LinearMulticlassClassifier(nn.Module):
    """
    Simple linear multiclass classifier
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        output_logit: bool = False,
        use_bias: bool = False,
    ):
        super(LinearMulticlassClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=use_bias)
        self.output_logit = output_logit

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        if self.output_logit:
            return self.fc(x)
        else:
            return F.softmax(self.fc(x), dim=1)


class TwoLayerMulticlassClassifier(nn.Module):
    """
    Two-layer fully connected multiclass classifier
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int,
        output_logit: bool = False,
        use_bias: bool = False,
    ):
        super(TwoLayerMulticlassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)  # First layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # Second layer
        self.output_logit = output_logit

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))  # First layer with ReLU activation
        if self.output_logit:
            return self.fc2(x)
        else:
            return F.softmax(self.fc2(x), dim=1)  # Second layer with Softmax activation


class LinearBinaryClassifier(nn.Module):
    """
    Simple linear binary classifier
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        output_logit: bool = False,
        use_bias: bool = False,
    ):
        if num_classes != 2:
            raise ValueError("num_classes must be 2 for binary classification")
        super(LinearBinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1, bias=use_bias)
        self.output_logit = output_logit

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        if self.output_logit:
            return self.fc(x)
        else:
            return torch.sigmoid(self.fc(x))


class ResNet(nn.Module):
    """
    ResNet model adapted for 64x64 input images
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        model_id: str,
        pretrained: bool = True,
        output_logit: bool = False,
        use_bias: bool = False,
    ):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.resnet = getattr(models, model_id)(pretrained=self.pretrained)
        self.output_logit = output_logit

        if input_dim != (3 * 224 * 224):
            warnings.warn(
                f"input_dim {input_dim} is not the standard input size for ResNet"
            )

        # No change, but could be useful to expose: ResNet uses this modification of a traditional conv2d layer
        self.resnet_conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,  # Resnet handles biases with Batch Normalization layers that follow convolutions
        )

        if self.pretrained:
            # Freeze all parameters except the last layer
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        else:
            # If not pretrained, make all parameters trainable
            for param in self.resnet.parameters():
                param.requires_grad = True

        # Modify the fully connected layer to output the correct number of classes, this is what we train
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, num_classes, bias=use_bias
        )

    def forward(self, x):
        if self.output_logit:
            return self.resnet(x)
        else:
            return torch.softmax(self.resnet(x), dim=1)


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
        print(f"use_one_hot_enc: {self.use_one_hot_enc}")

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

    def train(self, dataloader: Union[DataLoader, DataPrefetcher]) -> float:
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
    def evaluate(
        self, dataloader: Union[DataLoader, DataPrefetcher]
    ) -> Tuple[float, float]:
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
    ) -> Union[DataLoader, DataPrefetcher]:
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
            indices_by_class = {class_to_idx[cls]: [] for cls in self.class_list}
            for idx in subset_indices:
                class_idx = targets[idx]
                if class_idx in indices_by_class:
                    indices_by_class[class_idx].append(idx)
                else:
                    raise ValueError(
                        f"Class {class_idx} should not be present in dataset_indices."
                    )

            # Shuffle and select a proportion of indices for each class
            prop_to_use = n_samples_to_use / len(subset)
            selected_indices = []
            for class_idx, indices in indices_by_class.items():
                subset_size = int(len(indices) * prop_to_use)
                if rng is not None:
                    selected_indices.extend(
                        rng.choice(indices, subset_size, replace=False)
                    )
                else:
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
            logger.info(
                "Distribution of targets in subset: %s",
                dict(zip(self.class_list, target_counts)),
            )

        dataloader = DataLoader(
            selected_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=str(self.device),
            persistent_workers=persistent_workers,
        )
        # wrap in prefetcher if requested
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
        num_epochs: int = 200,
        eval_epoch_interval: int = 5,
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
                    f"Built train dataloader using {len(train_loader.dataset)} samples from the training split..."  # type: ignore
                )
                logger.info(f"Model: {self.model}")
                logger.info(f"Model parameters: {self.model.parameters()}")
                logger.info(f"Criterion: {self.criterion}")
                logger.info(f"Optimizer: {self.optimizer}")
                logger.info(
                    f"Optimizer parameters: {self.optimizer.param_groups[0]['params']}"
                )
                logger.info(f"Scheduler: {self.scheduler}")

            best_test_loss = float("inf")
            best_acc = 0.0
            early_stopping_counter = 0

            for epoch in tqdm(range(num_epochs), desc="Training"):
                train_loss = self.train(train_loader)
                if epoch % eval_epoch_interval == 0 or epoch == num_epochs - 1:
                    test_loss, accuracy = self.evaluate(test_loader)
                    curr_lr = self.optimizer.param_groups[0]["lr"]
                    if verbose:
                        logger.info(
                            "Epoch %d, Train Loss: %f, Test Loss: %f, Accuracy: %f%%, learning rate: %f",
                            epoch,
                            train_loss,
                            test_loss,
                            accuracy * 100,
                            curr_lr,
                        )

                    if wandb.run is not None:
                        print(f"logging to wandb run {wandb.run.id}")
                        wandb.log(
                            {
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                                "accuracy": accuracy,
                                "learning_rate": curr_lr,
                            },
                            step=epoch,
                        )
                        print(f"best test loss: {best_test_loss}")

                    if accuracy > best_acc:
                        best_acc = accuracy

                    if test_loss <= 0.998 * best_test_loss:
                        best_test_loss = test_loss
                        results[i] = {
                            "n_train_per_class": n_train_per_class,
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "accuracy": accuracy,
                            "best_acc": best_acc,
                            "epoch": epoch,
                        }
                        print("reset early stopping counter")
                        early_stopping_counter = 0  # Reset counter if improvement
                    else:
                        early_stopping_counter += eval_epoch_interval
                        print(f"early_stopping_counter: {early_stopping_counter}")

                    # early stopping conditions
                    if early_stopping_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        logger.info(f"Best test loss: {best_test_loss}")
                        break

                    if train_loss < 0.5 * test_loss:
                        logger.info("Generalization gap diverged at epoch %d", epoch)
                        results[i]["generalization_gap_diverged"] = True
                        break

                    if np.isnan(train_loss) or np.isnan(test_loss):
                        logger.warning(
                            f"NaN loss detected at epoch {epoch}, consider lower learning rate"
                        )
                        break

                if self.scheduler is not None:
                    self.scheduler.step()

            if wandb.run is not None:
                results_to_log = {
                    "n_train_per_class": n_train_per_class,
                    "best_test_loss": best_test_loss,
                    "final_acc": results[i]["accuracy"],
                    "best_acc": best_acc,
                }
                # metrics defined in train_classifier.py
                wandb.log(results_to_log)

            if model_save_dir is not None:
                model_save_name = f"n{n_train_per_class}_c{len(self.class_list)}.pth"
                model_save_path = os.path.join(model_save_dir, model_save_name)
                logger.info(f"Saving model to {model_save_path}")
                os.makedirs(model_save_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    model_save_path,
                )

        return results
