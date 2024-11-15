import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from diffusion_gmm.base import DataPrefetcher
from diffusion_gmm.utils import get_targets

logger = logging.getLogger(__name__)


# Define a simple linear multiclass classifier
class LinearMulticlassClassifier(nn.Module):
    """
    Simple linear multiclass classifier
    """

    def __init__(self, num_classes: int, input_dim: int):
        super(LinearMulticlassClassifier, self).__init__()
        self.fc = nn.Linear(
            input_dim, num_classes
        )  # Output num_classes for multiclass classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.softmax(self.fc(x))


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
        self.softmax = nn.Softmax(dim=1)  # Output layer for multiclass classification

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))  # First layer with ReLU activation
        return self.softmax(self.fc2(x))  # Second layer with Softmax activation


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
        self.fc = nn.Linear(input_dim, 1)  # Output 1 for binary classification
        self.sigmoid = nn.Sigmoid()
        self.output_logit = output_logit

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        if self.output_logit:
            return self.fc(x)
        else:
            return self.sigmoid(self.fc(x))


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
        self.sigmoid = nn.Sigmoid()
        self.output_logit = output_logit

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))  # First layer with ReLU activation
        if self.output_logit:
            return self.fc2(x)  # Return logits
        else:
            return self.sigmoid(self.fc2(x))  # Return probabilities


class MultiClassSubset(Dataset):
    """
    Wrap a subset of a dataset to apply a mapping of targets (class labels) to
    user-specified multi-class classification labels
    """

    def __init__(self, subset, class_to_index):
        self.subset = subset
        self.class_to_index = class_to_index

    def __getitem__(self, index):
        data, target = self.subset[index]
        if target not in self.class_to_index:
            raise ValueError(f"Target {target} not valid for multi-class subset")
        label = self.class_to_index[target]
        return data, label

    def __len__(self):
        return len(self.subset)


@dataclass
class ClassifierExperiment:
    """
    Base class for classifier experiments
    """

    train_subset: Subset
    test_subset: Subset

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
    rseed: int = 99
    verbose: bool = False

    def __post_init__(self):
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device(self.device)
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")
        print("device: ", self.device)
        print("current device: ", torch.cuda.current_device())

        self.criterion = self.criterion_class()

        sample, _ = self.train_subset.dataset[0]
        if isinstance(sample, torch.Tensor):
            self.img_shape = sample.shape
        else:
            raise ValueError("Sample is not a tensor")

        self.rng = np.random.default_rng(self.rseed)

        if self.model_class in [LinearBinaryClassifier, TwoLayerBinaryClassifier]:
            # BCELoss and MSELoss require labels to be floats (same type as output)
            # Furthremore, predictions are made by rounding scalar, rather than taking torch.max
            self.use_binary_classifier = True
        else:
            self.use_binary_classifier = False

    def make_model_and_optimizer(self):
        self.model = self.model_class(
            num_classes=len(self.class_list),
            input_dim=np.prod(self.img_shape),
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
            inputs, labels = (
                inputs.to(self.device, non_blocking=True).float(),
                labels.to(self.device, non_blocking=True).long()
                if not self.use_binary_classifier
                else labels.to(self.device, non_blocking=True).float(),
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
                inputs, labels = (
                    inputs.to(self.device, non_blocking=True).float(),
                    labels.to(self.device, non_blocking=True).long()
                    if not self.use_binary_classifier
                    else labels.to(self.device, non_blocking=True).float(),
                )
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                if self.use_binary_classifier:
                    preds = torch.round(outputs)  # binary predictions
                else:
                    _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()

            n_total = len(dataloader.dataset)  # type: ignore
        avg_loss = running_loss / n_total
        accuracy = correct / n_total
        return avg_loss, accuracy

    def _build_dataloader(
        self,
        subset: Subset,
        prop_samples_to_use: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
        make_prefetcher: bool = True,
    ) -> DataLoader | DataPrefetcher:
        dataset = subset.dataset
        class_to_idx = dataset.class_to_idx  # type: ignore
        # Map class indices to user-specified multi-class classification labels
        class_idx_to_label = {
            class_to_idx[cls]: idx for idx, cls in enumerate(self.class_list)
        }
        if prop_samples_to_use < 1.0:
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
            selected_indices = []
            rng = rng if rng is not None else self.rng
            for class_idx, indices in class_to_indices.items():
                if shuffle:
                    rng.shuffle(indices)
                subset_size = int(len(indices) * prop_samples_to_use)
                selected_indices.extend(indices[:subset_size])

            selected_subset = MultiClassSubset(
                Subset(dataset, selected_indices), class_idx_to_label
            )
        else:
            selected_subset = MultiClassSubset(subset, class_idx_to_label)

        if self.verbose:
            target_counts = np.bincount([label for _, label in selected_subset])
            logger.info("Built dataloader with %d samples...", len(selected_subset))
            logger.info("Distribution of targets in subset: %s", target_counts)

        dataloader = DataLoader(
            selected_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=str(self.device),
            persistent_workers=True,
        )
        if make_prefetcher:
            dataloader = DataPrefetcher(dataloader, self.device)

        return dataloader

    def run(
        self,
        prop_train_schedule: Sequence[float],
        n_runs: int = 1,
        reset_model_random_seed: bool = False,
        num_epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> Dict[str, List[List[Tuple[float, float, int]]]]:
        """
        Run the classifier experiment
        Args:
            prop_train_schedule: Schedule for proportion of samples in training split to use for training
            train_indices: Indices to use for training
            test_subset_inds: Indices to use for testing
        """
        if verbose:
            logger.info(
                "Running classifier experiment with %s classes...", self.class_list
            )
            logger.info("prop_train_schedule: %s", prop_train_schedule)
            logger.info(
                "Building test dataloader with %d samples...",
                len(self.test_subset.indices),
            )

        n_props_train = len(prop_train_schedule)

        test_loader = self._build_dataloader(
            self.test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            make_prefetcher=True,
        )

        rng_stream = self.rng.spawn(n_runs)

        results_all_runs = []
        for run_idx in tqdm(range(n_runs), desc="Running classifier experiment"):
            # fix random seed for each run
            rng = rng_stream[run_idx]

            if reset_model_random_seed:
                # set torch random seed for this run
                rseed = self.rseed + run_idx
                print(f"Setting torch random seed to {rseed} for run {run_idx}")
                torch.manual_seed(rseed)
                # If using CUDA, you should also set the seed for CUDA for full reproducibility
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(rseed)
                    torch.cuda.manual_seed_all(rseed)  # if you have multiple GPUs

            # train on different proportions of the training set, defined by prop_train_schedule
            results: List[Dict[str, Any]] = [{}] * n_props_train
            for prop_idx in tqdm(
                range(n_props_train),
                desc="Training on different proportions of the training set",
            ):
                # reset the model and optimizer and scheduler to the initial state
                self.make_model_and_optimizer()

                prop_train = prop_train_schedule[prop_idx]

                # build train dataloader
                train_loader = self._build_dataloader(
                    self.train_subset,
                    prop_samples_to_use=prop_train,
                    batch_size=batch_size,
                    rng=rng,
                    shuffle=True,
                    num_workers=4,
                    make_prefetcher=True,
                )

                if verbose:
                    logger.info(
                        f"run {run_idx}, prop_train {prop_train_schedule[prop_idx]}"
                    )
                    logger.info(
                        f"Building train dataloader using {len(train_loader.dataset)} samples from the training split..."  # type: ignore
                    )

                best_test_loss = float("inf")
                for epoch in tqdm(range(num_epochs), desc="Training"):
                    train_loss = self.train(train_loader)
                    test_loss, accuracy = self.evaluate(test_loader)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        results[prop_idx] = {
                            "num_train_samples": len(train_loader.dataset),  # type: ignore
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "accuracy": accuracy,
                            "epoch": epoch,
                        }
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if verbose and (epoch + 1) % 50 == 0:
                        logger.info(
                            "Epoch [%d/%d], Train Loss: %f, Test Loss: %f, Accuracy: %f%%",
                            epoch + 1,
                            num_epochs,
                            train_loss,
                            test_loss,
                            accuracy * 100,
                        )
                        logger.info(
                            "Learning rate: %f", self.optimizer.param_groups[0]["lr"]
                        )

                logger.info(
                    "Final results for run %d, prop %f: %s",
                    run_idx,
                    prop_train,
                    results[prop_idx],
                )
            results_all_runs.append(results)

        results_dict = {
            "num_train_samples": [
                [result[prop_idx]["num_train_samples"] for result in results_all_runs]
                for prop_idx in range(n_props_train)
            ],
            "train_losses": [
                [result[prop_idx]["train_loss"] for result in results_all_runs]
                for prop_idx in range(n_props_train)
            ],
            "test_losses": [
                [result[prop_idx]["test_loss"] for result in results_all_runs]
                for prop_idx in range(n_props_train)
            ],
            "accuracies": [
                [result[prop_idx]["accuracy"] for result in results_all_runs]
                for prop_idx in range(n_props_train)
            ],
            "epochs": [
                [result[prop_idx]["epoch"] for result in results_all_runs]
                for prop_idx in range(n_props_train)
            ],
        }
        if verbose:
            logger.info("Results:\n%s", json.dumps(results_dict, indent=4))
        return results_dict
