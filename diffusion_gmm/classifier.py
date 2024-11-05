import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm


# Define a simple linear classifier
class BinaryLinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryLinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Output 1 for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.sigmoid(self.fc(x))


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
        multi_class_target = self.class_to_index[target]
        return data, multi_class_target

    def __len__(self):
        return len(self.subset)


@dataclass
class ClassifierExperiment:
    """
    Base class for classifier experiments
    """

    dataset: DatasetFolder

    # model parameters
    model: nn.Module
    criterion_class: nn.Module
    optimizer_class: Type[optim.Optimizer] = optim.SGD
    lr_scheduler_class: Optional[Type[lr_scheduler.LRScheduler]] = None

    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # training parameters
    num_epochs: int = 20
    lr: float = 1e-3
    train_split: float = 0.8
    batch_size: int = 64
    device: Union[torch.device, str] = "cpu"
    rseed: int = 99

    def __post_init__(self):
        # Set up device
        self.device = torch.device(self.device)
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")
        self.model.to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.lr,  # type: ignore
            **self.optimizer_kwargs,
        )
        self.criterion = self.criterion_class()
        if self.lr_scheduler_class is not None:
            self.scheduler = self.lr_scheduler_class(
                self.optimizer, **self.scheduler_kwargs
            )
        self.classes = self.dataset.classes  # type: ignore
        self.class_to_idx: Dict[str, int] = self.dataset.class_to_idx  # type: ignore
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.img_shape = tuple(self.dataset[0][0].shape)
        print("Image shape: ", self.img_shape)

        self.rng = np.random.default_rng(self.rseed)

    def train(self, dataloader: DataLoader) -> float:
        """
        Train the model for one epoch
        """
        self.model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = (
                inputs.to(self.device).float(),
                labels.to(self.device).float(),
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

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on the test set and return the average loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = (
                    inputs.to(self.device).float(),
                    labels.to(self.device).float(),
                )
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = torch.round(outputs)  # binary predictions
                correct += (preds == labels).sum().item()

            n_total = len(dataloader.dataset)  # type: ignore
        avg_loss = running_loss / n_total
        accuracy = correct / n_total
        return avg_loss, accuracy

    def _build_dataloader(
        self,
        class_list: List[str],
        num_samples: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Build the dataloaders for the training and test sets.
        Args:
            class_list: List of class names to use for classification
            num_samples: Number of samples to use for the training and test sets
        Returns:
            train_loader: DataLoader for the training set
            test_loader: DataLoader for the test set
        """
        # Convert target_class from string to integer index
        target_class_indices = [self.class_to_idx[cls] for cls in class_list]

        # For datasets like CIFAR10, use targets attribute
        if hasattr(self.dataset, "targets"):
            targets = self.dataset.targets  # type: ignore
        # For ImageFolder, reconstruct targets from samples
        elif hasattr(self.dataset, "samples"):
            targets = [class_idx for _, class_idx in self.dataset.samples]  # type: ignore
        else:
            raise AttributeError(
                "Dataset doesn't have 'targets' or 'samples' attribute"
            )

        valid_inds = [
            i
            for i, class_idx in enumerate(targets)
            if class_idx in target_class_indices
        ]

        self.rng.shuffle(valid_inds)
        n_valid_inds = len(valid_inds)

        if num_samples is not None:
            if num_samples > n_valid_inds:
                warnings.warn(
                    f"num_samples {num_samples} is greater than the number of valid samples {n_valid_inds}. Using all valid samples."
                )
            else:
                valid_inds = list(
                    self.rng.choice(valid_inds, num_samples, replace=False)
                )
                n_valid_inds = num_samples

        train_size = int(n_valid_inds * self.train_split)
        train_inds = valid_inds[:train_size]
        test_inds = valid_inds[train_size:]

        train_subset = Subset(self.dataset, train_inds)
        test_subset = Subset(self.dataset, test_inds)

        # Map class indices to user-specified multi-class classification labels
        class_to_idx = {
            self.class_to_idx[cls]: idx for idx, cls in enumerate(class_list)
        }
        train_subset = MultiClassSubset(train_subset, class_to_idx)
        test_subset = MultiClassSubset(test_subset, class_to_idx)

        target_counts = np.bincount([label for _, label in train_subset])
        print("Distribution of targets in train_subset:", target_counts)

        target_counts = np.bincount([label for _, label in test_subset])
        print("Distribution of targets in test_subset:", target_counts)

        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader

    def run(self, class_list: List[str], num_samples: Optional[int] = None) -> None:
        train_loader, test_loader = self._build_dataloader(class_list, num_samples)

        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            train_loss = self.train(train_loader)
            test_loss, accuracy = self.evaluate(test_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%"
            )
