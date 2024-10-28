import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from diffusion_gmm.base import ImageExperiment


# Define a simple linear classifier
class BinaryLinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryLinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Output 1 for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.sigmoid(self.fc(x))


class BinarySubset(Dataset):
    def __init__(self, subset, class_to_binary):
        self.subset = subset
        self.class_to_binary = class_to_binary

    def __getitem__(self, index):
        data, target = self.subset[index]
        if target not in self.class_to_binary:
            raise ValueError(f"Target {target} not valid for binary subset")
        binary_target = self.class_to_binary[target]
        return data, binary_target

    def __len__(self):
        return len(self.subset)


@dataclass
class BinaryClassifierExperiment(ImageExperiment):
    num_epochs: int = 20
    lr: float = 1e-3
    device: Union[torch.device, str] = "cpu"
    split_ratio: float = 0.8
    plot_history: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Set up device
        self.device = torch.device(self.device)
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")

        self.input_dim = np.prod(self.img_shape)
        # TODO: right now this is a linear classifier hard-coded
        self.model = BinaryLinearClassifier(self.input_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        # self.scheduler = lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=10, eta_min=0
        # )

    def _build_dataloader(
        self,
        class_list: List[str],
        num_samples: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        # Convert target_class from string to integer index
        target_class_indices = [self.class_to_idx[cls] for cls in class_list]

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

        train_size = int(n_valid_inds * self.split_ratio)
        train_inds = valid_inds[:train_size]
        test_inds = valid_inds[train_size:]

        train_subset = Subset(self.data, train_inds)
        test_subset = Subset(self.data, test_inds)

        # Map class indices to 0 or 1 for binary classification
        class_to_binary = {
            self.class_to_idx[class_list[0]]: 0,
            self.class_to_idx[class_list[1]]: 1,
        }

        # Wrap subsets with BinarySubset to apply the mapping of targets (class labels) to 0 or 1
        train_subset = BinarySubset(train_subset, class_to_binary)
        test_subset = BinarySubset(test_subset, class_to_binary)

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
        if len(class_list) != 2:
            raise ValueError(
                "for binary classification, class_list must contain exactly 2 classes"
            )
        train_loader, test_loader = self._build_dataloader(class_list, num_samples)

        # Train and evaluate the model
        train_loss_history = []
        test_loss_history = []
        accuracy_history = []
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            train_loss = self.train(train_loader)
            test_loss, accuracy = self.evaluate(test_loader)
            self.scheduler.step()

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            accuracy_history.append(accuracy)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%"
            )

        if self.plot_history:
            fig, ax1 = plt.subplots()

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color="tab:blue")
            ax1.plot(train_loss_history, label="Train Loss", color="tab:blue")
            ax1.plot(test_loss_history, label="Test Loss", color="tab:orange")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel(
                "Accuracy", color="tab:green"
            )  # we already handled the x-label with ax1
            ax2.plot(accuracy_history, label="Accuracy", color="tab:green")
            ax2.tick_params(axis="y", labelcolor="tab:green")
            ax2.legend(loc="upper right")

            plt.title("Training and Test Loss with Accuracy")
            plt.savefig("figs/loss_accuracy.png", dpi=300)
            plt.close()

    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device).float()
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
        self.model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = torch.round(outputs)  # binary predictions
                correct += (preds == labels).sum().item()

            n_total = len(dataloader.dataset)  # type: ignore
        avg_loss = running_loss / n_total
        accuracy = correct / n_total
        return avg_loss, accuracy
