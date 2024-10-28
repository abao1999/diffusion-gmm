from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from diffusion_gmm.base import ImageExperiment


# Define a simple linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Output 1 for binary classification

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)


class LinearClassifierExperiment(ImageExperiment):
    class_list: List[str]
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
        self.model = LinearClassifier(self.input_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=0
        )

    def run(self, num_samples: int) -> None:
        train_loader = self._build_dataloader(
            num_samples, target_class=self.class_list[0]
        )
        test_loader = self._build_dataloader(
            num_samples, target_class=self.class_list[0]
        )

        # Train and evaluate the model
        train_loss_history = []
        test_loss_history = []
        accuracy_history = []
        for epoch in range(self.num_epochs):
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
            plt.plot(train_loss_history, label="Train Loss")
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig("figs/train_loss.png", dpi=300)
            plt.close()

            plt.plot(test_loss_history, label="Test Loss")
            plt.title("Test Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig("figs/test_loss.png", dpi=300)
            plt.close()

            plt.plot(accuracy_history, label="Accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.savefig("figs/accuracy.png", dpi=300)
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
                preds = torch.round(
                    torch.sigmoid(outputs)
                )  # Convert logits to binary predictions
                correct += (preds == labels).sum().item()

            n_total = len(dataloader.dataset)  # type: ignore
        avg_loss = running_loss / n_total
        accuracy = correct / n_total
        return avg_loss, accuracy
