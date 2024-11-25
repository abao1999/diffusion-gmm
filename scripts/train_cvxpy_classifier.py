import logging
import os

import cvxpy as cp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")


# Define a simple linear binary classifier
class LinearBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearBinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)


def sgd(data, y, d):
    w = cp.Variable(d)
    b = cp.Variable()
    cost = cp.sum_squares(w)
    constraints = [data @ w + b == y]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(cp.CLARABEL)
    return w, b


def train_cvxpy_classifier(
    train_data_dir, test_data_dir, num_epochs=50, batch_size=1024, lr=3e-4
):
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Load training dataset
    train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = 64 * 64 * 3  # Assuming 64x64 RGB images
    model = LinearBinaryClassifier(input_dim=input_dim)

    best_accuracy = 0.0
    # best_model_state = None

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        for inputs, labels in train_dataloader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs
            labels = labels.float().unsqueeze(1)  # Convert labels to float

            w, b = sgd(inputs, labels, input_dim)

            # # Define CVXPY variables and problem
            # w = cp.Variable((input_dim, 1))
            # b = cp.Variable()
            # predictions = inputs.cpu().numpy() @ w + b
            # loss = cp.sum_squares(predictions - labels.cpu().numpy()) / (2 * batch_size)

            # # Solve the optimization problem
            # problem = cp.Problem(cp.Minimize(loss))
            # problem.solve(solver=cp.CLARABEL)

            # Check if the solution is valid
            if w.value is not None and b.value is not None:
                with torch.no_grad():
                    model.fc.weight.copy_(torch.tensor(w.value.T))
                    model.fc.bias.copy_(torch.tensor(b.value, dtype=torch.float32))
            else:
                logger.warning("Optimization problem did not solve correctly.")
        # Evaluation loop within training loop
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

                # Calculate test loss using MSELoss
                test_loss += nn.functional.mse_loss(
                    outputs, labels.float().unsqueeze(1)
                ).item()

        test_loss /= len(test_dataloader)
        accuracy = 100 * correct / total
        logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Check for best model
        if test_loss < best_accuracy:
            best_accuracy = test_loss
            # best_model_state = model.state_dict()

    # # Load the best model state
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    #     logger.info(f"Best model loaded with test loss: {best_accuracy:.4f}")

    # # Save model
    # save_dir = os.path.join(train_data_dir, "model")
    # os.makedirs(save_dir, exist_ok=True)
    # torch.save(
    #     model.state_dict(), os.path.join(save_dir, "linear_binary_classifier.pth")
    # )


if __name__ == "__main__":
    train_data_dir = os.path.join(DATA_DIR, "edm_imagenet64_train")
    test_data_dir = os.path.join(DATA_DIR, "edm_imagenet64_test")
    train_cvxpy_classifier(train_data_dir, test_data_dir)
