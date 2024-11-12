import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


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
        label = self.class_to_index[target]
        return data, label

    def __len__(self):
        return len(self.subset)


@dataclass
class ClassifierExperiment:
    """
    Base class for classifier experiments
    """

    # train_dataset: DatasetFolder
    # test_dataset: DatasetFolder
    dataset: DatasetFolder

    # model parameters
    model_class: Type[nn.Module]
    criterion_class: Type[nn.Module]
    optimizer_class: Type[optim.Optimizer] = optim.SGD
    lr_scheduler_class: Optional[Type[lr_scheduler.LRScheduler]] = None

    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # training parameters
    num_epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64
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

        self.classes = self.dataset.classes  # type: ignore
        self.class_to_idx: Dict[str, int] = self.dataset.class_to_idx  # type: ignore
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.img_shape = tuple(self.dataset[0][0].shape)
        print("Image shape: ", self.img_shape)

        self.rng = np.random.default_rng(self.rseed)
        self.targets = self._get_targets()

    def make_model_and_optimizer(self):
        self.model = BinaryLinearClassifier(input_dim=np.prod(self.img_shape)).to(
            self.device
        )
        # self.model.to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.lr,  # type: ignore
            **self.optimizer_kwargs,
        )
        if self.lr_scheduler_class is not None:
            self.scheduler = self.lr_scheduler_class(
                self.optimizer, **self.scheduler_kwargs
            )
        else:
            self.scheduler = None

    def _get_targets(self) -> List[int]:
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
        return targets

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

    @torch.no_grad()
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

    def get_split_indices(
        self,
        class_list: List[str],
        max_allowed_samples_per_class: Optional[int] = None,
        train_split: float = 0.8,
    ) -> Tuple[List[int], List[int]]:
        """
        Get the indices for the training and test subsets
        """
        # Group indices by class
        class_to_indices = {self.class_to_idx[cls]: [] for cls in class_list}
        for i, class_idx in enumerate(self.targets):
            # filter out only the indices for the targets that are in class_list
            if class_idx in class_to_indices:
                class_to_indices[class_idx].append(i)

        for indices in class_to_indices.values():
            self.rng.shuffle(indices)

        # Determine the minimum number of samples in the class with the fewest samples
        n_samples_per_class = min(len(indices) for indices in class_to_indices.values())

        # If max_allowed_samples_per_class is specified, limit the samples per class
        if max_allowed_samples_per_class is not None:
            n_samples_per_class = min(
                n_samples_per_class,
                max_allowed_samples_per_class,
            )

        if self.verbose:
            logger.info("number of samples in smallest class: %d", n_samples_per_class)
            logger.info("number of samples per class to use: %d", n_samples_per_class)

        # Sample equal number of indices from each class
        train_size_per_class = int(n_samples_per_class * train_split)
        balanced_train_inds = []
        balanced_test_inds = []
        for indices in class_to_indices.values():
            balanced_train_inds.extend(indices[:train_size_per_class])
            balanced_test_inds.extend(indices[train_size_per_class:n_samples_per_class])
        return balanced_train_inds, balanced_test_inds

    def _build_dataloader(
        self,
        class_list: List[str],
        dataset_indices: List[int],
        prop_samples_to_use: float = 1.0,
        use_augmentations: bool = False,
        rng: Optional[np.random.Generator] = None,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        Build the dataloaders for the training and test sets.
        Args:
            class_list: List of class names to use for classification
            dataset_indices: Indices to use from the dataset
            prop_samples_to_use: Proportion of samples in dataset to use for training
            rng: Random number generator to use for shuffling. Defaults to self.rng
            shuffle: Whether to shuffle the dataset indices
        Returns:
            dataloader: DataLoader for the training set
        """
        # Define the data augmentation pipeline
        if use_augmentations:
            transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomResizedCrop(self.img_shape[1:], scale=(0.8, 1.0)),
                    # transforms.ColorJitter(
                    #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    # ),
                    transforms.ToTensor(),  # Ensure the data is converted to a tensor
                ]
            )
            self.dataset.transform = transform

        # Group indices by class
        class_to_indices = {self.class_to_idx[cls]: [] for cls in class_list}
        for idx in dataset_indices:
            class_idx = self.targets[idx]
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

        # Map class indices to user-specified multi-class classification labels
        class_idx_to_label = {
            self.class_to_idx[cls]: idx for idx, cls in enumerate(class_list)
        }

        selected_subset = MultiClassSubset(
            Subset(self.dataset, selected_indices), class_idx_to_label
        )

        target_counts = np.bincount([label for _, label in selected_subset])

        if self.verbose:
            logger.info("Built dataloader with %d samples...", len(selected_indices))
            logger.info("Distribution of targets in subset: %s", target_counts)

        dataloader = DataLoader(
            selected_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        return dataloader

    def run(
        self,
        class_list: List[str],
        train_subset_inds: List[int],
        test_subset_inds: List[int],
        prop_train_schedule: Sequence[float],
        n_runs: int = 1,
        reset_model_random_seed: bool = False,
        use_augmentations: bool = False,
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
            logger.info("Running classifier experiment with %s classes...", class_list)
            logger.info("prop_train_schedule: %s", prop_train_schedule)
            logger.info(
                "Building test dataloader with %d samples...", len(test_subset_inds)
            )

        n_props_train = len(prop_train_schedule)
        test_loader = self._build_dataloader(
            class_list, test_subset_inds, prop_samples_to_use=1.0
        )

        rng_stream = self.rng.spawn(n_runs)

        final_train_losses_all_runs = []
        final_test_losses_all_runs = []
        final_accuracies_all_runs = []
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
            final_train_losses = [(-1.0, -1.0, -1)] * n_props_train
            final_test_losses = [(-1.0, -1.0, -1)] * n_props_train
            final_accuracies = [(-1.0, -1.0, -1)] * n_props_train
            for prop_idx in tqdm(
                range(n_props_train),
                desc="Training on different proportions of the training set",
            ):
                # reset the model and optimizer and scheduler to the initial state
                self.make_model_and_optimizer()

                prop_train = prop_train_schedule[prop_idx]

                if verbose:
                    logger.info(
                        f"run {run_idx}, prop_train {prop_train_schedule[prop_idx]}"
                    )
                    logger.info(
                        f"Building train dataloader using {prop_train} * {len(train_subset_inds)} samples from the training split...",
                    )

                train_loader = self._build_dataloader(
                    class_list,
                    train_subset_inds,
                    prop_samples_to_use=prop_train,
                    use_augmentations=use_augmentations,
                    rng=rng,
                )

                best_test_loss = float("inf")
                for epoch in tqdm(range(self.num_epochs), desc="Training"):
                    train_loss = self.train(train_loader)
                    test_loss, accuracy = self.evaluate(test_loader)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        final_train_losses[prop_idx] = (prop_train, train_loss, epoch)
                        final_test_losses[prop_idx] = (prop_train, test_loss, epoch)
                        final_accuracies[prop_idx] = (prop_train, accuracy, epoch)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if verbose and (epoch + 1) % 10 == 0:
                        logger.info(
                            "Epoch [%d/%d], Train Loss: %f, Test Loss: %f, Accuracy: %f%%",
                            epoch + 1,
                            self.num_epochs,
                            train_loss,
                            test_loss,
                            accuracy * 100,
                        )
                        logger.info(
                            "Learning rate: %f", self.optimizer.param_groups[0]["lr"]
                        )

                logger.info(f"run {run_idx} prop {prop_train} finished")
                logger.info(
                    f"train loss for prop {prop_train}: {final_train_losses[prop_idx]}"
                )
                logger.info(
                    f"test loss for prop {prop_train}: {final_test_losses[prop_idx]}"
                )
                logger.info(
                    f"accuracy for prop {prop_train}: {final_accuracies[prop_idx]}"
                )
            final_train_losses_all_runs.append(final_train_losses)
            final_test_losses_all_runs.append(final_test_losses)
            final_accuracies_all_runs.append(final_accuracies)

        results_dict = {
            "train_losses": final_train_losses_all_runs,
            "test_losses": final_test_losses_all_runs,
            "accuracies": final_accuracies_all_runs,
        }
        if verbose:
            logger.info("Results:\n%s", json.dumps(results_dict, indent=4))
        return results_dict
