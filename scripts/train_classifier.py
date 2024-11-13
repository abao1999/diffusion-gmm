import copy
import json
import logging
import os
from typing import List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import default_loader

from diffusion_gmm import classifier
from diffusion_gmm.classifier import ClassifierExperiment
from diffusion_gmm.utils import (
    get_targets,
    set_seed,
)


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


def split_dataset_balanced(
    dataset: DatasetFolder,
    class_list: List[str],
    max_allowed_samples_per_class: Optional[int] = None,
    train_split: float = 0.8,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[Subset, Subset]:
    """
    Get the indices for the training and test subsets
    """
    class_to_idx = dataset.class_to_idx
    targets = get_targets(dataset)
    # Group indices by class
    class_to_indices = {class_to_idx[cls]: [] for cls in class_list}
    for i, class_idx in enumerate(targets):
        # filter out only the indices for the targets that are in class_list
        if class_idx in class_to_indices:
            class_to_indices[class_idx].append(i)

    if rng is not None:
        for indices in class_to_indices.values():
            rng.shuffle(indices)

    # Determine the minimum number of samples in the class with the fewest samples
    n_samples_per_class = min(len(indices) for indices in class_to_indices.values())

    # If max_allowed_samples_per_class is specified, limit the samples per class
    if max_allowed_samples_per_class is not None:
        n_samples_per_class = min(
            n_samples_per_class,
            max_allowed_samples_per_class,
        )

    if verbose:
        logger.info("number of samples in smallest class: %d", n_samples_per_class)
        logger.info("number of samples per class to use: %d", n_samples_per_class)

    # Sample equal number of indices from each class
    train_size_per_class = int(n_samples_per_class * train_split)
    balanced_train_inds = []
    balanced_test_inds = []
    for indices in class_to_indices.values():
        balanced_train_inds.extend(indices[:train_size_per_class])
        balanced_test_inds.extend(indices[train_size_per_class:n_samples_per_class])

    dataset_copy = copy.deepcopy(dataset)
    # return balanced_train_inds, balanced_test_inds
    train_subset = Subset(dataset, balanced_train_inds)
    test_subset = Subset(dataset_copy, balanced_test_inds)
    return train_subset, test_subset


def setup_dataset(
    data_dir: str,
) -> Tuple[DatasetFolder, bool]:
    # Check if the directory contains any .npy files
    is_npy_dataset = any(
        fname.endswith(".npy")
        for root, dirs, files in os.walk(data_dir)
        for fname in files
    )
    dataset_cls = DatasetFolder if is_npy_dataset else ImageFolder
    loader = npy_loader if is_npy_dataset else default_loader
    dataset = dataset_cls(
        root=data_dir,
        loader=loader,
        is_valid_file=lambda path: path.endswith(".npy") if is_npy_dataset else True,
    )
    return dataset, is_npy_dataset


def make_train_test_subsets(
    class_list: List[str],
    train_data_dir: str,
    test_data_dir: Optional[str],
    max_allowed_samples_per_class: Optional[int],
    train_split: float = 0.8,
    use_augmentations: bool = False,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[Subset, Subset]:
    train_dataset, is_npy_dataset = setup_dataset(train_data_dir)
    train_subset, test_subset = split_dataset_balanced(
        train_dataset,
        class_list=class_list,
        max_allowed_samples_per_class=max_allowed_samples_per_class,
        train_split=train_split,
        rng=rng,
        verbose=verbose,
    )
    validate_subsets(train_subset, test_subset)

    if test_data_dir is not None:
        test_dataset, _ = setup_dataset(test_data_dir)
        _, test_subset = split_dataset_balanced(
            test_dataset,
            class_list=class_list,
            max_allowed_samples_per_class=max_allowed_samples_per_class,
            train_split=train_split,
            rng=rng,
            verbose=verbose,
        )

    if is_npy_dataset:
        return train_subset, test_subset

    # Define the basic transformation for the test dataset
    test_transform = transforms.ToTensor()
    test_subset.dataset.transform = test_transform  # type: ignore

    # set up augmentations (for train subset only)
    sample, _ = train_subset.dataset[0]
    if isinstance(sample, torch.Tensor):
        # If the sample is already a tensor (e.g., from npy files)
        img_shape = sample.shape
    else:
        # If the sample is a PIL image, convert it to a tensor first
        sample_tensor = transforms.ToTensor()(sample)
        img_shape = sample_tensor.shape

    # Define the data augmentation pipeline for the train dataset
    if use_augmentations:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(img_shape[1:], scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),  # Ensure the data is converted to a tensor
            ]
        )
    else:
        train_transform = transforms.ToTensor()

    train_subset.dataset.transform = train_transform  # type: ignore
    return train_subset, test_subset


def validate_subsets(train_subset: Subset, test_subset: Subset):
    """
    Verify that there are no shared indices between the train and test subsets
    """
    # train_inds = [i for i, _ in train_subset]
    # test_inds = [i for i, _ in test_subset]
    train_inds = train_subset.indices
    test_inds = test_subset.indices
    if set(train_inds) & set(test_inds):
        raise ValueError("Overlap detected between train and test indices.")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # set torch, cuda, and cudnn seeds
    set_seed(cfg.rseed)
    # set numpy rng
    rng = np.random.default_rng(cfg.rseed)

    logger.info(cfg.classifier)

    train_subset, test_subset = make_train_test_subsets(
        class_list=cfg.classifier.class_list,
        train_data_dir=cfg.classifier.train_data_dir,
        test_data_dir=cfg.classifier.test_data_dir,
        max_allowed_samples_per_class=cfg.classifier.max_allowed_samples_per_class,
        train_split=cfg.classifier.train_split,
        use_augmentations=cfg.classifier.use_augmentations,
        rng=rng,
        verbose=cfg.classifier.verbose,
    )

    model_cls = getattr(classifier, cfg.classifier.model.name)
    model_kwargs = getattr(cfg.classifier.model, f"{cfg.classifier.model.name}_kwargs")

    optimizer_cls = getattr(optim, cfg.classifier.optimizer.name)
    optimizer_kwargs = dict(
        getattr(cfg.classifier.optimizer, f"{cfg.classifier.optimizer.name}_kwargs")
    )

    scheduler_cls = None
    scheduler_kwargs = {}
    if cfg.classifier.scheduler.name is not None:
        scheduler_cls = getattr(lr_scheduler, cfg.classifier.scheduler.name)
        scheduler_kwargs = dict(
            getattr(cfg.classifier.scheduler, f"{cfg.classifier.scheduler.name}_kwargs")
        )

    loss_fn = getattr(nn, cfg.classifier.criterion)

    experiment = ClassifierExperiment(
        class_list=cfg.classifier.class_list,
        train_subset=train_subset,
        test_subset=test_subset,
        model_class=model_cls,
        model_kwargs=model_kwargs,
        criterion_class=loss_fn,
        optimizer_class=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_class=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        lr=cfg.classifier.lr,
        device=cfg.classifier.device,
        rseed=cfg.rseed,
        verbose=cfg.classifier.verbose,
    )

    prop_train_schedule = np.linspace(1.0, 0.1, cfg.classifier.n_props_train)

    results_dict = experiment.run(
        prop_train_schedule=prop_train_schedule,  # type: ignore
        n_runs=cfg.classifier.n_runs,
        reset_model_random_seed=cfg.classifier.reset_model_random_seed,
        num_epochs=cfg.classifier.num_epochs,
        batch_size=cfg.classifier.batch_size,
        verbose=cfg.classifier.verbose,
    )

    print(results_dict)
    save_dir = os.path.join(cfg.classifier.save_dir, cfg.classifier.model.name)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{cfg.classifier.save_name}_results.json"
    results_file_path = os.path.join(save_dir, save_name)

    with open(results_file_path, "w") as results_file:
        json.dump(results_dict, results_file, indent=4)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
