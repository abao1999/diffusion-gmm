import json
import os
from typing import List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import default_loader

from diffusion_gmm.classifier import BinaryLinearClassifier, ClassifierExperiment


def plot_training_history(
    train_loss_history_all_runs: List[List[Tuple[float, float]]],
    test_loss_history_all_runs: List[List[Tuple[float, float]]],
    accuracy_history_all_runs: List[List[Tuple[float, float]]],
    save_dir: str = "figs",
    save_name: str = "loss_accuracy.png",
) -> None:
    fig, ax1 = plt.subplots()

    os.makedirs(save_dir, exist_ok=True)

    train_losses = [[item[1] for item in x] for x in train_loss_history_all_runs]
    test_losses = [[item[1] for item in x] for x in test_loss_history_all_runs]
    accuracies = [[item[1] for item in x] for x in accuracy_history_all_runs]
    # prop_train = np.linspace(0.1, 1.0, len(train_losses[0]))
    prop_train_schedule = [[item[0] for item in x] for x in train_loss_history_all_runs]

    ax1.set_xlabel("Proportion of Training Data")
    ax1.set_ylabel("Loss", color="tab:blue")

    ax1.plot(prop_train_schedule, train_losses, label="Train Loss", color="tab:blue")
    ax1.plot(prop_train_schedule, test_losses, label="Test Loss", color="tab:orange")

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(
        "Accuracy", color="tab:green"
    )  # we already handled the x-label with ax1
    ax2.plot(prop_train_schedule, accuracies, label="Accuracy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper right")

    plt.title("Training and Test Loss with Accuracy")
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.close()


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    torch.manual_seed(cfg.rseed)
    # If using CUDA, you should also set the seed for CUDA for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.rseed)
        torch.cuda.manual_seed_all(cfg.rseed)  # if you have multiple GPUs

    # For deterministic behavior on GPU (reproducibility), use the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check if the directory contains any .npy files
    load_npy = any(
        fname.endswith(".npy")
        for root, dirs, files in os.walk(cfg.experiment.data_dir)
        for fname in files
    )
    dataset_cls = DatasetFolder if load_npy else ImageFolder
    loader = npy_loader if load_npy else default_loader
    transform_fn = transforms.Compose([transforms.ToTensor()])

    dataset = dataset_cls(
        root=cfg.experiment.data_dir,
        loader=loader,
        is_valid_file=lambda path: path.endswith(".npy") if load_npy else True,
        transform=transform_fn if not load_npy else None,
    )

    print("all classes in dataset: ", dataset.classes)
    print(f"Dataset length: {len(dataset)}")
    img_shape = dataset[0][0].shape
    print("img_shape: ", img_shape)
    input_dim = np.prod(img_shape)
    print("input_dim: ", input_dim)

    optimizer_cls = getattr(optim, cfg.classifier.optimizer.method)
    optimizer_kwargs = dict(
        getattr(cfg.classifier.optimizer, f"{cfg.classifier.optimizer.method}_kwargs")
    )

    scheduler_cls = getattr(lr_scheduler, cfg.classifier.scheduler.method)
    if scheduler_cls is not None:
        scheduler_kwargs = dict(
            getattr(
                cfg.classifier.scheduler, f"{cfg.classifier.scheduler.method}_kwargs"
            )
        )
    else:
        scheduler_kwargs = {}
    loss_fn = getattr(nn, cfg.classifier.criterion)

    experiment = ClassifierExperiment(
        dataset=dataset,
        model_class=BinaryLinearClassifier,
        criterion_class=loss_fn,
        optimizer_class=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_class=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        lr=cfg.classifier.lr,
        num_epochs=cfg.classifier.num_epochs,
        batch_size=cfg.classifier.batch_size,
        device=cfg.classifier.device,
        rseed=cfg.rseed,
    )

    train_inds, test_inds = experiment.get_split_indices(
        cfg.classifier.class_list,
        cfg.classifier.max_allowed_samples_per_class,
        cfg.classifier.train_split,
    )

    prop_train_schedule = np.linspace(1.0, 0.1, cfg.classifier.n_runs)

    results_dict = experiment.run(
        cfg.classifier.class_list,
        train_subset_inds=train_inds,
        test_subset_inds=test_inds,
        prop_train_schedule=prop_train_schedule,  # type: ignore
        n_runs=cfg.classifier.n_runs,
        verbose=cfg.classifier.verbose,
    )

    print(results_dict)
    save_dir = os.path.join(cfg.experiment.data_dir, "classifier_results")
    os.makedirs(save_dir, exist_ok=True)
    results_file_path = os.path.join(save_dir, "results.json")

    with open(results_file_path, "w") as results_file:
        json.dump(results_dict, results_file, indent=4)

    # generate plots for train and test loss, accuracy
    plot_training_history(
        results_dict["train_losses"],
        results_dict["test_losses"],
        results_dict["accuracies"],
    )


if __name__ == "__main__":
    main()
