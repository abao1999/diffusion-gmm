import json
import logging
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
    train_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    test_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    accuracy_history_all_runs: List[List[Tuple[float, float, int]]],
    save_dir: str = "figs",
    save_name: str = "loss_accuracy.png",
    title: str = "Binary Linear Classifier",
    plot_individual_runs: bool = False,
) -> None:
    fig, ax1 = plt.subplots()

    os.makedirs(save_dir, exist_ok=True)

    train_losses = [[item[1] for item in x] for x in train_loss_history_all_runs]
    test_losses = [[item[1] for item in x] for x in test_loss_history_all_runs]
    accuracies = [[item[1] for item in x] for x in accuracy_history_all_runs]
    # prop_train = np.linspace(0.1, 1.0, len(train_losses[0]))
    prop_train_schedule = [[item[0] for item in x] for x in train_loss_history_all_runs]
    n_runs = len(prop_train_schedule)
    assert (
        n_runs == len(train_losses) == len(test_losses) == len(accuracies)
    ), "Number of runs must match number of train, test, and accuracy lists"
    print("n_runs: ", n_runs)

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_test_losses = np.mean(test_losses, axis=0)
    mean_accuracies = np.mean(accuracies, axis=0)

    # Calculate standard deviation for train losses
    std_train_losses = np.std(train_losses, axis=0)
    std_test_losses = np.std(test_losses, axis=0)
    std_accuracies = np.std(accuracies, axis=0)

    ax1.set_xlabel("Proportion of Training Data")
    ax1.set_ylabel("Loss", color="tab:blue")

    if plot_individual_runs:
        for i in range(n_runs):
            ax1.plot(
                prop_train_schedule[i],
                train_losses[i],
                color="tab:blue",
                alpha=0.2,
            )
            ax1.plot(
                prop_train_schedule[i],
                test_losses[i],
                color="tab:orange",
                alpha=0.2,
            )
    # Plot average train loss
    ax1.plot(
        prop_train_schedule[0],  # Use the first schedule as they should all be the same
        mean_train_losses,
        label="Avg Train Loss",
        color="tab:blue",
        marker=".",
    )
    # Plot standard deviation envelope for train losses
    ax1.fill_between(
        prop_train_schedule[0],
        mean_train_losses - std_train_losses,
        mean_train_losses + std_train_losses,
        color="tab:blue",
        alpha=0.1,
    )
    # plot average test loss
    ax1.plot(
        prop_train_schedule[0],
        mean_test_losses,
        label="Avg Test Loss",
        color="tab:orange",
        marker=".",
    )
    # plot standard deviation envelope for test losses
    ax1.fill_between(
        prop_train_schedule[0],
        mean_test_losses - std_test_losses,
        mean_test_losses + std_test_losses,
        color="tab:orange",
        alpha=0.1,
    )

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:green")
    if plot_individual_runs:
        for i in range(n_runs):
            ax2.plot(
                prop_train_schedule[i],
                accuracies[i],
                color="tab:green",
                alpha=0.2,
            )
    # plot average accuracy
    ax2.plot(
        prop_train_schedule[0],
        mean_accuracies,
        label="Avg Accuracy",
        color="tab:green",
        marker=".",
    )
    # plot standard deviation envelope for accuracies
    ax2.fill_between(
        prop_train_schedule[0],
        mean_accuracies - std_accuracies,
        mean_accuracies + std_accuracies,
        color="tab:green",
        alpha=0.1,
    )
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.close()


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # run_names = [
    #     "gmm_imagenet64_english-springer_french-horn",
    #     "edm_imagenet64_english-springer_french-horn",
    # ]
    # for run_name in run_names:
    #     json_path = f"results/classifier/{run_name}_results.json"
    #     with open(json_path, "r") as f:
    #         results_dict = json.load(f)
    #     plot_training_history(
    #         train_loss_history_all_runs=results_dict["train_losses"],
    #         test_loss_history_all_runs=results_dict["test_losses"],
    #         accuracy_history_all_runs=results_dict["accuracies"],
    #         save_name=f"_loss_accuracy_{run_name}.png",
    #         title="Binary Linear Classifier",
    #     )

    # exit()

    logger.info(cfg.classifier)

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

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

    scheduler_cls = None
    scheduler_kwargs = {}
    if cfg.classifier.scheduler.method is not None:
        scheduler_cls = getattr(lr_scheduler, cfg.classifier.scheduler.method)
        scheduler_kwargs = dict(
            getattr(
                cfg.classifier.scheduler, f"{cfg.classifier.scheduler.method}_kwargs"
            )
        )
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
        verbose=cfg.classifier.verbose,
    )

    train_inds, test_inds = experiment.get_split_indices(
        cfg.classifier.class_list,
        cfg.classifier.max_allowed_samples_per_class,
        cfg.classifier.train_split,
    )

    prop_train_schedule = np.linspace(1.0, 0.1, cfg.classifier.n_props_train)

    results_dict = experiment.run(
        cfg.classifier.class_list,
        train_subset_inds=train_inds,
        test_subset_inds=test_inds,
        prop_train_schedule=prop_train_schedule,  # type: ignore
        n_runs=cfg.classifier.n_runs,
        reset_model_random_seed=cfg.classifier.reset_model_random_seed,
        verbose=cfg.classifier.verbose,
    )

    print(results_dict)
    save_dir = cfg.classifier.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_name = (
        f"{cfg.classifier.save_name}_results.json"
        if cfg.classifier.save_name is not None
        else "results.json"
    )
    results_file_path = os.path.join(save_dir, save_name)

    with open(results_file_path, "w") as results_file:
        json.dump(results_dict, results_file, indent=4)

    # generate plots for train and test loss, accuracy
    plot_training_history(
        results_dict["train_losses"],
        results_dict["test_losses"],
        results_dict["accuracies"],
        save_name=f"loss_accuracy_{cfg.classifier.save_name}.png",
        title="Binary Linear Classifier",
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
