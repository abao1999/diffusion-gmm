import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    train_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    test_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    accuracy_history_all_runs: List[List[Tuple[float, float, int]]],
    save_dir: str = "plots",
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


def plot_quantity(
    results: Dict[str, List[float]],
    num_samples_schedule: Dict[str, List[int]],
    save_dir: str = "plots",
    save_name: str = "loss_accuracy.png",
    title: str = "Binary Linear Classifier",
    label: str = "Accuracy",
) -> None:
    fig, ax1 = plt.subplots()

    os.makedirs(save_dir, exist_ok=True)

    for run_name, quantity_history in results.items():
        n_runs = len(quantity_history)
        assert n_runs == len(
            quantity_history
        ), "Number of runs must match number of quantities"
        print("n_runs: ", n_runs)

        mean_quantities = np.mean(quantity_history, axis=1)
        num_samples = np.mean(num_samples_schedule[run_name], axis=1)

        # Calculate standard deviation for quantities
        std_quantities = np.std(quantity_history, axis=1)

        ax1.plot(
            num_samples,
            mean_quantities,
            label=run_name,
            marker=".",
        )
        ax1.fill_between(
            num_samples,
            mean_quantities - std_quantities,
            mean_quantities + std_quantities,
            alpha=0.1,
        )

    ax1.set_xlabel("Number of Training Samples")
    ax1.set_ylabel(label)
    ax1.legend()
    plt.title(title)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.close()


def plot_results(run_json_paths: Dict[str, str], title: str) -> None:
    accuracies = {}
    test_losses = {}
    num_samples_schedule = {}
    for run_name, json_path in run_json_paths.items():
        with open(json_path, "r") as f:
            results_dict = json.load(f)
        accuracies[run_name] = results_dict["accuracies"]
        test_losses[run_name] = results_dict["test_losses"]
        num_samples_schedule[run_name] = results_dict["num_train_samples"]
        # plot_training_history(
        #     train_loss_history_all_runs=results_dict["train_losses"],
        #     test_loss_history_all_runs=results_dict["test_losses"],
        #     accuracy_history_all_runs=results_dict["accuracies"],
        #     save_name=f"loss_accuracy_{run_name}.png",
        #     title=title,
        #     save_dir="plots",
        # )

    plot_quantity(
        results=accuracies,
        num_samples_schedule=num_samples_schedule,
        save_name="accuracies.png",
        title=title,
        label="Accuracy",
        save_dir="plots",
    )

    plot_quantity(
        results=test_losses,
        num_samples_schedule=num_samples_schedule,
        save_name="test_losses.png",
        title=title,
        label="Test Loss",
        save_dir="plots",
    )


if __name__ == "__main__":
    # model_name = "TwoLayerMulticlassClassifier"
    # run_names_dict = {
    #     "real": "imagenette64_english-springer_french-horn_church",
    #     "diffusion": "edm_imagenet64_english-springer_french-horn_church",
    #     "gmm": "gmm_imagenet64_english-springer_french-horn_church",
    # }

    model_name = "LinearBinaryClassifier"
    # run_names_dict = {
    #     "real": "imagenette64_english-springer_church",
    #     "diffusion": "edm_imagenet64_english-springer_church",
    #     "gmm": "gmm_imagenet64_english-springer_church",
    # }
    run_names_dict = {
        # "real": "imagenette64_english-springer_french-horn",
        "diffusion": "edm_imagenet64_big_bs64_MSELoss_english-springer_french-horn_run1",
        "gmm": "gmm_edm_imagenet64_big_bs64_MSELoss_english-springer_french-horn_run1",
    }

    json_dir = f"results/classifier/{model_name}"
    run_json_paths = {
        name: os.path.join(json_dir, f"{run_names_dict[name]}_results.json")
        for name in run_names_dict
    }
    # plot_results(run_json_paths, title="Two Layer Classifier with 3 Classes")
    plot_results(run_json_paths, title="Linear Binary Classifier")
