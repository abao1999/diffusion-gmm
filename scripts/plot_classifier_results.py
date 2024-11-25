import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Define the style
plt.rcParams.update(
    {
        # Font and text size
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        # Axes style
        "axes.linewidth": 0.75,
        "axes.grid": False,
        "grid.color": "gray",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.prop_cycle": cycler(
            "color",
            [
                "#377eb8",
                "#ff7f0e",
                "#4daf4a",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ],
        ),
        # Ticks
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.major.size": 4,
        "ytick.minor.size": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Figure layout
        "figure.figsize": (3.25, 2.5),  # Inches (adjust based on your needs)
        "figure.dpi": 300,
        "figure.autolayout": True,
        # Legend
        "legend.loc": "upper right",
        "legend.frameon": False,
        # Savefig options
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.transparent": True,
    }
)


def plot_training_history(
    train_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    test_loss_history_all_runs: List[List[Tuple[float, float, int]]],
    accuracy_history_all_runs: List[List[Tuple[float, float, int]]],
    save_dir: str = "plots",
    save_name: str = "loss_accuracy",
    title: str = "Binary Linear Classifier",
    plot_individual_runs: bool = False,
) -> None:
    fig, ax1 = plt.subplots(figsize=(4, 3))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")
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

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax1.set_xlabel("Proportion of Training Data")
    ax1.set_ylabel("Loss", color=colors[0])

    if plot_individual_runs:
        for i in range(n_runs):
            ax1.plot(
                prop_train_schedule[i],
                train_losses[i],
                color=colors[0],
                alpha=0.2,
            )
            ax1.plot(
                prop_train_schedule[i],
                test_losses[i],
                color=colors[1],
                alpha=0.2,
            )
    # Plot average train loss
    ax1.plot(
        prop_train_schedule[0],  # Use the first schedule as they should all be the same
        mean_train_losses,
        label="Avg Train Loss",
        color=colors[0],
        marker=".",
    )
    # Plot standard deviation envelope for train losses
    ax1.fill_between(
        prop_train_schedule[0],
        mean_train_losses - std_train_losses,
        mean_train_losses + std_train_losses,
        color=colors[0],
        alpha=0.1,
    )
    # plot average test loss
    ax1.plot(
        prop_train_schedule[0],
        mean_test_losses,
        label="Avg Test Loss",
        color=colors[1],
        marker=".",
    )
    # plot standard deviation envelope for test losses
    ax1.fill_between(
        prop_train_schedule[0],
        mean_test_losses - std_test_losses,
        mean_test_losses + std_test_losses,
        color=colors[1],
        alpha=0.1,
    )

    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax1.legend(loc="upper left")

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=colors[2])
    if plot_individual_runs:
        for i in range(n_runs):
            ax2.plot(
                prop_train_schedule[i],
                accuracies[i],
                color=colors[2],
                alpha=0.2,
            )
    # plot average accuracy
    ax2.plot(
        prop_train_schedule[0],
        mean_accuracies,
        label="Avg Accuracy",
        color=colors[2],
        marker=".",
    )
    # plot standard deviation envelope for accuracies
    ax2.fill_between(
        prop_train_schedule[0],
        mean_accuracies - std_accuracies,
        mean_accuracies + std_accuracies,
        color=colors[2],
        alpha=0.1,
    )
    ax2.tick_params(axis="y", labelcolor=colors[2])
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.savefig(save_path)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.close()


def plot_quantity(
    results: Dict[str, Dict[int, List[float]]],
    save_dir: str = "plots",
    save_name: str = "loss_accuracy",
    title: str = "Binary Linear Classifier",
    label: str = "Accuracy",
    legend_loc: str = "upper right",
) -> None:
    fig, ax1 = plt.subplots(figsize=(4, 3))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "D", "v", "^", "<", ">", "p", "*", "h", "H", "X", "D", "d"]
    for i, (run_name, quantity_dict) in enumerate(results.items()):
        print(f"Processing {run_name}")
        n_runs = len(quantity_dict)
        assert n_runs == len(
            quantity_dict
        ), "Number of runs must match number of quantities"
        print("n_runs: ", n_runs)

        num_samples_list = list(quantity_dict.keys())
        quantity_history_list = list(quantity_dict.values())
        num_samples_list, quantity_history_list = zip(
            *sorted(zip(num_samples_list, quantity_history_list))
        )
        # # Pad the lists with NaN to handle inhomogeneous shapes
        # padded_values = list(zip_longest(*quantity_dict.values(), fillvalue=np.nan))
        # mean_quantities_list = np.nanmean(padded_values, axis=1)
        # std_quantities_list = np.nanstd(padded_values, axis=1)
        mean_quantities_list = np.array(
            [np.mean(quantities) for quantities in quantity_history_list]
        )
        std_quantities_list = np.array(
            [np.std(quantities) for quantities in quantity_history_list]
        )
        ax1.plot(
            num_samples_list,
            mean_quantities_list,
            label=run_name,
            marker=markers[i],
            markersize=2,
            linewidth=1,
            color=colors[i],
        )
        ax1.fill_between(
            num_samples_list,
            mean_quantities_list - std_quantities_list,
            mean_quantities_list + std_quantities_list,
            alpha=0.2,
            color=colors[i],
        )

    ax1.set_xlabel(r"$N_{Train}$")
    ax1.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    ax1.set_ylabel(label)
    ax1.legend(loc=legend_loc)
    plt.title(title)
    plt.savefig(save_path)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.close()


def plot_results(
    run_json_paths: Dict[str, List[str]],
    title: str,
    save_dir: str,
    save_name: str,
) -> None:
    accuracies = defaultdict(lambda: defaultdict(list))
    test_losses = defaultdict(lambda: defaultdict(list))
    num_samples_schedule = defaultdict(lambda: defaultdict(list))
    for run_name, json_paths_lst in run_json_paths.items():
        print(f"Processing {run_name}")
        print(json_paths_lst)
        for json_path in json_paths_lst:
            print(f"Processing {json_path}")
            with open(json_path, "r") as f:
                results_dict = json.load(f)["results"]
                num_samples_schedule = results_dict["num_train_samples"]
                for group_idx, group in enumerate(num_samples_schedule):
                    if all(x == group[0] for x in group):
                        num_samples = group[0]
                    else:
                        raise ValueError(
                            f"Warning: Not all elements in group {group} are equal."
                        )
                    accuracies[run_name][num_samples].extend(
                        results_dict["accuracies"][group_idx]
                    )
                    test_losses[run_name][num_samples].extend(
                        results_dict["test_losses"][group_idx]
                    )

    plot_quantity(
        results=accuracies,
        save_name=f"{save_name}_accuracies",
        title=title,
        label="Test Acc",
        save_dir=save_dir,
        legend_loc="lower right",
    )

    plot_quantity(
        results=test_losses,
        save_name=f"{save_name}_test_losses",
        title=title,
        label="Test Loss",
        save_dir=save_dir,
    )


if __name__ == "__main__":
    # model_name = "LinearBinaryClassifier"
    # run_names_dict = {
    #     "Diffusion": [
    #         f"edm_imagenet64_bs64_MSELoss_english-springer_french-horn_run{idx}"
    #         for idx in range(1, 5)
    #     ],
    #     "GMM": [
    #         f"gmm_edm_imagenet64_bs64_MSELoss_english-springer_french-horn_run{idx}"
    #         for idx in range(1, 5)
    #     ],
    # }

    # run_names_dict = {
    #     "Diffusion": [
    #         f"edm_imagenet64_bs128_MSELoss_church_tench_{timestamp}"
    #         for timestamp in ["11-24_14-12-28", "11-24_20-47-48"]
    #     ],
    #     "GMM": [
    #         f"gmm_edm_imagenet64_bs128_MSELoss_church_tench_{timestamp}"
    #         for timestamp in ["11-24_14-12-28", "11-24_20-47-48"]
    #     ],
    # }

    model_name = "LinearMulticlassClassifier"

    run_names_dict = {
        "Diffusion": [
            f"edm_imagenet64_bs128_MSELoss_english_springer_french_horn_church_tench_{timestamp}"
            for timestamp in ["11-24_17-32-12", "11-24_19-55-28"]
        ],
        "GMM": [
            f"gmm_edm_imagenet64_bs128_MSELoss_english_springer_french_horn_church_tench_{timestamp}"
            for timestamp in ["11-24_17-32-12", "11-24_19-55-28"]
        ],
    }

    json_dir = f"results/classifier/{model_name}"
    # json_dir = "results/classifier/backup"
    run_json_paths = {
        name: [
            os.path.join(json_dir, f"{run_name}.json")
            for run_name in run_names_dict[name]
        ]
        for name in run_names_dict
    }
    save_name = f"{model_name}_classifier"
    plot_results(
        run_json_paths,
        title="Linear Multiclass Classifier",
        save_dir="final_plots/classifier",
        save_name=save_name,
    )
