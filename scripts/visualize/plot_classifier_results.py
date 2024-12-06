import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["ggplot", "custom_style.mplstyle"])


def plot_quantity(
    results: Dict[str, Dict[int, List[float]]],
    num_classes: int,
    save_dir: str = "plots",
    save_name: str = "loss_accuracy",
    title: str = "Binary Linear Classifier",
    label: str = "Accuracy",
    use_percentage: bool = False,
    legend_loc: str = "upper right",
) -> None:
    fig, ax1 = plt.subplots(figsize=(4, 3))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["s", "o", "D", "v", "^", "<", ">", "p", "*", "h", "H", "X", "D", "d"]
    for i, (run_name, quantity_dict) in enumerate(results.items()):
        print(f"Processing {run_name}")
        num_train_splits = len(quantity_dict)
        assert num_train_splits == len(
            quantity_dict
        ), "Number of runs must match number of quantities"
        print("num_train_splits: ", num_train_splits)

        num_samples_list = list(quantity_dict.keys())
        num_samples_per_class_list = [x // num_classes for x in num_samples_list]
        quantity_history_list = list(quantity_dict.values())
        if use_percentage:
            quantity_history_list = [np.array(x) * 100 for x in quantity_history_list]
        num_samples_per_class_list, quantity_history_list = zip(
            *sorted(zip(num_samples_per_class_list, quantity_history_list))
        )
        for num_samples_per_class, quantities in zip(
            num_samples_per_class_list, quantity_history_list
        ):  # type: ignore
            print(
                f"{len(quantities)} runs for {num_samples_per_class} samples per class"
            )
        mean_quantities_list = np.array(
            [np.mean(quantities) for quantities in quantity_history_list]
        )
        std_quantities_list = np.array(
            [np.std(quantities) for quantities in quantity_history_list]
        )
        print(num_samples_per_class_list)
        ax1.plot(
            num_samples_per_class_list,
            mean_quantities_list,
            label=run_name,
            marker=markers[i],
            markersize=2,
            linewidth=1,
            color=colors[i],
        )
        ax1.fill_between(
            num_samples_per_class_list,
            mean_quantities_list - std_quantities_list,
            mean_quantities_list + std_quantities_list,
            alpha=0.2,
            color=colors[i],
        )

    ax1.set_xlabel(r"$N_{\text{Train per class}}$")
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
    num_classes: int,
    title: str,
    save_dir: str = "plots",
    save_name: str = "loss_accuracy",
) -> None:
    accuracies = defaultdict(lambda: defaultdict(list))
    test_losses = defaultdict(lambda: defaultdict(list))

    for run_name, json_paths_lst in run_json_paths.items():
        print(f"Processing {run_name}")
        for json_path in json_paths_lst:
            print(f"Processing {json_path}")
            with open(json_path, "r") as f:
                results_dict = json.load(f)["results"]
                num_samples_schedule = results_dict["num_train_samples"]

                for group_idx, group in enumerate(num_samples_schedule):
                    num_samples = group[0]
                    if not all(x == num_samples for x in group):
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
        results=accuracies,  # type: ignore
        num_classes=num_classes,
        save_name=f"{save_name}_accuracies",
        title=title,
        label="Test Acc (%)",
        use_percentage=True,
        save_dir=save_dir,
        legend_loc="lower right",
    )

    plot_quantity(
        results=test_losses,  # type: ignore
        num_classes=num_classes,
        save_name=f"{save_name}_test_losses",
        title=title,
        label="Test Loss",
        save_dir=save_dir,
    )


if __name__ == "__main__":
    # model_name = "LinearMulticlassClassifier"
    # run_name = "20_classes"
    # n_classes = 20

    # class_list = ["church", "tench", "english_springer", "french_horn"]
    # n_classes = len(class_list)
    # run_name = "-".join(class_list)

    model_name = "LinearBinaryClassifier"
    # class_list = ["church", "tench"]
    # class_list = ["racer", "mountain_bike"]
    # class_list = ["kimono", "coral_reef"]
    # class_list = ["garbage_truck", "polaroid"]
    class_list = ["english_springer", "french_horn"]
    n_classes = len(class_list)
    run_name = "-".join(class_list)

    print(run_name)

    json_dir = os.path.join("results/classifier", model_name, run_name)

    run_json_paths = {
        "Diffusion": glob.glob(
            os.path.join(json_dir, f"edm_imagenet64_*{run_name}*.json"),
        ),
        "GMM": glob.glob(
            os.path.join(json_dir, f"gmm_edm_imagenet64_*{run_name}*.json")
        ),
    }
    print(run_json_paths)
    save_name = f"{model_name}_{run_name}"
    plot_results(
        run_json_paths,
        num_classes=n_classes,
        title="Linear Classifier",
        save_dir="final_plots/classifier",
        save_name=save_name,
    )
