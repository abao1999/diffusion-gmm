import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["ggplot", "custom_style.mplstyle"])


def plot_quantity(
    results: Dict[str, Dict[int, List[float]]],
    save_dir: str,
    save_name: str,
    title: str,
    label: str,
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

        n_train_per_class_list = list(quantity_dict.keys())
        quantity_history_list = list(quantity_dict.values())
        if use_percentage:
            quantity_history_list = [np.array(x) * 100 for x in quantity_history_list]
        n_train_per_class_list, quantity_history_list = zip(
            *sorted(zip(n_train_per_class_list, quantity_history_list))
        )
        for n_train_per_class, quantities in zip(
            n_train_per_class_list, quantity_history_list
        ):  # type: ignore
            print(f"{len(quantities)} runs for {n_train_per_class} samples per class")
        mean_quantities_list = np.array(
            [np.mean(quantities) for quantities in quantity_history_list]
        )
        std_quantities_list = np.array(
            [np.std(quantities) for quantities in quantity_history_list]
        )
        print(n_train_per_class_list)
        ax1.plot(
            n_train_per_class_list,
            mean_quantities_list,
            label=run_name,
            marker=markers[i],
            markersize=2,
            linewidth=1,
            color=colors[i],
        )
        ax1.fill_between(
            n_train_per_class_list,
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
    title: str,
    save_dir: str = "plots",
    save_name: str = "loss_accuracy",
    num_classes: Optional[int] = None,
    plot_best_acc: bool = True,
    sample_splits_to_exclude: List[int] = [],
) -> None:
    accuracy = defaultdict(lambda: defaultdict(list))
    test_loss = defaultdict(lambda: defaultdict(list))

    for run_name, json_paths_lst in run_json_paths.items():
        print(f"Processing {run_name}")
        for json_path in json_paths_lst:
            print(f"Processing {json_path}")
            with open(json_path, "r") as f:
                results_dict = json.load(f)["results"]
                if "n_train_per_class" in results_dict:
                    n_train_per_class_schedule = results_dict["n_train_per_class"]
                else:
                    # backwards compatibility with old results that saved total number of train samples instead of per class
                    if num_classes is None:
                        raise ValueError(
                            "num_classes must be provided when using old results"
                        )
                    # n_train_per_class_schedule = (
                    #     np.array(results_dict["num_train_samples"]) // num_classes
                    # ).tolist()
                    n_train_per_class_schedule = [
                        (np.array(ns) // num_classes).tolist()
                        for ns in results_dict["num_train_samples"]
                    ]
                print(n_train_per_class_schedule)
                for group_idx, group in enumerate(n_train_per_class_schedule):
                    n_train_per_class = group[0]
                    print(n_train_per_class)
                    if n_train_per_class in sample_splits_to_exclude:
                        print(
                            f"Excluding results from train on {n_train_per_class} per class from {run_name}"
                        )
                        continue

                    if not all(x == n_train_per_class for x in group):
                        raise ValueError(
                            f"Warning: Not all elements in group {group} are equal."
                        )

                    accuracy[run_name][n_train_per_class].extend(
                        results_dict["best_acc"][group_idx]
                        if "best_acc" in results_dict and plot_best_acc
                        else results_dict["accuracy"][group_idx]
                    )
                    test_loss[run_name][n_train_per_class].extend(
                        results_dict["test_loss"][group_idx]
                        if "test_loss" in results_dict
                        else []
                    )

    plot_quantity(
        results=accuracy,  # type: ignore
        save_name=f"{save_name}_accuracy",
        title=title,
        label="Test Acc (%)",
        use_percentage=True,
        save_dir=save_dir,
        legend_loc="lower right",
    )

    plot_quantity(
        results=test_loss,  # type: ignore
        save_name=f"{save_name}_test_loss",
        title=title,
        label="Test Loss",
        save_dir=save_dir,
    )


if __name__ == "__main__":
    # model_name = "LinearMulticlassClassifier"
    # run_name = "goldfinch-patas_monkey-trimaran-tabby"
    # # run_name = "church-tench-english_springer-french_horn"
    # n_classes = 4
    # # # run_name = "10_classes"
    # # # n_classes = 10

    # # class_list = ["church", "tench", "english_springer", "french_horn"]
    # class_list = [
    #     "english_springer",
    #     "french_horn",
    #     "church",
    #     "tench",
    # ]

    # n_classes = len(class_list)
    # run_name = "-".join(class_list)

    model_name = "LinearBinaryClassifier"
    class_list = ["goldfinch", "trimaran"]
    n_classes = len(class_list)
    run_name = "-".join(class_list)

    print(run_name)

    json_dir = os.path.join("results/classifier", f"{model_name}", f"{run_name}")
    print(json_dir)
    run_json_paths = {
        "Diffusion": glob.glob(
            os.path.join(json_dir, f"edm_imagenet64_*{run_name}*.json"),
        ),
        "GMM": glob.glob(
            os.path.join(json_dir, f"gmm_edm_imagenet64_*{run_name}*.json")
        ),
    }
    # run_json_paths = {
    #     "Representations": glob.glob(
    #         os.path.join(json_dir, f"representations_*{run_name}*.json"),
    #     ),
    #     "GMM": glob.glob(
    #         os.path.join(json_dir, f"gmm_representations_*{run_name}*.json")
    #     ),
    # }
    print(run_json_paths)
    save_name = f"{model_name}_{run_name}"
    plot_results(
        run_json_paths,
        # num_classes=n_classes,
        title="Linear Classifier",
        save_dir="final_plots/classifier",
        save_name=save_name,
        plot_best_acc=True,
        # sample_splits_to_exclude=[512, 128, 1024],
        # sample_splits_to_exclude=[4096],
    )
