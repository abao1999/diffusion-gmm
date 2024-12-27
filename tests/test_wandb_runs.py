import argparse

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    args = parser.parse_args()
    sweep_id = args.sweep_id
    print(sweep_id)

    api = wandb.Api()
    sweep = api.sweep(f"abao/diffusion-gmm/sweeps/{sweep_id}")
    # Get best run parameters
    best_run = sweep.best_run(order="test_loss")
    parameters = best_run.config
    print(f"parameters: {parameters}")
    print(f"parameter keys: {list(parameters.keys())}")

    print(f"run name: {best_run.name}")
    summary_params_lst = [
        "use_bias",
        "output_logit",
        "n_train_samples_per_class",
        "n_test_samples_per_class",
        "rseed",
        "train_data_dir",
    ]
    for param in summary_params_lst:
        if param in parameters:
            print(f"{param}: {parameters[param]}")

    best_sweeped_parameters = {
        "lr": parameters["lr"],
        "batch_size": parameters["batch_size"],
        "CosineAnnealingWarmRestarts.T0": parameters[
            "classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_0"
        ],
        "CosineAnnealingWarmRestarts.eta_min": parameters[
            "classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.eta_min"
        ],
    }
    print(f"best_sweeped_parameters: {best_sweeped_parameters}")
    print(f"best_run.summary keys: {best_run.summary.keys()}")
    for key in best_run.summary.keys():
        print(f"{key}: {best_run.summary[key]}")

    final_metrics_lst = ["best_acc", "best_test_loss", "final_acc"]
    for metric in final_metrics_lst:
        metric_value = best_run.history(keys=[metric], pandas=False)
        print(f"{metric}: {metric_value}")
