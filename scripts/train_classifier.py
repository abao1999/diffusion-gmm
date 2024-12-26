import json
import logging
import os
from typing import Any, Dict, List, Union

import hydra
import numpy as np
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim import lr_scheduler
from tqdm import tqdm

import wandb
from diffusion_gmm import classifier
from diffusion_gmm.classifier import ClassifierExperiment
from diffusion_gmm.utils import (
    get_img_shape,
    make_balanced_subsets,
    set_seed,
)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # Convert the entire config to a dictionary
    cfg_dict = OmegaConf.to_container(cfg.classifier, resolve=True)
    cfg_dict["rseed"] = cfg.rseed  # type: ignore
    cfg_dict["run_name"] = cfg.run_name  # type: ignore
    logger.info(cfg_dict)

    # set up wandb project and logging if enabled
    if cfg.wandb.log:
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            config=cfg_dict,  # type: ignore
            sync_tensorboard=False,
            group=cfg.wandb.group_name,
            resume=cfg.wandb.resume,
            tags=cfg.wandb.tags,
        )

    # set torch, cuda, and cudnn seeds
    set_seed(cfg.rseed)
    logger.info(f"Setting torch random seed to {cfg.rseed}")
    rng = np.random.default_rng(cfg.rseed)

    train_subset, test_subset = make_balanced_subsets(
        class_list=cfg.classifier.class_list,
        data_dir=cfg.classifier.train_data_dir,
        max_allowed_samples_per_class=cfg.classifier.max_allowed_samples_per_class,
        train_split=cfg.classifier.train_split,
        train_augmentations=None,
        rng=rng,
        verbose=cfg.classifier.verbose,
    )

    if test_subset is None:
        raise ValueError("Test subset is required")

    model_cls = getattr(classifier, cfg.classifier.model.name)
    model_kwargs = getattr(
        cfg.classifier.model, f"{cfg.classifier.model.name}_kwargs", {}
    )
    model_kwargs.update(
        {
            "use_bias": cfg.classifier.model.use_bias,
            "output_logit": cfg.classifier.model.output_logit,
        }
    )
    logger.info(f"Model kwargs: {model_kwargs}")

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

    img_shape = get_img_shape(train_subset.dataset)  # type: ignore
    if img_shape != get_img_shape(test_subset.dataset):  # type: ignore
        raise ValueError("Train and test subsets have different image shapes")
    input_dim = int(np.prod(img_shape))
    print(f"Input dimension: {input_dim}")

    experiment = ClassifierExperiment(
        input_dim=input_dim,
        class_list=cfg.classifier.class_list,
        model_class=model_cls,
        model_kwargs=model_kwargs,
        criterion_class=loss_fn,
        optimizer_class=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_class=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        lr=cfg.classifier.lr,
        device=cfg.classifier.device,
        verbose=cfg.classifier.verbose,
    )

    os.makedirs(cfg.classifier.save_dir, exist_ok=True)
    save_name = f"{cfg.classifier.save_name}.json"
    results_file_path = os.path.join(cfg.classifier.save_dir, save_name)

    model_save_dir = (
        os.path.join(
            cfg.classifier.model_save_dir,
            cfg.classifier.model.name,
            os.path.basename(cfg.classifier.train_data_dir),
        )
        if cfg.classifier.model_save_dir
        else None
    )

    if cfg.classifier.sweep_mode:
        assert (
            cfg.classifier.n_props_train == cfg.classifier.n_runs == 1
        ), "Sweep mode is enabled, so n_props_train and n_runs must be 1"

    n_train_per_class_schedule = (
        np.linspace(1.0, 0.05, cfg.classifier.n_props_train)
        * cfg.classifier.n_train_samples_per_class
    ).astype(int)
    # n_train_per_class_schedule = np.array(
    #     [2048, 1658, 1269, 880, 491, 256, 102]
    # ).astype(int)

    logger.info(
        f"Running classifier experiment, {cfg.classifier.class_list} classes, "
        f"num train samples per class schedule: {n_train_per_class_schedule}, "
        f"num test samples per class: {cfg.classifier.n_test_samples_per_class}, "
        f"Saving results to {results_file_path}",
    )

    n_runs = cfg.classifier.n_runs
    rng_stream = rng.spawn(n_runs)

    results_all_runs = []

    for run_idx in tqdm(range(n_runs), desc="Running classifier experiment"):
        rng = rng_stream[run_idx]

        if cfg.classifier.reset_model_random_seed:
            rseed = rng.integers(np.iinfo(np.int32).max)
            print(f"Setting torch random seed to {rseed} for run {run_idx}")
            set_seed(rseed)

        result_dict_lst: List[Dict[str, Any]] = experiment.run(
            train_subset=train_subset,
            test_subset=test_subset,
            rng=rng,
            n_train_per_class_schedule=n_train_per_class_schedule,  # type: ignore
            n_test_samples_per_class=cfg.classifier.n_test_samples_per_class,
            num_epochs=cfg.classifier.num_epochs,
            early_stopping_patience=cfg.classifier.early_stopping_patience,
            eval_epoch_interval=cfg.classifier.eval_epoch_interval,
            batch_size=cfg.classifier.batch_size,
            dataloader_kwargs=cfg.classifier.dataloader_kwargs,
            model_save_dir=model_save_dir if run_idx == 0 else None,
            verbose=cfg.classifier.verbose,
            log_wandb=cfg.wandb.log,
        )

        if cfg.wandb.log and not cfg.classifier.sweep_mode:
            for i, n_train_per_class in enumerate(n_train_per_class_schedule):
                wandb.log(result_dict_lst[i], step=n_train_per_class)

        results_all_runs.append(result_dict_lst)

        combined_results_dict: Dict[str, List[List[Union[float, int]]]] = {
            key: [
                [result[i][key] for result in results_all_runs]
                for i in range(len(n_train_per_class_schedule))
            ]
            for key in [
                "num_train_samples",
                "accuracy",
                "best_acc",
                "test_loss",
                "train_loss",
                "epoch",
            ]
        }
        # save results to json file
        with open(results_file_path, "w") as results_file:
            json.dump(
                {"results": combined_results_dict, "config": cfg_dict},
                results_file,
                indent=4,
            )

    # terminate wandb run after training
    if cfg.wandb.log:
        run.finish()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
