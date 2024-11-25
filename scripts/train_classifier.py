import json
import logging
import os

import hydra
import numpy as np
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim import lr_scheduler

from diffusion_gmm import classifier
from diffusion_gmm.classifier import ClassifierExperiment
from diffusion_gmm.utils import (
    make_balanced_subsets,
    set_seed,
)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # Convert the entire config to a dictionary
    cfg_dict = OmegaConf.to_container(cfg.classifier, resolve=True)
    cfg_dict["rseed"] = cfg.rseed  # type: ignore

    logger.info(cfg_dict)

    # set torch, cuda, and cudnn seeds
    set_seed(cfg.rseed)
    rng = np.random.default_rng(cfg.rseed)

    train_subset, test_subset = make_balanced_subsets(
        class_list=cfg.classifier.class_list,
        data_dir=cfg.classifier.train_data_dir,
        max_allowed_samples_per_class=cfg.classifier.max_allowed_samples_per_class,
        train_split=cfg.classifier.train_split,
        train_augmentations=None,
        rng=rng if cfg.classifier.resample_train_subset else None,
        verbose=cfg.classifier.verbose,
    )

    if cfg.classifier.test_data_dir is not None:
        test_subset, _ = make_balanced_subsets(
            class_list=cfg.classifier.class_list,
            data_dir=cfg.classifier.test_data_dir,
            max_allowed_samples_per_class=cfg.classifier.max_allowed_samples_per_class_test,
            train_split=1.0,
            train_augmentations=None,
            rng=rng if cfg.classifier.resample_test_subset else None,
            verbose=cfg.classifier.verbose,
        )

    if test_subset is None:
        raise ValueError("Test subset is required")

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

    prop_train_schedule = np.linspace(1.0, 0.05, cfg.classifier.n_props_train)

    save_dir = os.path.join(cfg.classifier.save_dir, cfg.classifier.model.name)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{cfg.classifier.save_name}.json"
    results_file_path = os.path.join(save_dir, save_name)

    results_dict = experiment.run(
        prop_train_schedule=prop_train_schedule,  # type: ignore
        n_runs=cfg.classifier.n_runs,
        reset_model_random_seed=cfg.classifier.reset_model_random_seed,
        num_epochs=cfg.classifier.num_epochs,
        batch_size=cfg.classifier.batch_size,
        dataloader_kwargs=cfg.classifier.dataloader_kwargs,
        save_results_path=results_file_path,
        save_interval=1,
        early_stopping_patience=cfg.classifier.early_stopping_patience,
        verbose=cfg.classifier.verbose,
    )

    with open(results_file_path, "w") as results_file:
        json.dump(
            {"results": results_dict, "config": cfg_dict},
            results_file,
            indent=4,
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
