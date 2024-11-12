import json
import logging
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import default_loader

from diffusion_gmm.classifier import BinaryLinearClassifier, ClassifierExperiment
from diffusion_gmm.utils import plot_training_history


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


def set_seed(rseed: int):
    """
    Set the seed for the random number generator for torch, cuda, and cudnn
    """
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

    torch.manual_seed(rseed)
    # If using CUDA, you should also set the seed for CUDA for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rseed)
        torch.cuda.manual_seed_all(rseed)  # if you have multiple GPUs

    # For deterministic behavior on GPU (reproducibility), use the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    logger.info(cfg.classifier)
    set_seed(cfg.rseed)

    # Check if the directory contains any .npy files
    load_npy = any(
        fname.endswith(".npy")
        for _, _, files in os.walk(cfg.experiment.data_dir)
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
        class_list=cfg.classifier.class_list,
        model_class=BinaryLinearClassifier,
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
        num_epochs=cfg.classifier.num_epochs,
        batch_size=cfg.classifier.batch_size,
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
