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
    print("load_npy: ", load_npy)
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

    # .to(device) not necessary as this is done in ClassifierExperiment
    model = BinaryLinearClassifier(input_dim=input_dim)

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
        model=model,
        criterion_class=loss_fn,
        optimizer_class=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_class=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        lr=cfg.classifier.lr,
        num_epochs=cfg.classifier.num_epochs,
        train_split=cfg.classifier.train_split,
        batch_size=cfg.classifier.batch_size,
        device=cfg.classifier.device,
        rseed=cfg.classifier.rseed,
    )

    experiment.run(
        cfg.c