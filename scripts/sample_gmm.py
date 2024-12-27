import logging
import os

import hydra
import numpy as np
import torchvision.transforms as transforms

from diffusion_gmm.image_gmm import ImageGMM
from diffusion_gmm.utils import setup_dataset


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    cfg_dict = {**cfg.gmm, "rseed": cfg.rseed}
    logger.info(f"cfg_dict: {cfg_dict}")

    logger.info(
        f"\nFitting GMM on {cfg.gmm.n_samples_fit} samples from {cfg.gmm.data_dir}.\n"
        f"Class list: {cfg.gmm.classes}.\n"
        f"Saving {cfg.gmm.n_samples_generate} samples to {cfg.gmm.save_dir}."
    )

    dataset, is_npy_dataset = setup_dataset(
        cfg.gmm.data_dir, dataset_name=cfg.gmm.dataset_name
    )
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None
    logger.info(f"dataset: {dataset}")

    gmm = ImageGMM(
        n_components=cfg.gmm.n_components,
        dataset=dataset,
        classes=cfg.gmm.classes,
        covariance_type=cfg.gmm.covariance_type,
        verbose=True,
        rseed=cfg.rseed,
    )

    # gmm.fit(cfg.gmm.n_samples_fit)
    # gmm.save_samples(n_samples=cfg.gmm.n_samples_generate, save_dir=cfg.gmm.save_dir)

    class_stats = gmm.compute_mean_and_covariance(
        num_samples_per_class=cfg.gmm.n_samples_fit,
    )

    if cfg.gmm.stats_save_dir is not None:
        os.makedirs(cfg.gmm.stats_save_dir, exist_ok=True)
        for class_name, stats in class_stats.items():
            mean = stats["mean"]
            covariance = stats["covariance"]
            np.save(
                os.path.join(cfg.gmm.stats_save_dir, f"{class_name}_mean.npy"), mean
            )
            np.save(
                os.path.join(cfg.gmm.stats_save_dir, f"{class_name}_covariance.npy"),
                covariance,
            )
            logger.info(
                f"Saved {class_name} mean and covariance to {cfg.gmm.stats_save_dir}"
            )

    # gmm.sample_from_computed_stats(
    #     class_stats=class_stats,
    #     n_samples_per_class=cfg.gmm.n_samples_generate,
    #     save_dir=cfg.gmm.save_dir,
    #     batch_size=cfg.gmm.batch_size,
    #     sample_idx=cfg.gmm.sample_idx,
    # )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
