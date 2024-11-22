import logging

import hydra
import torchvision.transforms as transforms

from diffusion_gmm.image_gmm import ImageGMM
from diffusion_gmm.utils.data_utils import setup_dataset


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    cfg_dict = {**cfg.gmm, "rseed": cfg.rseed}
    logger.info(f"cfg_dict: {cfg_dict}")

    logger.info(
        f"\nFitting GMM on {cfg.gmm.n_samples_fit} samples from {cfg.gmm.data_dir}.\n"
        f"Class list: {cfg.gmm.class_list}.\n"
        f"Saving {cfg.gmm.n_samples_generate} samples to {cfg.gmm.save_dir}."
    )

    dataset, is_npy_dataset = setup_dataset(
        cfg.gmm.data_dir, dataset_name=cfg.gmm.dataset_name
    )
    dataset.transform = transforms.ToTensor() if not is_npy_dataset else None
    logger.info(f"dataset: {dataset}")

    gmm = ImageGMM(
        n_components=cfg.gmm.n_components,
        dataset=cfg.gmm.dataset,
        class_list=cfg.gmm.class_list,
        covariance_type=cfg.gmm.covariance_type,
        verbose=True,
        rseed=cfg.rseed,
    )

    # gmm.fit(cfg.gmm.n_samples_fit)
    # gmm.save_samples(n_samples=cfg.gmm.n_samples_generate, save_dir=cfg.gmm.save_dir)

    class_stats = gmm.compute_mean_and_covariance(
        num_samples_per_class=cfg.gmm.n_samples_fit
    )

    gmm.sample_from_computed_stats(
        class_stats=class_stats,
        n_samples_per_class=cfg.gmm.n_samples_generate,
        save_dir=cfg.gmm.save_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
