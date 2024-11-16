import logging

import hydra

from diffusion_gmm.image_gmm import ImageGMM


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    gmm = ImageGMM(
        n_components=cfg.gmm.n_components,
        data_dir=cfg.gmm.data_dir,
        dataset_name=cfg.gmm.dataset_name,
        covariance_type=cfg.gmm.covariance_type,
        custom_transform=cfg.gmm.custom_transform,
        verbose=True,
        rseed=cfg.rseed,
    )

    # gmm.fit(
    #     cfg.gmm.n_samples_fit,
    #     target_class=cfg.gmm.target_class,
    #     batch_size=cfg.gmm.batch_size,
    # )

    # gmm.save_samples(
    #     n_samples=cfg.gmm.n_samples_generate,
    #     save_dir=cfg.gmm.save_dir,
    # )

    # # means = gmm.means_
    # print("means.shape: ", means.shape)  # type: ignore
    # covariances = gmm.covariances_
    # print("covariances.shape: ", covariances.shape)  # type: ignore

    mean, covariance = gmm.compute_mean_and_covariance(
        num_samples=cfg.gmm.n_samples_fit,
        target_class=cfg.gmm.target_class,
        batch_size=cfg.gmm.batch_size,
    )

    gmm.save_samples_single_class(
        mean=mean,
        covariance=covariance,
        n_samples=cfg.gmm.n_samples_generate,
        save_dir=cfg.gmm.save_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
