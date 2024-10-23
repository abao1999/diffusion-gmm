import hydra

from diffusion_gmm.image_gmm import ImageGMM

FIGS_DIR = "figs"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    gmm = ImageGMM(
        n_components=cfg.gmm.n_components,
        datapath=cfg.gmm.datapath,
        dataset_name=cfg.gmm.dataset_name,
        batch_size=cfg.gmm.batch_size,
        custom_transform=cfg.gmm.custom_transform,
        verbose=True,
    )

    gmm.fit(cfg.gmm.n_samples_fit)

    gmm.save_samples(
        n_samples=cfg.gmm.n_samples_generate,
        save_dir=cfg.gmm.save_dir,
        plot_kwargs={
            "save_grid_dir": FIGS_DIR,
            "save_grid_shape": (10, 10),
            "process_fn": None,
        },
    )


if __name__ == "__main__":
    main()
