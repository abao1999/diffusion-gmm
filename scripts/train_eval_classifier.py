import hydra

from diffusion_gmm.classifier import BinaryClassifierExperiment


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    experiment = BinaryClassifierExperiment(
        data_dir=cfg.classifier_experiment.data_dir,
        dataset_name=cfg.classifier_experiment.dataset_name,
        batch_size=cfg.classifier_experiment.batch_size,
        custom_transform=cfg.classifier_experiment.custom_transform,
        num_epochs=cfg.classifier_experiment.num_epochs,
        lr=cfg.classifier_experiment.lr,
        device=cfg.classifier_experiment.device,
        split_ratio=cfg.classifier_experiment.split_ratio,
        verbose=cfg.classifier_experiment.verbose,
        load_npy=cfg.classifier_experiment.load_npy,
        plot_history=cfg.classifier_experiment.plot_history,
        rseed=cfg.rseed,
    )

    experiment.run(
        cfg.classifier_experiment.class_list,
        num_samples=cfg.classifier_experiment.num_samples,
    )


if __name__ == "__main__":
    main()
