import os

import hydra

from diffusion_gmm.gram_spectrum import (
    GramSpectrumCNNExperiment,
    GramSpectrumExperiment,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "vision_datasets")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    if cfg.gram_experiment.cnn_model_id is not None:
        if cfg.gram_experiment.hook_layer is None:
            raise ValueError("Hook layer must be specified if CNN model ID is provided")
        experiment = GramSpectrumCNNExperiment(
            data_dir=cfg.gram_experiment.data_dir,
            dataset_name=cfg.gram_experiment.dataset_name,
            load_npy=cfg.gram_experiment.load_npy,
            batch_size=cfg.gram_experiment.batch_size,
            custom_transform=cfg.gram_experiment.custom_transform,
            verbose=cfg.gram_experiment.verbose,
            cnn_model_id=cfg.gram_experiment.cnn_model_id,
            hook_layer=cfg.gram_experiment.hook_layer,
            rseed=cfg.rseed,
        )
    else:
        experiment = GramSpectrumExperiment(
            data_dir=cfg.gram_experiment.data_dir,
            dataset_name=cfg.gram_experiment.dataset_name,
            load_npy=cfg.gram_experiment.load_npy,
            batch_size=cfg.gram_experiment.batch_size,
            custom_transform=cfg.gram_experiment.custom_transform,
            verbose=cfg.gram_experiment.verbose,
            rseed=cfg.rseed,
        )
    experiment.run(
        num_samples=cfg.gram_experiment.num_samples,
        save_dir=cfg.gram_experiment.save_dir,
        save_name=cfg.gram_experiment.save_name,
        target_class=cfg.gram_experiment.target_class,
    )


if __name__ == "__main__":
    main()
