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
    if cfg.gs_exp.cnn_model_id is not None:
        if cfg.gs_exp.hook_layer is None:
            raise ValueError("Hook layer must be specified if CNN model ID is provided")
        experiment = GramSpectrumCNNExperiment(
            data_dir=cfg.gs_exp.data_dir,
            dataset_name=cfg.gs_exp.dataset_name,
            load_npy=cfg.gs_exp.load_npy,
            batch_size=cfg.gs_exp.batch_size,
            custom_transform=cfg.gs_exp.custom_transform,
            verbose=cfg.gs_exp.verbose,
            cnn_model_id=cfg.gs_exp.cnn_model_id,
            hook_layer=cfg.gs_exp.hook_layer,
            rseed=cfg.rseed,
        )
    else:
        experiment = GramSpectrumExperiment(
            data_dir=cfg.gs_exp.data_dir,
            dataset_name=cfg.gs_exp.dataset_name,
            load_npy=cfg.gs_exp.load_npy,
            batch_size=cfg.gs_exp.batch_size,
            custom_transform=cfg.gs_exp.custom_transform,
            verbose=cfg.gs_exp.verbose,
            rseed=cfg.rseed,
        )
    experiment.run(
        num_samples=cfg.gs_exp.num_samples,
        save_dir=cfg.gs_exp.save_dir,
        save_name=cfg.gs_exp.save_name,
        target_class=cfg.gs_exp.target_class,
    )


if __name__ == "__main__":
    main()
