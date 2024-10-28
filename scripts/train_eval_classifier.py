import hydra

from diffusion_gmm.classifier import LinearClassifierExperiment


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    experiment = LinearClassifierExperiment(**cfg.classifier_experiment)
    experiment.run(cfg.classifier_experiment.num_samples)


if __name__ == "__main__":
    main()
