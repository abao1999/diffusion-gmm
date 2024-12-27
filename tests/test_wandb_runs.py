import wandb

api = wandb.Api()
sweep = api.sweep("abao/diffusion/sweeps/31nvxdsw")

# Get best run parameters
best_run = sweep.best_run(order="validation/accuracy")
best_parameters = best_run.config
print(best_parameters)
