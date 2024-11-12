# diffusion-gmm
Investigating the representations of data generated from diffusion models

## Setup
To setup, make a conda environment with `python=3.10` and run:
```
$ pip install -e .
```

## Development Goals
+ Add on-the-fly augmentations (random crop, jitter, rotations, reflections) for classifier training
+ Validate results for Binary Linear Classifier, go to Multiclass setting -> fully connected two layer network -> Resnet
+ Try Toeplitz statistics for GMM samples
