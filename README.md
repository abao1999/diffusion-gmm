# diffusion-gmm
Investigating the representations of data generated from diffusion models

## Setup
To setup, make a conda environment with `python=3.10` and run:
```
$ pip install -r requirements.txt
$ pip install -e .
```

## Generate Images from Diffusion Model
In the [scripts](scripts/) directory, we provide a script [generate_images.py](scripts/generate_images.py) to generate image samples from pre-trained Diffusion generative models. And the generated samples are saved into a specified directory, as image files. Additional choices of diffusion models can be added in [diffusions.py](diffusion_gmm/diffusions.py), which all currently load build from the `diffusers` library.

`python scripts/generate_images.py --steps 50 --n_samples 1024`

## Generate Images from GMM
In the [scripts](scripts/) directory, we provide a script [sample_gmm.py](scripts/sample_gmm.py) to sample from a Gaussian Mixture Model (GMM) fit on the flattened pixel-wise means and covariances computed from images stored in a directory. This uses a custom ImageGMM class defined in [gmm.py](diffusion_gmm/gmm.py) which fits a GMM with the specificed parameters and converts the generated samples to the shape of the original images used for fitting the model. And the generated samples are saved into a specified directory, as image files.

`python scripts/sample_gmm.py`

## Compute Gram Matrix Spectrum
In the [scripts](scripts/) directory, we provide a script [gram_spectrum.py](scripts/gram_spectrum.py) to compute all eigenvalues of the gram matrices created from the specified samples from a chosen dataset (real `cifar10` images, Diffusion generated images, GMM samples). All eigenvalues are saved into npy files in a specified directory, for downsteam analysis.

To compute the gram matrix spectrum for the real `cifar10` data, run:

`python scripts/gram_spectrum.py real --num_images 1024`

To compute the gram matrix spectrum for the Diffusion generated data, run:

`python scripts/gram_spectrum.py diffusion --num_images 1024`

To compute the gram matrix spectrum for the GMM samples fit on the real data, run:

`python scripts/gram_spectrum.py gmm --num_images 1024`

Once all npy files have been saved, the spectra can be plotted together with `python scripts/plot_spectra.py`.

## TODOS
- Fix and verify scaling for GMM
- Ensure consistent data processing for all modes
- Tests to compute and compare sample statistics
- Interpretation