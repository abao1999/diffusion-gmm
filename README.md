# diffusion-gmm
Investigating the characteristics of data generated from image diffusion generative models.
This repository contains the code to reproduce the experiments presented in our arXiv preprint [arXiv:2501.07741v2](https://arxiv.org/abs/2501.07741v2)

Paper abstract:

>"We show via a combination of mathematical arguments and empirical evidence that data distributions sampled from diffusion models satisfy an Approximate Concentration of Measure Property saying that any Lipschitz $1$-dimensional projection of a random vector is not too far from its mean with high probability. This implies that such models are quite restrictive and gives an explanation for a fact previously observed in the literature that conventional diffusion models cannot capture "heavy-tailed" data (i.e. data $\mathbf{X}$ for which the norm $\|\mathbf{X}\|_2$ does not possess a sub-Gaussian tail) well. We then proceed to train a generalized linear model using stochastic gradient descent (SGD) on the diffusion-generated data for a multiclass classification task and observe empirically that a Gaussian universality result holds for the test error. In other words, the test error depends only on the first and second order statistics of the diffusion-generated data in the linear setting. Results of such forms are desirable because they allow one to assume the data itself is Gaussian for analyzing performance of the trained classifier. Finally, we note that current approaches to proving universality do not apply to this case as the covariance matrices of the data tend to have vanishing minimum singular values for the diffusion-generated data, while the current proofs assume that this is not the case (see Subsection \ref{subsec: limits} for more details). This leaves extending previous mathematical universality results as an intriguing open question."

## Setup
To setup, make a conda environment with `python=3.10` and run:
```
$ pip install -e .
```

For reproducibility, we have provided some of our sweeps publicly at https://wandb.ai/abao/diffusion-gmm

## Citation
If you use this codebase or otherwise find our work valuable, please cite us:
```
@misc{ghane2025_com_diffusion,
      title={Concentration of Measure for Distributions Generated via Diffusion Models}, 
      author={Reza Ghane and Anthony Bao and Danil Akhtiamov and Babak Hassibi},
      year={2025},
      eprint={2501.07741},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2501.07741}, 
}
```
