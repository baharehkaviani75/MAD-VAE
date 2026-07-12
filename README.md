# MAD-VAE: Density Estimation Helps Adversarial Robustness

This repository contains the official PyTorch implementation of:

**"Density Estimation Helps Adversarial Robustness"**
Published in the 13th International Conference on Computer and Knowledge Engineering (ICCKE 2023).

The project proposes **MAD-VAE (Multi-head Adversarial Defense Variational Autoencoder)**, a framework that improves the robustness of deep neural networks against adversarial attacks by incorporating **density estimation into the representation learning process**.

---

## Abstract

Deep neural networks are highly vulnerable to adversarial examples, where small imperceptible perturbations can significantly change model predictions.

This work introduces a novel defense strategy based on learning a robust latent representation using a Variational Autoencoder. By modeling the data distribution in the latent space, the proposed method encourages the classifier to learn a more stable and compact feature manifold.

The framework jointly optimizes:

* Classification objective
* Reconstruction objective
* Latent distribution regularization

The learned density-aware latent representation helps the model distinguish natural samples from adversarial perturbations and improves robustness against various attack methods.

---

## Method Overview

The proposed MAD-VAE architecture consists of three main components:

1. **Encoder Network**

   Maps input images into a latent representation:

[
z = Encoder(x)
]

2. **Density Estimation Decoder**

   Reconstructs the input distribution and learns the underlying data manifold:

[
\hat{x}=Decoder(z)
]

3. **Classification Head**

   Performs classification using robust latent features:

[
y=Classifier(z)
]

The model jointly learns discriminative and generative representations, forcing adversarial examples to deviate from the learned latent distribution.

---

## Training Objective

The overall optimization objective combines classification loss, reconstruction loss, and KL divergence:

[
L = L_{classification} + \alpha L_{reconstruction} + \beta KL(q(z|x)||p(z))
]

where:

* `alpha` controls the contribution of reconstruction loss
* `beta` controls latent distribution regularization

Default configuration:

```python
z_dim = 128
alpha = 0.85
beta = 1
```

---

## Adversarial Attacks

The robustness of the proposed method is evaluated against several gradient-based adversarial attacks:

* FGSM (Fast Gradient Sign Method)
* R-FGSM (Randomized FGSM)
* MI-FGSM (Momentum Iterative FGSM)
* PGD (Projected Gradient Descent)

Evaluation metrics include:

* Clean accuracy
* Adversarial accuracy
* Attack success rate
* Latent space visualization

---

# Repository Structure

```
MAD-VAE/
│
├── models/
│   ├── madvae_mnist.py
│   ├── madvae_resnet.py
│
├── train/
│   ├── train_adv.py
│   ├── train_vanilla.py
│
├── utils/
│   ├── adversarial.py
│   ├── dataset.py
│   ├── generate_data.py
│   ├── loss_function.py
│   └── scheduler.py
│
├── datasets/
│
├── plots/
│
├── requirements.txt
│
└── README.md
```

---

# Supported Datasets

The implementation supports experiments on:

* MNIST
* CIFAR-10
* SVHN
* GTSRB
* CelebA

The architecture automatically adapts to different image sizes and numbers of classes.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your_username/MAD-VAE.git

cd MAD-VAE
```

Create an environment:

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

## Vanilla Classifier

Train a standard classifier without density estimation:

```bash
python train_vanilla.py
```

## MAD-VAE Training

Train the proposed adversarial defense model:

```bash
python train_adv.py
```

---

# Latent Space Visualization

The learned latent representations can be analyzed using dimensionality reduction techniques:

```bash
python visualize_umap.py
```

The visualization helps compare:

* Natural samples
* Adversarial samples
* Latent distribution separation

---

# Main Contributions

The main contributions of this work are:

* Introducing a density estimation based defense mechanism against adversarial attacks.
* Combining discriminative classification with generative latent representation learning.
* Learning a robust latent manifold to reduce adversarial vulnerability.
* Demonstrating improved robustness against multiple adversarial attack algorithms.

---

## Paper

Paper:
Density Estimation Helps Adversarial Robustness

ICCKE 2023

[Paper Link](https://www.researchgate.net/profile/Bahareh-Kaviani-Baghbaderani/publication/375978279_Density_Estimation_Helps_Adversarial_Robustness/links/65719cfcfc4b416622a503b3/Density-Estimation-Helps-Adversarial-Robustness.pdf)

# Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{kaviani2023density,
title={Density Estimation Helps Adversarial Robustness},
author={Kaviani Baghbaderani, Bahareh and others},
booktitle={13th International Conference on Computer and Knowledge Engineering (ICCKE)},
year={2023}
}
```

---

# License

This repository is released for research and educational purposes.
