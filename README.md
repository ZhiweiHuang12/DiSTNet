# MasterDPM: Diffusion Probabilistic Models for Master Equation Solving and Inference

<p align="center">
  <!-- Add a relevant image or logo here if available -->
</p>

## Brief Introduction

DiSTNet is a framework designed to solve and infer master equations using diffusion probabilistic models (DPMs). The framework leverages conditional diffusion models to handle stochastic dynamics, enabling both the solution of master equations and the inference of parameters from observed data. This approach provides an effective way to model complex stochastic systems in various domains, such as chemical reactions, population dynamics, and biological networks.

The framework is extensible and user-friendly. Users can define their specific master equation problems by providing configuration files, allowing for both forward solving and inverse inference tasks. MasterDPM consists of two main components: MasterDPM-Solver and MasterDPM-Inferrer.

## DiSTNet-Solver: Solving Master Equations with Diffusion Models

Traditional approaches to solving master equations face significant computational challenges, especially for high-dimensional systems. Analytical and numerical methods often require substantial computational resources or trade accuracy for efficiency. MasterDPM-Solver employs a conditional diffusion model architecture based on U-Net to efficiently approximate the solution of complex master equations.

The diffusion process gradually adds noise to the system state, while the reverse process learns to denoise step-by-step, effectively learning the underlying dynamics of the system governed by the master equation.

## DiSTNet-Inferrer: Parameter Inference from Observed Data

Parameter inference from observed data is crucial for understanding stochastic systems. MasterDPM-Inferrer uses a trained neural network to efficiently estimate posterior distributions of master equation parameters through conditional sampling. By leveraging automatic differentiation and neural approximations, the inferrer can efficiently learn system parameters from limited observations.

## Installation

Run the following command to create a new environment and activate the environment:

```bash
conda create --name masterdpm_env python=3.8
conda activate masterdpm_env
```

Download the code and use the `cd` command to navigate to the directory containing the setup files:

```bash
git clone <repo-url>
cd distnet
```

Install the required dependencies:

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install .
```

## Usage

To train and sample from the conditional diffusion model:

```bash
python train_and_sample_conditional_diffusion.py
```

The script supports various datasets including:
- Synthetic biological models (Auto-feedback loops, ON-OFF networks, etc.)

## Architecture

The framework consists of:

- **DDPM Module**: Core diffusion probabilistic model implementation
- **Conditional UNet**: Neural network architecture for conditional generation
- **Data Handlers**: Specialized loaders for different types of master equation data
- **Training Utilities**: Components for training and evaluation
- **Sampling Methods**: Various inference algorithms for model sampling

## Supported Systems

The framework can model various stochastic systems described by master equations:

- Auto-feedback loop models
- ON-OFF networks
- SIRS epidemic models
- Delay differential systems
- Toggle switches
- Center dogma models

## Applications

DiSTNet can be used for:

- Solving complex master equations efficiently using diffusion models
- Inferring parameters of stochastic systems from observational data
- Modeling biological and chemical reaction networks
- Generating synthetic data consistent with known dynamics
- Uncertainty quantification in stochastic systems
