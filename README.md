# JAX Capability Analysis

This repository contains the demo project for my analysis of the capabilities of the Python library **JAX**.

## 🛠️ Installation

This project only works on **Linux x86_64** systems that have a **CUDA compatible GPU** and **CUDA 12.0** as well as CuDNN 8.1 installed. The project has been tested on Ubuntu 20.04 with a NVIDIA GeForce RTX 3060 Ti hosted on [Genesis Cloud](https://genesiscloud.com/).

Install the Python 3.11 project using the following steps:

```bash
# create a virtual environment
python3.11 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the dependencies
poetry install
```

## 🌋 Run the Analysis Scripts

The analysis scripts have been structured as a CLI using Click (https://click.palletsprojects.com/en/8.1.x/). The following commands will show you how to run the experiments of the different sections:

```bash
# show help menu of the whole project
python main.py --help

# show help menu of the individual sections
python main.py xla --help
python main.py autograd --help
python main.py deep-learning --help
python main.py scipy --help

# an example of how to run the XLA experiment for
# jax with a non-default matrix size
python main.py xla jax --matrix-size 1000
```

⚠️ After calling the CLI, it might take a few seconds until the experiment starts. This is mainly because the PyTorch library is very big and has to be loaded into memory.

TODO: add improvements here in the README
