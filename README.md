# Sequential Optimal Experimental Design of Perturbation Screens

This repository hosts the code base for the paper

**Sequential Optimal Experimental Design of Perturbation Screens Guided by Multimodal Priors**\
Kexin Huang, Romain Lopez, Jan-Christian Hütter, Takamasa Kudo, Antonio Rios, Aviv Regev


## Overview

<p align="center"><img src="https://github.com/Genentech/iterative-perturb-seq/blob/master/img/illustration.png" alt="logo" width="800px" /></p>

Understanding a cell's transcriptional response to genetic perturbations answers vital biological questions such as cell reprogramming and target discovery. Despite significant advances in the Perturb-seq technology, the demand for vast experimental configurations surpasses the capacity for existing assays. Recent machine learning models, trained on existing Perturb-seq data sets, predict perturbation outcomes but face hurdles due to sub-optimal training set selection, resulting in weak predictions for unexplored perturbation space. In this study, we propose a sequential approach to the design of Perturb-seq experiments that uses the model to strategically select the most informative perturbations at each step, for follow-up experiments. This enables a significantly more efficient exploration of the perturbation space, while predicting the effect of the rest of the perturbations with high-fidelity. We conduct a preliminary data analysis on a large-scale Perturb-seq experiment, which reveals that our setting is severely restricted by the number of examples and rounds, falling into a non-conventional active learning regime called ''active learning under budget''. Motivated by this insight, we develop IterPert that exploits rich and multi-modal prior knowledge in order to efficiently guide the selection of perturbations. Making use of prior knowledge for this task is novel, and crucial for our setting of active learning under budget. We validate our method using in-silico benchmarking of active learning, constructed from a large-scale CRISPRi Perturb-seq data set. Our benchmarking reveals that IterPert outperforms contemporary active learning strategies, and delivering comparable accuracy with only a third of the amount of perturbations profiled. All in all, these results demonstrate the potential of sequentially designing perturbation screens.



## Installation

Use the API:

```bash
conda create --name iterpert_env python=3.8
conda activate iterpert_env
conda install pyg -c pyg
pip install iterpert
```

Use the raw source code:

```bash
conda create --name iterpert_env python=3.8
conda activate iterpert_env
conda install pyg -c pyg
git clone https://github.com/Genentech/iterative-perturb-seq.git
cd iterative-perturb-seq
pip install -r requirements.txt
```

## API interface

First, initialize the `IterPert` module:

```python
from iterpert.iterpert import IterPert
interface = IterPert(weight_bias_track = True, 
                     exp_name = strategy,
                     device = 'cuda:0', 
                     seed = 1)
```

The arguments are:

- `weight_bias_track`: True/False, if use weights and bias tracking
- `device`: cuda device
- `proj_name`: weights and bias project name
- `exp_name`: weights and bias experiment name
- `seed`: random seed in data split
- `run`: random seed in training run

Then, initialize the data:

```python
path = 'YOUR PATH'
interface.initialize_data(path = path,
                          dataset_name='replogle_k562_essential_1000hvg',
                          batch_size = 256)
```

The arguments are:

- `path`: path to save the data
- `dataset_name`: name of the dataset
- `batch_size`: number of cells in a batch
- `test_fraction`: fraction of the hold out test set

Then, initialize the GEARS model:

```python
interface.initialize_model(epochs = 20, hidden_size = 64)
```

The arguments are:

- `epochs`: the number of training epochs
- `hidden_size`: the number of hidden size of the model
- `retrain`: True/False, whether to retrain the model in each round


Then, initialize the active learning strategy:

You can also choose from baselines `Random`, `BALD`, `BatchBALD`, `BAIT`, `ACS-FW`, `Core-Set`, `BADGE`, `LCMD` or specify our method `IterPert`

```python
interface.initialize_active_learning_strategy(strategy = 'IterPert')
```

Lastly, kick off the training:

```python
interface.start(n_init_labeled = 100, n_round = 5, n_query = 100)
```

The arguments are:

- `n_init_labeled`: the number of initialized number of samples
- `n_round`: the number of rounds
- `n_query`: the number of queries per round



## Demo

We provide  tutorials to get started with iterative perturb-seq:

| Name  | Description                                             |
|-------|---------------------------------------------------------|
| [Data Tutorial](demo/data_tutorial.ipynb)   | Introduce the data loader and how to use your own data |
| [Training Tutorial](demo/train_tutorial.ipynb)   | A demo on training iterpert |
| [Knowledge Kernel Tutorial](demo/knowledge_kernels_process.ipynb)   | A tutorial on creating knowledge kernel for your own data |


## Reproduce experiments

Please refer to `reproduce_repo` directory to reproduce each experiment. Notably, the `README.md` contains sh files to generate all experiments. `figX.ipynb` is the notebook that produces the figures. 