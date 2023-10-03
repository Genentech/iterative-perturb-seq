# Sequential Optimal Experimental Design of Perturbation Screens

This repository hosts the code base for the paper

**Sequential Optimal Experimental Design of Perturbation Screens Guided by Multimodal Priors**\
Kexin Huang, Romain Lopez, Jan-Christian Hütter, Takamasa Kudo, Antonio Rios, Aviv Regev


### Overview

<p align="center"><img src="https://raw.githubusercontent.com/Genentech/iterative-perturb-seq/master/img/illustration.png" alt="logo" width="800px" /></p>


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

# API interface

```python

from iterpert.iterpert import IterPert
strategy = 'IterPert' # choose from 'Random', 'BALD', 'BatchBALD', 'BAIT', 'ACS-FW', 'Core-Set', 'BADGE', 'LCMD', 'IterPert'
interface = IterPert(weight_bias_track = True, 
                     exp_name = strategy,
                     device = 'cuda:0', 
                     seed = 1)

path = 'YOUR PATH'
interface.initialize_data(path = path,
                          dataset_name='replogle_k562_essential_1000hvg',
                          batch_size = 256)

interface.initialize_model(epochs = 20, hidden_size = 64)
interface.initialize_active_learning_strategy(strategy = strategy)

interface.start(n_init_labeled = 100, n_round = 5, n_query = 100)

```

# Reproduce experiments

Please refer to `reproduce_repo` directory to reproduce each experiment. Notably, the `README.md` contains sh files to generate all experiments. `figX.ipynb` is the notebook that produces the figures.