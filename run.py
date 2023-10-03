import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default="IterPert")
args = parser.parse_args()

from iterpert.iterpert import IterPert
strategy = args.strategy
interface = IterPert(weight_bias_track = True, 
                     exp_name = strategy,
                     device = 'cuda', 
                     seed = 1)

path = './gears_data/'
interface.initialize_data(path = path,
                          dataset_name='replogle_k562_essential_1000hvg',
                          batch_size = 256)

interface.initialize_model(epochs = 20, hidden_size = 64)
interface.initialize_active_learning_strategy(strategy = strategy)

interface.start(n_init_labeled = 100, n_round = 5, n_query = 100)