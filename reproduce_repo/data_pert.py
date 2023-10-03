import numpy as np
import torch
from gears import PertData
import pickle

class Data:
    def __init__(self, path, data_name, batch_size, test_fraction = 0.1, seed = 1, custom_test = None):
        
        
        pert_data = PertData(path) # specific saved folder
        if data_name == 'adamson':
            pert_data.load(data_path = '/home/huangk28/projects/active_pert/tutorials/data/adamson')
        else:
            pert_data.load(data_path = path + data_name)

        if custom_test:
            print('Using custom test set!')
            split = 'custom_test'
            test_perts = pickle.load(open(custom_test, "rb"))['test']
        else:
            split = 'active'
            test_perts = None

        pert_data.prepare_split(split = split,
                                test_perts = test_perts,
                                combo_single_split_test_set_fraction = test_fraction, 
                                seed = seed)

        self.batch_size = batch_size

        self.pert_data = pert_data
        self.pert_train = np.array(pert_data.set2conditions['train'])
        self.pert_test = np.array(pert_data.set2conditions['test'])
        
        self.n_pool = len(self.pert_train)
        self.n_test = len(self.pert_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.seed(42)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
        return tmp_idxs[:num]
    
    def get_labeled_data(self, batch_exp = None):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        print('Number of labeled perts at this round: ' + str(len(labeled_idxs)))
        ## get validation set for training
        np.random.shuffle(labeled_idxs)        
        labeled_idxs_train = labeled_idxs[:int(0.9*len(labeled_idxs))]
        labeled_idxs_valid = labeled_idxs[int(0.9*len(labeled_idxs)):]

        

        return labeled_idxs, {'train_loader': self.pert_data.get_dataloader_from_pert_list(self.pert_train[labeled_idxs_train], 
                                                                                           self.batch_size, shuffle = True, 
                                                                                           eval_mode = False, batch_exp = batch_exp),
                              'val_loader': self.pert_data.get_dataloader_from_pert_list(self.pert_train[labeled_idxs_valid], 
                                                                                         self.batch_size, shuffle = False, 
                                                                                         eval_mode = True, batch_exp = batch_exp),
                              'valid_pert': self.pert_train[labeled_idxs_valid],
                              'train_pert': self.pert_train[labeled_idxs_train],
                              'train_valid_pert': self.pert_train[labeled_idxs]
                            }
    
    def get_unlabeled_data(self, get_distinct_perts = False):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.pert_data.get_dataloader_from_pert_list(self.pert_train[unlabeled_idxs], self.batch_size, shuffle = False, eval_mode = True, get_distinct_perts = get_distinct_perts)
    
    def get_train_data(self, get_distinct_perts = False, batch_exp = None):
        return self.labeled_idxs.copy(), self.pert_data.get_dataloader_from_pert_list(self.pert_train, self.batch_size, 
                                                                                      shuffle = False, eval_mode = True, 
                                                                                      get_distinct_perts = get_distinct_perts,
                                                                                      batch_exp = batch_exp)

    def get_test_data(self, get_distinct_perts = False):
        return self.pert_data.get_dataloader_from_pert_list(self.pert_test, self.batch_size, shuffle = False, eval_mode = True, get_distinct_perts = get_distinct_perts)
    
    def cal_test_acc(self, preds):
        raise NotImplementedError