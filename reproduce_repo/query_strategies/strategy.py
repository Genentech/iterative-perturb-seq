import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, batch_exp = None):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data(batch_exp)
        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds['pred']

    def predict_prob(self, data):
        res = self.net.predict_prob(data)
        return res

    def predict_prob_dropout(self, data, n_drop=10):
        raise NotImplementedError

    def predict_prob_dropout_split(self, data, n_drop=10):
        raise NotImplementedError
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings
    
    def eval(self, data):
        res_all = self.net.predict(data, detail_eval = True)
        return res_all
    
    def get_latent_emb(self, data, latent_type):
        return self.net.get_latent_emb(data, latent_type)

