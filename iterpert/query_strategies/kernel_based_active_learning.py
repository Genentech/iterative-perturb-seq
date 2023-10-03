import numpy as np
from .strategy import Strategy
from tqdm import tqdm
import pandas as pd
from ..bmdal.feature_data import TensorFeatureData
from ..bmdal.algorithms import select_batch, BatchSelectorImpl
import torch
import pickle

def get_index(A, B):
    index_dict_A = {val: i for i, val in enumerate(A)}
    return np.array([index_dict_A[b] for b in B])


class kernel_based_active_learning(Strategy):
    def __init__(self, dataset, net, selection_method, base_kernel, kernel_transforms, device, 
                 sel_with_train = False, reduce_latent_feat_dim_via_pca = False,
                 use_prior_only = False, integrate_mode = 'mean', normalize_mode = 'diag', 
                 prior_kernel_list = None, prior_kernel_pert_list = None, train_gold = None, 
                add_ctrl = False, 
                 prior_feat_list = None, gene_hvg_idx = None, lamb = None):
        super(kernel_based_active_learning, self).__init__(dataset, net)
        self.selection_method = selection_method
        self.base_kernel = base_kernel
        self.kernel_transforms = kernel_transforms
        self.sel_with_train = sel_with_train
        self.device = device
        self.reduce_latent_feat_dim_via_pca = reduce_latent_feat_dim_via_pca
        self.use_prior_only = use_prior_only
        self.integrate_mode = integrate_mode
        self.normalize_mode = normalize_mode
        self.prior_kernel_list = prior_kernel_list
        self.prior_kernel_pert_list = prior_kernel_pert_list
        self.train_gold = train_gold
        self.add_ctrl = add_ctrl
        self.prior_feat_list = prior_feat_list
        self.gene_hvg_idx = gene_hvg_idx
        self.lamb = lamb
    def query(self, n, save_kernel = False, save_name = None, valid_perts = None, round = None):
        strategy = self
        
        def agg_fct(x):
            return np.mean(np.vstack(x.values), axis = 0)
        base_kernel_list = ['before_gene_specific_layer1', 'before_cross_gene', 'cross_gene_embed', 'cross_gene_out', 'diff_effect']
        
        if self.base_kernel == 'linear':
            raise ValueError
        
        #if self.base_kernel in base_kernel_list:
            #labeled_idxs, train_data = strategy.dataset.get_train_data(get_distinct_perts = True)
        labeled_idxs, train_data = strategy.dataset.get_train_data(get_distinct_perts = True)
        res = strategy.get_latent_emb(train_data, self.base_kernel)
        print('Feature size is: ' + str(res['latent_feat'].shape[1]))
        pert2pred = dict(pd.DataFrame(zip(res['pert_cat'], res['latent_feat'])).groupby(0)[1].agg(agg_fct))
        base_kernel = 'linear'
        
        if res['latent_feat'].shape[1] > 10000:
            use_reduct = True
        else:
            use_reduct = False
        #else:
        #    labeled_idxs, train_data = strategy.dataset.get_train_data()
        #    res, out = strategy.eval(train_data)
        #    pert2pred = dict(pd.DataFrame(zip(res['pert_cat'], res['pred'])).groupby(0)[1].agg(agg_fct))
        #    base_kernel = self.base_kernel
        #    use_reduct = False
        pert_list = np.array(list(pert2pred.keys()))
        embeddings = np.stack([pert2pred[i] for i in pert_list])
        

        ### add ctrl cell embedding
        if self.add_ctrl and (self.base_kernel == 'diff_effect'):
            print('adding the control mean to diff effect ')
            embeddings += strategy.dataset.pert_data.ctrl_mean

        if self.gene_hvg_idx is not None:
            print('Using top ' + str(self.gene_hvg_idx) + ' HVGs!')
            embeddings = embeddings[:, self.gene_hvg_idx]

        if self.reduce_latent_feat_dim_via_pca or use_reduct:
            from sklearn.decomposition import PCA
            emb_size = min(embeddings.shape[0], 100)
            pca = PCA(n_components=emb_size)
            print('Before pca, dimension: ' + str(embeddings.shape[1]))
            embeddings = pca.fit_transform(embeddings)
            print('After pca, dimension: ' + str(embeddings.shape[1]))
            print('PCA explained ratio: ' + str(sum(pca.explained_variance_ratio_)))
        
        x_pool = torch.tensor(embeddings[np.where(np.isin(pert_list, strategy.dataset.pert_train[~labeled_idxs]))[0]]).to(self.device)
        x_train = torch.tensor(embeddings[np.where(np.isin(pert_list, strategy.dataset.pert_train[labeled_idxs]))[0]]).to(self.device)

        train_data = TensorFeatureData(x_train)
        pool_data = TensorFeatureData(x_pool)
        
        if save_kernel:
            print('Computing raw gradient kernel...')
            bs = BatchSelectorImpl([self.net], {'train': train_data, 'pool': pool_data}, 0)
            out = bs.select(selection_method='random', sel_with_train=True,
                        base_kernel='linear', kernel_transforms=[],
                        batch_size=n)
            save_matrix = {
                'pool_pool': bs.features['pool'].get_kernel_matrix(bs.features['pool']),
                'pool_train': bs.features['pool'].get_kernel_matrix(bs.features['train']),
                'train_train': bs.features['train'].get_kernel_matrix(bs.features['train']),
                'pool_list': pert_list[np.where(np.isin(pert_list, strategy.dataset.pert_train[~labeled_idxs]))[0]],
                'train_list': pert_list[np.where(np.isin(pert_list, strategy.dataset.pert_train[labeled_idxs]))[0]],
                'pool_feat': x_pool.detach().cpu().numpy(),
                'train_feat': x_train.detach().cpu().numpy()
            }
            with open(save_name + '_kernel.pkl', 'wb') as f:
                pickle.dump(save_matrix, f)
        

        if self.prior_kernel_list:
            print('Using prior kernels...')
            
            pool_list = pert_list[np.where(np.isin(pert_list, strategy.dataset.pert_train[~labeled_idxs]))[0]]
            train_list = pert_list[np.where(np.isin(pert_list, strategy.dataset.pert_train[labeled_idxs]))[0]]
            pool_list = [i.split('+')[0] for i in pool_list]
            train_list = [i.split('+')[0] for i in train_list]

            print('length of pool list: ' + str(len(pool_list)))
            print('length of train list: ' + str(len(train_list)))

            reindex = get_index(self.prior_kernel_pert_list, pool_list + train_list)
            reindex_train = get_index(self.prior_kernel_pert_list, train_list)
            
            if self.integrate_mode == 'learn':
                prior_kernel_list_reindex = [f[reindex].squeeze() for f in self.prior_feat_list]
                train_gold = self.train_gold[reindex][:, reindex].squeeze()
            else:
                prior_kernel_list_reindex = [k[reindex][:, reindex].squeeze() for k in self.prior_kernel_list]
                train_gold = self.train_gold[reindex_train][:, reindex_train].squeeze() ## just the part where gold label is available

            if valid_perts is not None:
                print('Using validation perts: ')
                print(valid_perts)
                valid_perts = [i.split('+')[0] for i in valid_perts]
                reindex = get_index(self.prior_kernel_pert_list, valid_perts)
                train_gold = self.train_gold[reindex][:, reindex].squeeze()
                print(train_gold.shape)
                valid_perts = get_index(pool_list + train_list, valid_perts)
                print(valid_perts)
                
            bs = BatchSelectorImpl([self.net], {'train': train_data, 'pool': pool_data}, 0, train_gold = train_gold, prior_kernel_list = prior_kernel_list_reindex, 
                        use_prior_only = self.use_prior_only, integrate_mode = self.integrate_mode, normalize_mode = self.normalize_mode, valid_perts = valid_perts, 
                        lamb = self.lamb, round = round)
            new_idxs, _ = bs.select(selection_method=self.selection_method + '_prior', sel_with_train=self.sel_with_train,
                        base_kernel=base_kernel, kernel_transforms=self.kernel_transforms,
                        batch_size=n)
            

        else:
            new_idxs, _ = select_batch(batch_size=n, models=[self.net], 
                            data={'train': train_data, 'pool': pool_data}, y_train=0,
                            selection_method=self.selection_method, sel_with_train=self.sel_with_train,
                            base_kernel=base_kernel, kernel_transforms=self.kernel_transforms, 
                            lamb = self.lamb, round = round)
        p_list = pert_list[np.where(np.isin(pert_list, strategy.dataset.pert_train[~labeled_idxs]))[0]][new_idxs.detach().cpu().numpy()]
        unc_index = np.where(np.in1d(self.dataset.pert_train, np.array(p_list)))[0]
        return unc_index