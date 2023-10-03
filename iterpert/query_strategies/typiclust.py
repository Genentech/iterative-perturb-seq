import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import NearestNeighbors

import math
from scipy.optimize import nnls


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_

def get_nn(features, num_neighbors):
    # calculates nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='auto', metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]

def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance

def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality

def normalize_kernel(K, mode = 'diag'):
    if mode == 'cov':
        sqrt_diag_product = np.sqrt(np.outer(np.diagonal(K), np.diagonal(K)))
        return K / sqrt_diag_product
    elif mode == 'max':
        return K/np.max(K)
    elif mode == 'diag':
        lamb = 1/math.sqrt(np.mean(np.diag(K)))
        return lamb**2*K

def get_index(A, B):
    index_dict_A = {val: i for i, val in enumerate(A)}
    return np.array([index_dict_A[b] for b in B])

class TypiClust(Strategy):
    def __init__(self, dataset, net, base_kernel, use_prior_only = False, 
                 integrate_mode = 'mean', normalize_mode = 'diag', 
                 prior_kernel_list = None, prior_kernel_pert_list = None,
                 train_gold = None, train_feat = None, prior_feat_list = None, 
                 use_kernel_for_kmeans = False, add_ctrl = False, mode = 'kmeans', 
                 dim_reduce = 'NA', num_dim = 100, 
                 normalize_method = 'NA', gene_hvg_idx = None):
        super(TypiClust, self).__init__(dataset, net)
        self.base_kernel = base_kernel
        self.use_prior_only = use_prior_only
        self.integrate_mode = integrate_mode
        self.normalize_mode = normalize_mode
        self.prior_kernel_list = prior_kernel_list
        self.prior_kernel_pert_list = prior_kernel_pert_list
        self.train_gold = train_gold
        self.prior_feat_list = prior_feat_list
        self.train_feat = train_feat
        self.use_kernel_for_kmeans = use_kernel_for_kmeans
        self.add_ctrl = add_ctrl
        self.mode = mode
        self.gene_hvg_idx = gene_hvg_idx

    def query(self, n, save_kernel = False, save_name = None, valid_perts = None, round = None):
        strategy = self
        
        if self.base_kernel == 'linear':
            raise ValueError

        def agg_fct(x):
            return np.mean(np.vstack(x.values), axis = 0)

        #if self.base_kernel in ['before_gene_specific_layer1', 'before_cross_gene', 'cross_gene_embed', 'cross_gene_out', 'diff_effect']:
        labeled_idxs, train_data = strategy.dataset.get_train_data(get_distinct_perts = True)
        res = strategy.get_latent_emb(train_data, self.base_kernel)
        print('Feature size is: ' + str(res['latent_feat'].shape[1]))
        pert2pred = dict(pd.DataFrame(zip(res['pert_cat'], res['latent_feat'])).groupby(0)[1].agg(agg_fct))
        #else:
        #    unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        #    res, out = strategy.eval(unlabeled_data)
        #    pert2pred = dict(pd.DataFrame(zip(res['pert_cat'], res['pred'])).groupby(0)[1].agg(agg_fct))

        pert_list = np.array(list(pert2pred.keys())) ## set of unlabeled pool pert list, the order of embedd
        embeddings = np.stack([pert2pred[i] for i in pert_list])

        ### add ctrl cell embedding
        if self.add_ctrl and (self.base_kernel == 'diff_effect'):
            print('adding the control mean to diff effect ')
            print(embeddings.shape)
            embeddings += strategy.dataset.pert_data.ctrl_mean
            print(embeddings.shape)

        if self.gene_hvg_idx is not None:
            print('Using top ' + str(self.gene_hvg_idx) + ' HVGs!')
            embeddings = embeddings[:, self.gene_hvg_idx]


        if self.base_kernel in ['before_gene_specific_layer1', 'before_cross_gene', 'cross_gene_embed', 'cross_gene_out', 'diff_effect']:
            if res['latent_feat'].shape[1] > 10000:
                from sklearn.decomposition import PCA
                emb_size = min(embeddings.shape[0], 100)
                pca = PCA(n_components=emb_size)
                print('Before pca, dimension: ' + str(embeddings.shape[1]))
                embeddings = pca.fit_transform(embeddings)
                print('After pca, dimension: ' + str(embeddings.shape[1]))
                print('PCA explained ratio: ' + str(sum(pca.explained_variance_ratio_)))


        if self.prior_kernel_list is not None:
            ### use prior
            match_list = [i.split('+')[0] for i in pert_list]
            reindex = get_index(self.prior_kernel_pert_list, match_list)
            if self.use_kernel_for_kmeans:
                base_k = normalize_kernel(embeddings.dot(embeddings.T), mode = self.normalize_mode)

                ### looking at kernel similarity across all nodes
                prior_kernel_list = [k[reindex][:, reindex].squeeze() for k in self.prior_kernel_list]
                train_gold = self.train_gold[reindex][:, reindex].squeeze()
                print('normalizing prior kernel using ' + str(self.normalize_mode))
                prior_kernel_list = [normalize_kernel(k, mode = self.normalize_mode) for k in prior_kernel_list]

                if self.integrate_mode == 'mean':
                    print('using mean to integrate across kernels')
                    if not self.use_prior_only:
                        k_agg = base_k
                        kernel_num = 1
                    else:
                        k_agg = np.zeros_like(prior_kernel_list[0])
                        kernel_num = 0

                    for k in prior_kernel_list:
                        k_agg += k
                    k_agg /= (len(prior_kernel_list) + kernel_num)

                elif self.integrate_mode == 'coeff':
                    raise ValueError('Not implemented, aborted...')

                    print('using inferred coeff to integrate across kernels')

                    if not self.use_prior_only:
                        kernel_all = prior_kernel_list + [base_k]
                    else:
                        kernel_all = prior_kernel_list
                    
                    kernel_sub = [k[-len(self.features['train']):,-len(self.features['train']):] for k in kernel_all]
                    coeff = compute_coefficients(self.train_gold, kernel_sub)
                    k_agg = np.sum([kernel_all[idx] * i for idx, i in enumerate(coeff)], axis = 0)
                print('Embeddings-before shape: ' + str(embeddings.shape)) 
                embeddings = k_agg
                print('Embeddings-after shape: ' + str(embeddings.shape)) 

            else:
                prior_feat_list_reindex = [f[reindex].squeeze() for f in self.prior_feat_list]
                if not self.use_prior_only:
                    # combine with 
                    prior_feat_list_reindex.append(embeddings)
                else:
                    print('just using the prior...')
                
                embeddings = np.hstack(prior_feat_list_reindex)
                print('final feature has dimension:' + str(embeddings.shape))
                if embeddings.shape[1] > 10000:
                    from sklearn.decomposition import PCA
                    emb_size = min(embeddings.shape[0], 100)
                    pca = PCA(n_components=emb_size)
                    print('Before pca, dimension: ' + str(embeddings.shape[1]))
                    embeddings = pca.fit_transform(embeddings)
                    print('After pca, dimension: ' + str(embeddings.shape[1]))
                    print('PCA explained ratio: ' + str(sum(pca.explained_variance_ratio_)))

        pool_list = np.where(np.isin(pert_list, strategy.dataset.pert_train[~labeled_idxs]))[0]
        labeled_list = np.where(np.isin(pert_list, strategy.dataset.pert_train[labeled_idxs]))[0]

        num_clusters = sum(labeled_idxs) + n
        print(f'Clustering into {num_clusters} clustering...')
        clusters = kmeans(embeddings, num_clusters=num_clusters)
        K_NN = 20
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(clusters, return_counts=True)

        id2size = dict(zip(cluster_ids, cluster_sizes))
        cluster_sizes = np.array([id2size[i] if i in id2size else 0 for i in range(num_clusters)])
        cluster_ids = list(range(num_clusters))
        cluster_labeled_counts = np.bincount(clusters[labeled_list], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        #clusters_df = clusters_df[clusters_df.cluster_size > MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        clusters[labeled_list] = -1
        selected = []

        flag = True
        i = 0
        #for i in range(n):
        while flag:
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (clusters == cluster).nonzero()[0]
            if len(indices) != 0:
                rel_feats = embeddings[indices]
                # in case we have too small cluster, calculate density among half of the cluster
                typicality = calculate_typicality(rel_feats, min(K_NN, len(indices) // 2))
                idx = indices[typicality.argmax()]
                selected.append(idx)
                clusters[idx] = -1
            
            if len(selected) == n:
                flag = False
            else:
                i += 1

        selected = np.array(selected)
        assert len(selected) == n, 'added a different number of samples'
        assert len(np.intersect1d(selected, labeled_list)) == 0, 'should be new samples'
        p_list = pert_list[selected]
        unc_index = np.where(np.in1d(self.dataset.pert_train, p_list))[0]
        return unc_index