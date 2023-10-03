import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
import pandas as pd

import math
from scipy.optimize import nnls

def compute_coefficients(A, matrices):
    # Flatten A into a column vector
    a = A.flatten()

    # Flatten all matrices in the list and add them as columns to X
    X = np.column_stack([matrix.flatten() for matrix in matrices])

    # Compute the QR decomposition of X
    Q, R = np.linalg.qr(X)

    # Solve for the coefficients
    coefficients = np.linalg.solve(R, Q.T @ a)

    return coefficients

def non_negative_compute_coefficients(A, matrices):
    # Flatten A into a column vector
    a = A.flatten()

    # Flatten all matrices in the list and add them as columns to X
    X = np.column_stack([matrix.flatten() for matrix in matrices])

    # Compute the QR decomposition of X
    Q, R = np.linalg.qr(X)

    # Solve for the coefficients
    #coefficients = np.linalg.solve(R, Q.T @ a)
    coefficients, rnorm = nnls(R, Q.T @ a)
    return coefficients

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

class KMeansSampling(Strategy):
    def __init__(self, dataset, net, base_kernel, use_prior_only = False, 
                 integrate_mode = 'mean', normalize_mode = 'diag', 
                 prior_kernel_list = None, prior_kernel_pert_list = None,
                 train_gold = None, train_feat = None, prior_feat_list = None, 
                 use_kernel_for_kmeans = False, add_ctrl = False, 
                 mode = 'kmeans', dim_reduce = 'NA', num_dim = 100, 
                 normalize_method = 'NA', gene_hvg_idx = None):
        super(KMeansSampling, self).__init__(dataset, net)
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

        self.dim_reduce = dim_reduce
        self.num_dim = num_dim
        self.normalize_method = normalize_method
        self.gene_hvg_idx = gene_hvg_idx

    def query(self, n, save_kernel = False, save_name = None, valid_perts = None, round = None):
        strategy = self
        #if self.base_kernel == 'linear':
        #    raise ValueError
        
        print('Using ' + self.base_kernel + ' base kernel...')

        def agg_fct(x):
            return np.mean(np.vstack(x.values), axis = 0)
        
        if self.base_kernel in ['before_gene_specific_layer1', 'before_cross_gene', 'cross_gene_embed', 'cross_gene_out', 'diff_effect']:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(get_distinct_perts = True)
            res = strategy.get_latent_emb(unlabeled_data, self.base_kernel)
            print('Feature size is: ' + str(res['latent_feat'].shape[1]))
            pert2pred = dict(pd.DataFrame(zip(res['pert_cat'], res['latent_feat'])).groupby(0)[1].agg(agg_fct))
        else:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            res, out = strategy.eval(unlabeled_data)
            pert2pred = dict(pd.DataFrame(zip(res['pert_cat'], res['pred'])).groupby(0)[1].agg(agg_fct))

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


        if self.prior_kernel_list is not None:
            ### use prior
            match_list = [i.split('+')[0] for i in pert_list]
            reindex = get_index(self.prior_kernel_pert_list, match_list)
            if self.use_kernel_for_kmeans:
                base_k = embeddings.dot(embeddings.T)
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
        

        # embedding post-processing...
        dim_reduce = self.dim_reduce # None / PCA / UMAP
        num_dim = self.num_dim # 50 / 20 / 5
        normalize_method = self.normalize_method # None / min-max / feature-scale / centering / unit-length

        print('dim_reduce mode: ' + dim_reduce)
        print('num_dim: ' + str(num_dim))
        print('normalize_method: ' + normalize_method)


        if dim_reduce == 'PCA':
            from sklearn.decomposition import PCA
            emb_size = min(embeddings.shape[0], num_dim)
            pca = PCA(n_components=emb_size)
            print('Before pca, dimension: ' + str(embeddings.shape[1]))
            embeddings = pca.fit_transform(embeddings)
            print('After pca, dimension: ' + str(embeddings.shape[1]))
            print('PCA explained ratio: ' + str(sum(pca.explained_variance_ratio_)))
        elif dim_reduce == 'UMAP':
            import umap
            reducer = umap.UMAP(n_components = num_dim, random_state=42)
            embeddings = reducer.fit_transform(embeddings)


        from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
        print('Normalizing base kernel with ' + normalize_method)
        if normalize_method != 'NA':
            if normalize_method == 'min-max':
                normalizer = MinMaxScaler()
            elif normalize_method == 'feature-scale':
                normalizer = StandardScaler()
            elif normalize_method == 'centering':
                normalizer = StandardScaler(with_std=False)
            elif normalize_method == 'unit-length':
                normalizer = Normalizer(norm='l2')
            embeddings = normalizer.fit_transform(embeddings)


        if self.mode == 'kmeans':
            print(embeddings.shape)
            cluster_learner = KMeans(n_clusters=n)
            cluster_learner.fit(embeddings)
            print('Clustering learning finished...')
            cluster_idxs = cluster_learner.predict(embeddings)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (embeddings - centers)**2
            dis = dis.sum(axis=1)

            q_idxs = []
            num_empty_clusters = 0
            for i in range(n):
                if len(dis[cluster_idxs==i]) == 0:
                    print('empty cluster ' + str(i))
                    num_empty_clusters += 1
                else:
                    idx = np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()]
                    q_idxs.append(idx)
            if num_empty_clusters == 0:
                q_idxs = np.array(q_idxs)
            else:
                print('Adding points to ' + str(num_empty_clusters) + ' empty clusters...')
                pool = np.setdiff1d(np.arange(embeddings.shape[0]), q_idxs)
                np.random.seed(42)
                add_set = np.random.choice(pool, num_empty_clusters, replace = False)
                q_idxs = np.concatenate((q_idxs, add_set))
                assert len(q_idxs) == n
            #q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            unc_index = np.where(np.in1d(self.dataset.pert_train, pert_list[q_idxs]))[0]
        
        elif self.mode == 'Agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            cluster_learner = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
            cluster_idxs = cluster_learner.fit_predict(embeddings)
            print('Clustering learning finished...')
            centers = np.array([embeddings[cluster_idxs == i].mean(axis=0) for i in range(cluster_learner.n_clusters)])
            centers = centers[cluster_idxs]
            dis = (embeddings - centers)**2
            dis = dis.sum(axis=1)
            q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            unc_index = np.where(np.in1d(self.dataset.pert_train, pert_list[q_idxs]))[0]

        elif self.mode == 'Spectral':
            from sklearn.cluster import SpectralClustering
            cluster = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', assign_labels='kmeans')
            cluster_idxs = cluster.fit_predict(embeddings)

            # Calculate cluster centers
            centers = np.array([embeddings[cluster_idxs == i].mean(axis=0) for i in range(cluster.n_clusters)])
            centers = centers[cluster_idxs]
            dis = (embeddings - centers)**2
            dis = dis.sum(axis=1)
            q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            unc_index = np.where(np.in1d(self.dataset.pert_train, pert_list[q_idxs]))[0]


        elif self.mode == 'gmm_max':
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=n, random_state=0).fit(embeddings)
            mixture_label = gm.predict(embeddings)
            likelihood = gm.score_samples(embeddings)

            q_idxs = []
            for label in np.unique(mixture_label):
                filter_ = np.where(mixture_label == label)
                idx = np.argmax(likelihood[filter_])
                q_idxs.append(filter_[0][idx])

            q_idxs = np.array(q_idxs)
            unc_index = np.where(np.in1d(self.dataset.pert_train, pert_list[q_idxs]))[0]

        elif self.mode == 'gmm_min':
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=n, random_state=0).fit(embeddings)
            mixture_label = gm.predict(embeddings)
            likelihood = gm.score_samples(embeddings)

            q_idxs = []
            for label in np.unique(mixture_label):
                filter_ = np.where(mixture_label == label)
                idx = np.argmin(likelihood[filter_])
                q_idxs.append(filter_[0][idx])

            q_idxs = np.array(q_idxs)
            unc_index = np.where(np.in1d(self.dataset.pert_train, pert_list[q_idxs]))[0]

        elif self.mode == 'kmeans++':
            from sklearn.cluster import kmeans_plusplus
            centers, indices = kmeans_plusplus(embeddings, 
                                            n_clusters=n, 
                                            random_state=0,
                                            n_local_trials = 1)
            p_list = pert_list[indices]
            
            unc_index = np.where(np.in1d(self.dataset.pert_train, np.array(p_list)))[0]
        return unc_index