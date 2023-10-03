import argparse
import numpy as np
import torch
import os
from utils import get_strategy
from data_pert import Data
from nets_pert import Net
from pprint import pprint
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--run', type=int, default=1, help="random run")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--test_fraction', type=float, default=0.1, help="test_fraction")
parser.add_argument('--hidden_size', type=int, default=64, help="hidden_size")
parser.add_argument('--uncertainty', default=True, action="store_false")
parser.add_argument('--uncertainty_reg', type=float, default=1, help="uncertainty_reg")
parser.add_argument('--direction_lambda', type=float, default=1, help="direction_lambda")
parser.add_argument('--epoch_per_cycle', type=int, default=5, help="epoch_per_cycle")
parser.add_argument('--simple_loss', default=False, action="store_true")

parser.add_argument('--retrain', default=False, action="store_true")
parser.add_argument('--wandb', default=False, action="store_true")
parser.add_argument('--wb_proj_name', type=str, default='active_gears', help="cuda device")
parser.add_argument('--wb_exp_name', type=str, default='gears', help="cuda device")
parser.add_argument('--model_name', type=str, default='GEARS', choices = ['GEARS', 'scGPT'])

parser.add_argument('--device', type=str, default='cuda:1', help="cuda device")
parser.add_argument('--n_init_labeled', type=int, default=10, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=10, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="adamson", choices=["adamson", 
                                                                            "replogle_k562_gw_1000hvg",
                                                                            "replogle_k562_essential_1000hvg",
                                                                            "replogle_k562_essential_1000hvg+pert_in_gene"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="kernel_based_active_learning", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KMeansUncertainty",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool", 
                             "kernel_based_active_learning", 
                             "MaxDist",
                             "TypiClust",
                             "EssentialSampling",
                             "EssentialWeighted"], help="query strategy")

#parser.add_argument('--selection_method', type=str, default="maxdet", choices=['random', 'maxdiag', 'maxdet', 'bait', 'fw', 'maxdist', 'kmeanspp', 'lcmd'])
parser.add_argument('--kernel_strategy', type=str, choices=['Random', 'BALD', 'BatchBALD', 
                                                            'BAIT', 'ACS-FW', 'Core-Set', 
                                                            'LCMD', 'BADGE', 'MAXDIST', 
                                                            'KMEANSPP', 'MAXDET', 'MAXDIAG', 
                                                            'BAIT', 'FW', 'D-OptimalDesign', 
                                                            'DIR'])
parser.add_argument('--base_kernel', type=str, default="linear", choices=['ll', 'grad', 'linear', 
                                                                          'nngp', 'ntk', 'laplace', 
                                                                          'before_gene_specific_layer1', 
                                                                          'before_cross_gene', 'cross_gene_embed',
                                                                          'cross_gene_out', 'diff_effect', 'linear_fix_ctrl'])
#parser.add_argument('--kernel_transforms', type=str, default="train", choices=["train", 'pool', 'scale', 'rp', 'acs-rf'])
#parser.add_argument('--not_sel_with_train', action = 'store_true', default=False)
#parser.add_argument('--sigma', type=int, default=0.1)
#parser.add_argument('--factor', type=int, default=None)
parser.add_argument('--reduce_latent_feat_dim_via_pca', action = 'store_true', default=False)
parser.add_argument('--custom_split', action = 'store_true', default=False)
parser.add_argument('--save_kernel', action = 'store_true', default=False)

parser.add_argument('--use_prior', action = 'store_true', default=False)
parser.add_argument('--use_prior_only', action = 'store_true', default=False)
parser.add_argument('--integrate_mode',  type=str, choices=['mean', 'coeff', 'learn', 'mean_new', 'product', 'max', 'alignment'], default='mean')
parser.add_argument('--normalize_mode',  type=str, choices=['diagonal', 'mean', 'max', 'trace', 'frobenius', 'row_sum', 'centering', 'ID', 'diag'], default='diag')
parser.add_argument('--use_single_prior', action = 'store_true', default=False)
parser.add_argument('--single_prior', type=str, choices=['kg_kernel', 'pops_kernel', 'rpe1_kernel', 
                                                         'esm_kernel', 'biogpt_kernel', 'node2vec_kernel', 
                                                         'ground_truth_delta', 'gears_kernel', 'ops_kernel', 
                                                         'ops_A549_kernel', 'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel'], default='ground_truth_delta')
parser.add_argument('--use_kernel_for_kmeans', action = 'store_true', default=False)
parser.add_argument('--ops_cell',  type=str, choices=['A549', 'HeLa'], default='A549')
parser.add_argument('--ops_plate',  type=str, default='A')

parser.add_argument('--normalize_kernel', action = 'store_true', default=False)
parser.add_argument('--normalize_method',  type=str, choices=['min-max', 'feature-scale', 
                                                              'unit-length', 'centering', 'NA'], default='NA')
parser.add_argument('--sample_cells_training', action = 'store_true', default=False)
parser.add_argument('--fix_evaluation', action = 'store_true', default=False)
parser.add_argument('--kernel_normalize_feat', action = 'store_true', default=False)

parser.add_argument('--cluster_mode',  type=str, choices=['kmeans', 'kmeans++', 
                                                          'gmm_max', 'gmm_min',
                                                          'typicust', 'Agglomerative', 'Spectral'], default='kmeans')

parser.add_argument('--num_dim', type=int, default=100, help="number of init labeled samples")
parser.add_argument('--dim_reduce', type=str, default='NA', help="number of init labeled samples")

parser.add_argument('--essential_assay',  type=str, choices=['hap1', 'kbm7', 'NA'], default='NA')
parser.add_argument('--hvg_kernel', action = 'store_true', default=False)
parser.add_argument('--hvg_num', type=int, default=100)
parser.add_argument('--valid_perts', action = 'store_true', default=False)

parser.add_argument('--lamb', type=float, default=2)
parser.add_argument('--batch_exp', action = 'store_true', default=False)

args = parser.parse_args()

if args.batch_exp:
    args.dataset_name += '_batch_exp'

if args.single_prior == 'ops_kernel':
    args.single_prior = 'ops_' + args.ops_cell + '_' + args.ops_plate + '_kernel'
    print('Using ' + args.single_prior)

if args.base_kernel == 'linear_fix_ctrl':
    print('using linear_fix_ctrl...')
    args.base_kernel = 'diff_effect'
    print(args.base_kernel)
    add_ctrl = True
else:
    add_ctrl = False

args.wb_exp_name = '_'.join([args.model_name, 
                             str(args.n_query), str(args.n_round), 
                             str(args.n_init_labeled), str(args.batch_size), 
                             str(args.seed)])

if args.retrain:
    args.wb_exp_name += '_rt'
else:
    args.wb_exp_name += '_nrt'

if args.epoch_per_cycle != 20:
    args.wb_exp_name += '_epo' + str(args.epoch_per_cycle)

if args.reduce_latent_feat_dim_via_pca:
    args.wb_exp_name += '_reduce_dim'

if args.simple_loss:
    args.wb_exp_name += '_simple_loss_v2'
    args.uncertainty = False

if args.dataset_name == 'replogle_k562_gw_1000hvg':
    args.wb_exp_name += '_gw'
elif args.dataset_name == 'replogle_k562_essential_1000hvg':
    args.wb_exp_name += '_ess_1k'
elif args.dataset_name == 'replogle_rpe1_essential_1000hvg':
    args.wb_exp_name += '_rpe1'

if args.normalize_kernel:
    args.wb_exp_name += '_' + args.normalize_method

if args.fix_evaluation:
    args.wb_exp_name += '_fix_eval'

if args.cluster_mode != 'kmeans':
    args.wb_exp_name += '_' + args.cluster_mode

if args.strategy_name == 'EssentialWeighted':
    args.wb_exp_name += '_' + args.essential_assay

if args.use_prior:
    args.wb_exp_name += '_prior'
    if args.use_prior_only:
        print('Using prior, without the base kernel from the model!')
        args.wb_exp_name += '_only'
    args.wb_exp_name += args.integrate_mode + '_'+ args.normalize_mode
    if args.use_single_prior:
        print('Just using one prior: ' + args.single_prior)
        args.wb_exp_name += '_single_' + args.single_prior
    if args.strategy_name == 'KMeansSampling':
        if args.use_kernel_for_kmeans:
            print('use kernel integration...')
            args.wb_exp_name += '_kernel'
    if args.kernel_normalize_feat:
        args.wb_exp_name += '_feat_norm'

if add_ctrl:
    args.wb_exp_name += '_add_ctrl'

if args.sample_cells_training:
    args.wb_exp_name += '_sct'

if args.custom_split:
    if args.dataset_name != 'replogle_k562_gw_1000hvg':
        raise ValueError
    args.wb_exp_name += '_test_ess'
    custom_test = '/home/huangk28/scratch/perturb_seq_data/gears_data/replogle_k562_essential_1000hvg+pert_in_gene/splits/replogle_k562_essential_1000hvg+pert_in_gene_active_1_0.75.pkl'
else:
    custom_test = None
    
if args.lamb != 2:
    args.wb_exp_name += '_lamb' + str(args.lamb)

if args.strategy_name == 'EssentialSampling':
    custom_test = '/home/huangk28/scratch/perturb_seq_data/gears_data/replogle_k562_essential_1000hvg+pert_in_gene/splits/replogle_k562_essential_1000hvg+pert_in_gene_active_1_0.75.pkl'
    essential_gene = pickle.load(open(custom_test, 'rb'))['test'] + pickle.load(open(custom_test, 'rb'))['train']

args.wb_exp_name += '_run' + str(args.run)


if args.batch_exp:
    args.wb_exp_name += '_batch_exp'

#args.wb_exp_name += '_test_shuffle'

print(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.run)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device(args.device if use_cuda else "cpu")
print(device)

path = '/home/huangk28/scratch/perturb_seq_data/gears_data/'
dataset = Data(path, args.dataset_name, args.batch_size, args.test_fraction, args.seed, custom_test)
n_features = dataset.pert_data.adata.X.shape[1]
test_data = dataset.get_test_data()
print('Using overall framework:' + args.strategy_name)
if args.strategy_name == 'kernel_based_active_learning':
    print('Using strategy:' + args.kernel_strategy)
    args.wb_exp_name += '_' + args.kernel_strategy
    if args.base_kernel != 'linear':
        args.wb_exp_name += '_' + args.base_kernel
    print(args.wb_exp_name)
    if args.kernel_strategy == 'Random':
        selection_method = 'random'
        kernel_transforms=[]
        sel_with_train = True
    elif args.kernel_strategy == 'BALD':
        selection_method='maxdiag'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'BatchBALD':
        selection_method='maxdet'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'BAIT':
        selection_method='bait'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'ACS-FW':
        selection_method = 'fw'
        #kernel_transforms=[('rp', [512]), ('acs-rf', [512, 0.1, None])]
        kernel_transforms=[('acs-rf', [512, 0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'Core-Set':
        selection_method = 'maxdist'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[]
        sel_with_train = True
    elif args.kernel_strategy == 'BADGE':
        selection_method = 'kmeanspp'
        #kernel_transforms=[('rp', [512]), ('acs-rf', [512, 0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'MAXDIST':
        selection_method = 'maxdist'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'KMEANSPP':
        selection_method = 'kmeanspp'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'MAXDET':
        selection_method = 'maxdet'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'MAXDIAG':
        selection_method = 'maxdiag'
        kernel_transforms=[]
        sel_with_train = False
    #elif args.kernel_strategy == 'BAIT':
    #    selection_method = 'bait'
    #    kernel_transforms=[]
    #    sel_with_train = False
    elif args.kernel_strategy == 'FW':
        selection_method = 'fw'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'LCMD':
        selection_method = 'lcmd'
        kernel_transforms=[('rp', [512])] 
        sel_with_train = True
    elif args.kernel_strategy == 'D-OptimalDesign':
        selection_method = 'maxdet'
        kernel_transforms=[]
        sel_with_train = True
    elif args.kernel_strategy == 'DIR':
        selection_method = 'dir'
        kernel_transforms=[]
        sel_with_train = True
else:
    if args.base_kernel != 'linear':
        args.wb_exp_name += '_' + args.base_kernel
    args.wb_exp_name += '_' + args.strategy_name


save_dir = '/home/huangk28/projects/active_pert/save_dir/' + args.wb_exp_name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
params = {
    'weight_bias_track': args.wandb,
    'wb_proj_name': args.wb_proj_name,
    'wb_exp_name': args.wb_exp_name,
    'hidden_size': args.hidden_size,
    'uncertainty' : args.uncertainty, 
    'uncertainty_reg' : args.uncertainty_reg,
    'direction_lambda' : args.direction_lambda,
    'device': args.device,
    'epoch_per_cycle': args.epoch_per_cycle,
    'retrain': args.retrain,
    'simple_loss': args.simple_loss
}

if args.num_dim != 100:
    args.wb_exp_name += '_'  + str(args.num_dim)
if args.dim_reduce != 'NA':
    args.wb_exp_name += '_'  + str(args.dim_reduce)
if args.normalize_method != 'NA':
    args.wb_exp_name += '_'  + str(args.normalize_method)


if args.hvg_kernel:
    adata = dataset.pert_data.adata
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
    gene_hvg_idx = np.array([geneid2idx[i] for i in adata.var.sort_values('dispersions_norm')[::-1][:args.hvg_num].index.values])
    args.wb_exp_name += '_hvg' + str(args.hvg_num)
else:
    gene_hvg_idx = None

if args.valid_perts:
    args.wb_exp_name += '_val_pert'

if args.wandb:
    import wandb
    wandb.login(host='https://genentech.wandb.io', key=os.environ.get('WANDB_API_KEY'))
    wandb.init(project=args.wb_proj_name, name=args.wb_exp_name) 
    wandb.config.update(params)

net = Net(params, device, dataset.pert_data, args.model_name, save_dir, args.fix_evaluation)                   # load network

if args.use_prior:
    #if args.dataset_name != 'replogle_k562_essential_1000hvg+pert_in_gene':
    #    raise ValueError('prior currently imnplemented for this dataset only...')

    import pickle

    def load_kernel(kernel_name):
        if args.dataset_name == 'replogle_k562_gw_1000hvg':
            kernel_path = '/home/huangk28/scratch/knowledge_kernels_gw/'
        elif args.dataset_name == 'replogle_k562_essential_1000hvg+pert_in_gene':
            kernel_path = '/home/huangk28/scratch/knowledge_kernels/'
        elif args.dataset_name == 'replogle_rpe1_essential_1000hvg':
            kernel_path = '/home/huangk28/scratch/knowledge_kernels_rpe1/'
        else:
            kernel_path = '/home/huangk28/scratch/knowledge_kernels_1k/'
        if not os.path.exists(kernel_path + kernel_name):
            raise ValueError('Kernel does not exist')
        with open(kernel_path + kernel_name + '/pert_list.pkl', 'rb') as f:
            pert_list = pickle.load(f)
        with open(kernel_path + kernel_name + '/kernel.pkl', 'rb') as f:
            kernel_npy = pickle.load(f)
        with open(kernel_path + kernel_name + '/feat.pkl', 'rb') as f:
            feat = pickle.load(f)
        return pert_list, kernel_npy, feat

    if args.use_single_prior:
        kernel_list = [args.single_prior]
    else:
        if args.dataset_name == 'replogle_k562_gw_1000hvg':
            kernel_list = ['pops_kernel', 'esm_kernel', 
                       'biogpt_kernel', 'node2vec_kernel', 'ops_A549_kernel',
                       'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel']
        elif args.dataset_name == 'replogle_rpe1_essential_1000hvg':
            kernel_list = ['pops_kernel', 'k562_kernel', 'esm_kernel', 
                       'biogpt_kernel', 'node2vec_kernel', 'ops_A549_kernel',
                       'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel']
        else:
            kernel_list = ['pops_kernel', 'rpe1_kernel', 'esm_kernel', 
                       'biogpt_kernel', 'node2vec_kernel', 'ops_A549_kernel',
                       'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel']
    prior_kernel_list, prior_feat_list = [],[] 
    
    for i in kernel_list:
        _, k, f = load_kernel(i)
        if args.kernel_normalize_feat:
            print('normalizing feature and then compute kernels!')
            from sklearn.preprocessing import StandardScaler    
            normalizer = StandardScaler()
            f = normalizer.fit_transform(f)
            k = np.dot(f, f.T)
            prior_kernel_list.append(k)
        else:
            prior_kernel_list.append(k)
        prior_feat_list.append(f)

    pert_list, true_gold, truth_feat = load_kernel('ground_truth_delta')
    pert_list = [i.split('+')[0] for i in pert_list]
    #print(pert_list)
    #print(prior_kernel_list)
    if args.strategy_name == 'kernel_based_active_learning':
        strategy = get_strategy(args.strategy_name)(dataset, net, selection_method, args.base_kernel, 
                                                    kernel_transforms, device, sel_with_train, 
                                                    args.reduce_latent_feat_dim_via_pca, use_prior_only = args.use_prior_only, 
                                                    integrate_mode = args.integrate_mode, normalize_mode = args.normalize_mode, 
                                                    prior_kernel_list = prior_kernel_list, prior_kernel_pert_list = pert_list, 
                                                    train_gold = true_gold, normalize_kernel = args.normalize_kernel,
                                                    normalize_method = args.normalize_method, add_ctrl = add_ctrl, 
                                                    prior_feat_list = prior_feat_list, gene_hvg_idx = gene_hvg_idx, lamb = args.lamb)
    elif args.strategy_name in ['KMeansSampling', 'MaxDist', 'TypiClust']:
        strategy = get_strategy(args.strategy_name)(dataset, net, args.base_kernel, use_prior_only = args.use_prior_only, 
                                                    integrate_mode = args.integrate_mode, normalize_mode = args.normalize_mode, 
                                                    prior_kernel_list = prior_kernel_list, prior_kernel_pert_list = pert_list, 
                                                    train_gold = true_gold, train_feat = truth_feat, prior_feat_list = prior_feat_list,
                                                    use_kernel_for_kmeans = args.use_kernel_for_kmeans, 
                                                    add_ctrl = add_ctrl, mode = args.cluster_mode, dim_reduce = args.dim_reduce, 
                                                    num_dim = args.num_dim, normalize_method = args.normalize_method, gene_hvg_idx = gene_hvg_idx)
else:
    if args.strategy_name == 'kernel_based_active_learning':
        strategy = get_strategy(args.strategy_name)(dataset, net, selection_method, args.base_kernel, 
                                                    kernel_transforms, device, sel_with_train, 
                                                    args.reduce_latent_feat_dim_via_pca,
                                                    normalize_kernel = args.normalize_kernel,
                                                    normalize_method = args.normalize_method,
                                                    add_ctrl = add_ctrl, gene_hvg_idx = gene_hvg_idx, lamb = args.lamb)
    elif args.strategy_name in ['KMeansSampling', 'KCenterGreedy', 'KMeansUncertainty', 'MaxDist', 'TypiClust']:
        strategy = get_strategy(args.strategy_name)(dataset, net, 
                                                    args.base_kernel, 
                                                    add_ctrl = add_ctrl,
                                                    mode = args.cluster_mode, 
                                                    dim_reduce = args.dim_reduce, 
                                                    num_dim = args.num_dim, 
                                                    normalize_method = args.normalize_method, 
                                                    gene_hvg_idx = gene_hvg_idx)  # load strategy
    else:
        strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
init_idx = dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
if args.batch_exp:
    genes_available_per_round = {}
    genes_available_per_round[0] = dataset.pert_train[init_idx]
    all_batch_idx = dataset.pert_data.all_batch_idx
    np.random.seed(args.seed)
    np.random.shuffle(all_batch_idx)
    batch_idx_round = {}
    rounds = args.n_round
    rounds += 1 # initialized round
    num_batches_per_run = int(len(all_batch_idx) / rounds)
    for round in range(rounds):
        batch_idx_round[round] = all_batch_idx[num_batches_per_run * round : num_batches_per_run * (round+1)]
    strategy.train({'batch_idx': batch_idx_round, 'genes_available_per_round': genes_available_per_round})  
else:
    strategy.train()

res, out = strategy.eval(test_data)
metrics = ['pearson_delta', 
            'frac_opposite_direction_top20_non_dropout',
            'mse_non_dropout',
            'mse_top20_de_non_dropout',
            'pearson_delta_top20_de_non_dropout', 'mse_4_non_dropout', 'mse_4_top20_de_non_dropout']

if args.wandb:
    for m in metrics:
        wandb.log({'test_round_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

print(f"Round 0 pearson delta: {np.mean([j['pearson_delta'] for i,j in out.items() if 'pearson_delta' in j])}")


round2query = {}

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    if args.strategy_name == 'EssentialSampling':
        query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), essential_gene)
    elif args.strategy_name == 'EssentialWeighted':
        if args.dataset_name == 'replogle_k562_gw_1000hvg':
            gene2ess = pickle.load(open('/home/huangk28/scratch/knowledge_kernels_gw/' + args.essential_assay + '_essential/gene2ess.pkl', 'rb'))
        else:
            gene2ess = pickle.load(open('/home/huangk28/scratch/knowledge_kernels/' + args.essential_assay + '_essential/gene2ess.pkl', 'rb'))
        query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), gene2ess)
    else:
        if args.valid_perts:
            valid_perts = strategy.net.gears_model.valid_perts
            query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), valid_perts, round = rd)
        else:
            query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), round = rd + 1)
    try:
        round2query[rd] = dataset.pert_train[query_idxs]
    except:
        print('Querying not saved...')
    print('Querying ' + str(len(query_idxs)) + ' new perturbations!')
    # update labels
    strategy.update(query_idxs)


    if args.batch_exp:
        genes_available_per_round[rd] = dataset.pert_train[query_idxs]
        strategy.train({'batch_idx': batch_idx_round, 'genes_available_per_round': genes_available_per_round})
    else:
        strategy.train()

    # calculate accuracy
    res, out = strategy.eval(test_data)

    if args.wandb:
        for m in metrics:
            wandb.log({'test_round_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

    print(f"Round {rd} pearson delta: {np.mean([j['pearson_delta'] for i,j in out.items() if 'pearson_delta' in j])}")


import pickle
with open('./res/' + args.wb_exp_name + '.pkl', 'wb') as f:
    pickle.dump(round2query, f)