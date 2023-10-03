import json
import sys
import time
import copy
from pathlib import Path
import warnings
from copy import deepcopy

import torch
import numpy as np
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

set_seed(42)
# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2

n_hvg = 0  # number of highly variable genes
include_zero_gene = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True
load_model = "/home/huangk28/projects/active_pert/save/scGPT_human"
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]
schedule_interval = 1
early_stop = 5
# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100


class scGPT_active_interface:

    def __init__(self, pert_data, device, save_dir):
        self.pert_data = pert_data
        self.device = device
        self.save_dir = save_dir
        self.adata = pert_data.adata
        self.eval_class = eval_analysis_fast(self.adata)


    def model_initialize(self):
        logger = scg.logger
        scg.utils.add_file_handler(logger, self.save_dir + "/run.log")
        # log running date and current git commit
        logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if load_model is not None:
            model_dir = Path(load_model)
            model_config_file = model_dir / "args.json"
            model_file = model_dir / "best_model.pt"
            vocab_file = model_dir / "vocab.json"

            vocab = GeneVocab.from_file(vocab_file)
            for s in special_tokens:
                if s not in vocab:
                    vocab.append_token(s)

            self.pert_data.adata.var["id_in_vocab"] = [
                1 if gene in vocab else -1 for gene in self.pert_data.adata.var["gene_name"]
            ]
            gene_ids_in_vocab = np.array(self.pert_data.adata.var["id_in_vocab"])
            logger.info(
                f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
                f"in vocabulary of size {len(vocab)}."
            )
            genes = self.pert_data.adata.var["gene_name"].tolist()

            # model
            with open(model_config_file, "r") as f:
                model_configs = json.load(f)
            logger.info(
                f"Resume model from {model_file}, the model args will override the "
                f"config {model_config_file}."
            )
            embsize = model_configs["embsize"]
            nhead = model_configs["nheads"]
            d_hid = model_configs["d_hid"]
            nlayers = model_configs["nlayers"]
            n_layers_cls = model_configs["n_layers_cls"]
        else:
            genes = self.pert_data.adata.var["gene_name"].tolist()
            vocab = Vocab(
                VocabPybind(genes + special_tokens, None)
            )  # bidirectional lookup [gene <-> int]
        vocab.set_default_index(vocab["<pad>"])
        self.gene_ids = np.array(
            [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
        )
        self.n_genes = len(genes)

        ntokens = len(vocab)  # size of vocabulary
        model = TransformerGenerator(
            ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=n_layers_cls,
            n_cls=1,
            vocab=vocab,
            dropout=dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            pert_pad_id=pert_pad_id,
            do_mvc=MVC,
            cell_emb_style=cell_emb_style,
            mvc_decoder_style=mvc_decoder_style,
            use_fast_transformer=use_fast_transformer,
        )
        if load_param_prefixs is not None and load_model is not None:
            # only load params that start with the prefix
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if any([k.startswith(prefix) for prefix in load_param_prefixs])
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        elif load_model is not None:
            try:
                model.load_state_dict(torch.load(model_file))
                logger.info(f"Loading all model params from {model_file}")
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_file)
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    logger.info(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
        model.to(self.device)
        self.model = model
        self.best_model = deepcopy(self.model)
        self.logger = logger

    def train(self, epochs, lr, data):
        train_loader = data['train_loader']
        valid_loader = data['val_loader']
        epochs = epochs

        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        best_val_loss = float("inf")
        patience = 0

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            train_scgpt(
                self.model,
                train_loader,
                self.device, self.n_genes, self.gene_ids, criterion,
                scaler, scheduler, self.logger, 
                optimizer, epoch
            )
            val_loss, val_mre = evaluate_scgpt(
                self.model,
                valid_loader,
                self.device, self.n_genes, 
                self.gene_ids, criterion
            )
            elapsed = time.time() - epoch_start_time
            self.logger.info("-" * 89)
            self.logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} |"
            )
            self.logger.info("-" * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model)
                self.logger.info(f"Best model with score {best_val_loss:5.4f}")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    self.logger.info(f"Early stop at epoch {epoch}")
                    break

            scheduler.step()
        torch.save(self.best_model.state_dict(), self.save_dir + "/best_model.pt")


    def predict_from_loader(self, dataloader, detail_eval):
        print('Start evaluating...')   
        s = time.time()     
        res = eval_perturb(dataloader, self.best_model, self.device, self.gene_ids)
        
        print('Finished evaluating with time '+ str((time.time() - s)/60) + ' min')
        if detail_eval:
            s = time.time()
            print('Starting initializing eval_analysis_fast analysis...')
            out_class = self.eval_class.get_res_all_perts(res)
            print('eval_analysis_fast finished with time ' + str((time.time() - s)/60) + ' min')
            
            return res, out_class
        else:
            return res


def train_scgpt(model, train_loader, 
                device, n_genes, gene_ids, criterion,
                scaler, scheduler, logger, 
                optimizer, epoch):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def evaluate_scgpt(model, val_loader, device, n_genes, gene_ids, criterion):
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)


def eval_perturb(loader, model, device, gene_ids):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(batch, include_zero_gene, gene_ids=gene_ids)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy()
    results["truth"] = truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy()
    results["truth_de"] = truth_de.detach().cpu().numpy()

    return results


from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

class eval_analysis_fast:

    def __init__(self, adata):
        self.adata = adata
        
        ## in silico modeling and upperbounding
        self.pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
        self.geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
        self.geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
        self.ctrl = np.mean(adata.X[np.where(adata.obs.condition == 'ctrl')[0]], axis = 0)
        self.gene_list = adata.var['gene_name'].values

        
    def per_perturb_analysis(self, pert, test_res):
        res = {}
        #if len(self.ctrl) > 1:
        #    ctrl = [self.ctrl]
        #else:
        ctrl = np.array(self.ctrl).reshape(1, -1)
        pert_idx = np.where(test_res['pert_cat'] == pert)[0]
        de_idx = [self.geneid2idx[i] for i in self.adata.uns['top_non_dropout_de_20'][self.pert2pert_full_id[pert]]]

        # 'pearson_delta'
        val = pearsonr(test_res['pred'][pert_idx].mean(0)- ctrl[0], test_res['truth'][pert_idx].mean(0)-ctrl[0])[0]
        if np.isnan(val):
            val = 0
        res['pearson_delta'] = val

        # 'frac_opposite_direction_top20_non_dropout',
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]))            
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(de_idx)
        res['frac_opposite_direction_top20_non_dropout'] = frac_direction_opposite

        # 'mse_top20_de_non_dropout'
        val = mse(test_res['pred'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][pert_idx].mean(0)[de_idx]-ctrl[0][de_idx])
        if np.isnan(val):
            val = 0
        res['mse_top20_de_non_dropout'] = val
        return res

    def get_res_all_perts(self, test_res):
        pert2res = {}
        unique_perts = np.unique(test_res['pert_cat'])
        #with multiprocessing.Pool(int(os.cpu_count()/2)) as p:
        #    r = list(tqdm(p.imap(self.per_perturb_analysis, unique_perts), total = len(unique_perts)))
        #return dict(zip(unique_perts, r))
        for pert in tqdm(unique_perts):
            pert2res[pert] = self.per_perturb_analysis(pert, test_res)
        return pert2res