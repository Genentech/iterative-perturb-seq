# Reproducing repository


### Configurations for IterPert (Fig 4a)

```
for run in {1..10}
do
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda:1' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name kernel_based_active_learning \
            --kernel_strategy Core-Set\
            --base_kernel diff_effect \
            --simple_loss \
            --use_prior \
            --wandb \
            --fix_evaluation \
            --integrate_mode mean_new \
            --normalize_mode max \
            --save_kernel
done
```

### Configurations for Kernel Based Baselines (Fig 4b)

```
for run in {1..10}
do
for strategy in Random BatchBALD BADGE Core-Set ACS-FW BALD LCMD
do
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name kernel_based_active_learning \
            --kernel_strategy $strategy\
            --base_kernel cross_gene_out \
            --simple_loss \
            --fix_evaluation \
            --save_kernel \
            --wandb
done
done
```

### Configurations for TypiClust (Fig 4b)

```
for run in {1..10}
do
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name TypiClust \
            --base_kernel linear_fix_ctrl \
            --simple_loss \
            --fix_evaluation \
            --wandb
```

### Configurations for Kmeans (Fig 4b)
```
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name KMeansSampling \
            --base_kernel linear_fix_ctrl \
            --simple_loss \
            --fix_evaluation \
            --wandb
done
```

### Run genome-wide experiments (Fig 6): change dataset name to replogle_k562_gw_1000hvg, for example, for iterpert:

```
for run in {1..10}
do
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 10 \
            --device 'cuda' \
            --n_init_labeled 300 \
            --n_query 300 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_gw_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name kernel_based_active_learning \
            --kernel_strategy Core-Set\
            --base_kernel diff_effect \
            --simple_loss \
            --use_prior \
            --wandb \
            --fix_evaluation \
            --integrate_mode mean_new \
            --normalize_mode max \
            --custom_split \
            --save_kernel
done
```


### Run individual priors (Fig 4c)

```
for run in {1..5}
do
for prior in rpe1_kernel ops_A549_kernel pops_kernel esm_kernel biogpt_kernel node2vec_kernel ops_HeLa_HPLM_kernel ops_HeLa_DMEM_kernel 
do
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name kernel_based_active_learning \
            --kernel_strategy Core-Set\
            --base_kernel diff_effect \
            --simple_loss \
            --use_prior \
            --wandb \
            --use_single_prior \
            --fix_evaluation \
            --single_prior $prior \
            --normalize_mode max \
            --integrate_mode mean_new
done
done
```

### Run batch effect experiment (Fig 7)
```
for run in {1..10}
do
# iterpert
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name kernel_based_active_learning \
            --kernel_strategy Core-Set\
            --base_kernel diff_effect \
            --simple_loss \
            --use_prior \
            --fix_evaluation \
            --integrate_mode mean_new \
            --normalize_mode max \
            --batch_exp \
            --wandb
# random

python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name kernel_based_active_learning \
            --kernel_strategy Random\
            --base_kernel linear_fix_ctrl \
            --simple_loss \
            --fix_evaluation \
            --batch_exp \
            --wandb

# typiclust
python run.py --seed 1 \
            --run $run \
            --epoch_per_cycle 20 \
            --device 'cuda' \
            --n_init_labeled 100 \
            --n_query 100 \
            --n_round 5 \
            --batch_size 256 \
            --dataset_name replogle_k562_essential_1000hvg \
            --retrain \
            --model_name GEARS\
            --strategy_name TypiClust \
            --base_kernel linear_fix_ctrl \
            --simple_loss \
            --fix_evaluation \
            --batch_exp \
            --wandb

done
```