### Pretrain
```bash
$ mkdir data
$ mkdir logs
$ python train_pmnet.py [-data_dir DATA_DIR] [-dataset DATASET]
                        [-pretrain_batch_size PRETRAIN_BATCH_SIZE]
                        [-num_pretrain_layers NUM_PRETRAIN_LAYERS] [-rbf_size RBF_SIZE]
                        [-hidden_size_pretrain HIDDEN_SIZE_PRETRAIN]
                        [-num_feats NUM_FEATS] [-num_elems NUM_ELEMS]
                        [-num_bond_types NUM_BOND_TYPES]
                        [-pretrain_epochs PRETRAIN_EPOCHS]
                        [-lr LR] [-dropout DROPOUT] [-cuda]
                        [-gpu GPU] [-pretrain]
                        [-num_head NUM_HEAD]
# example
$ python train_pmnet.py -cuda -pretrain
```

### Fine-tune
```bash
$ python train_pmnet.py [-data_dir DATA_DIR] [-dataset DATASET]
                        [-num_feats NUM_FEATS]
                        [-ckpt_step CKPT_STEP]
                        [-ckpt_file CKPT_FILE]
                        [-lr LR] [-dropout DROPOUT] [-cuda] [-scaffold]
                        [-train_valid_test TRAIN_VALID_TEST [TRAIN_VALID_TEST ...]]
                        [-gpu GPU] [-pred PRED]
                        [-pred_batch_size PRED_BATCH_SIZE]
                        [-pred_epochs PRED_EPOCHS]
# example
$ python train_pmnet.py -pred biochem -cuda -data_dir data -dataset estrogen-beta -scaffold -ckpt_file logs/epoch_119.th -pred_batch_size 32 -ckpt_step 50 -lr 5e-5 -pred_epochs 200
```

### Arguments
```bash
  -data_dir DATA_DIR
  -dataset DATASET
  -pretrain_batch_size PRETRAIN_BATCH_SIZE
  -num_pretrain_layers NUM_PRETRAIN_LAYERS
  -rbf_size RBF_SIZE
  -hidden_size_pretrain HIDDEN_SIZE_PRETRAIN
  -num_feats NUM_FEATS
  -num_elems NUM_ELEMS
  -num_bond_types NUM_BOND_TYPES
  -ckpt_step CKPT_STEP
  -ckpt_file CKPT_FILE
  -pretrain_epochs PRETRAIN_EPOCHS
  -lr LR
  -dropout DROPOUT
  -cuda
  -scaffold
  -train_valid_test TRAIN_VALID_TEST [TRAIN_VALID_TEST ...]
  -gpu GPU
  -pretrain
  -pred PRED
  -pred_batch_size PRED_BATCH_SIZE
  -num_head NUM_HEAD
  -pred_epochs PRED_EPOCHS
```