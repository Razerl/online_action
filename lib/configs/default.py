import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# COMMON
# -----------------------------------------------------------------------------
_C.common = CN()
_C.common.dist = CN()
_C.common.dist.port = '25929'
_C.common.dist.local_rank = 0

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.model = CN()
_C.model.architecture = 'OadTR'
_C.model.num_class = 17
_C.model.frozen_weights = None
_C.model.resume = None
_C.model.tune_from = None
# -----------------------------OadTR-------------------------------------------
_C.model.enc_layers = 64
_C.model.patch_dim = 1
_C.model.embedding_dim = 1024
_C.model.num_heads = 8
_C.model.num_layers = 3
_C.model.dropout_rate = 0.1
_C.model.attn_dropout_rate = 0.1
_C.model.hidden_dim = 1024
_C.model.query_num = 8
_C.model.decoder_embedding_dim = 1024
_C.model.decoder_attn_dropout_rate = 0.1
_C.model.decoder_num_heads = 4
_C.model.decoder_layers = 5
_C.model.decoder_embedding_dim_out = 512

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.solver = CN()
_C.solver.optimizer = 'adam'
_C.solver.lr = 0.0001
_C.solver.momentum = 0.9
_C.solver.weight_decay = 0.0001

_C.solver.lr_scheduler = 'step'
_C.solver.epochs = 60
_C.solver.warmup_epoch = 0
_C.solver.lr_decay_rate = 0.1
_C.solver.lr_steps = [20, 45, 55]
_C.solver.lr_drop = 1
_C.solver.warmup_multiplier = 10

_C.solver.batch_size = 128

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.dataset = CN()
_C.dataset.data_root = ''
_C.dataset.position = 'all'
_C.dataset.num_workers = 16

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
_C.training = CN()
_C.training.start_epoch = 0
_C.training.evaluate = False
_C.training.epochs = 20
_C.training.training_print_freq = 40
_C.training.max_norm = 1.0

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.loss = CN()
_C.loss.enc_loss_coef = 1
_C.loss.dec_loss_coef = 0.3
_C.loss.similar_loss_coef = 0.1
_C.loss.enc_loss_coef = 1
_C.loss.sample_cls_index = -1
_C.loss.sample_weight = 3.0
_C.loss.contrastive_loss_margin = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.output_dir = None
