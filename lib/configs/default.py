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
_C.model.numclass = 17

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



# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "output/mask-rcnn-r-50-c4-1x/"

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
