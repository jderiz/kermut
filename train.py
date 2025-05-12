# %%
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from kermut.data import (
    filter_datasets,
    prepare_GP_inputs,
    prepare_GP_kwargs,
    split_inputs,
    standardize,
)
from kermut.gp import instantiate_gp, optimize_gp, predict
from omegaconf import DictConfig


cfg = DictConfig({})


DMS_id = 'GB1'

df, y, x_toks, x_embed, x_zero_shot = prepare_GP_inputs(cfg, DMS_id)
gp_inputs = prepare_GP_kwargs(cfg, DMS_id, target_seq)

gp, likelihood = instantiate_gp(
    cfg=cfg, train_inputs=gp_inputs, train_targets=train_targets, gp_inputs=gp_inputs
)


gp, likelihood = optimize_gp( gp=gp,
    likelihood=likelihood,
    train_inputs=train_inputs,
    train_targets=train_targets,
    lr=cfg.optim.lr,
    n_steps=cfg.optim.n_steps,
    progress_bar=cfg.optim.progress_bar,
)



# save model




# load model