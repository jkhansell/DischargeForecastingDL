import numpy as np
import pandas as pd
from datetime import datetime
import re
import os
import socket
import matplotlib.pyplot as plt

from utils.datautils import (
    get_data, build_multi_horizon_dataset, 
    get_discharges, create_train_val_test,
    batch_iterator, prefetch_batches, shard_batch, reshape_fn,
    feature_engineering, trim_to_batches
)

from utils.trainingutils import (
    quantile_loss_complex, cosine_annealing, 
    train_step, eval_step, 
    ModelTrainState
)

from models.LTCN import LTCNRegressor 

# import DL libraries
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from jax.tree_util import tree_leaves

import flax.linen as nn 
from flax.jax_utils import unreplicate

import optax

# logging
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger()

# model parameter saving
import orbax.checkpoint as ocp
from orbax.checkpoint import utils as ocp_utils
from flax.jax_utils import unreplicate

# We'll be studying the Guadalupe River in Kerr County, TX the USGS sites are the following
# Previously exploring the data some stations report the discharge variable in another column of the data frame

sites = ["08165300", "08165500", "08166000", "08166140", "08166200"]
nan_sites = ["08166140", "08166200"]

import polars as pl

def test_model():
    Q = get_data("/work/jovillalobos/hidrologia/DischargeForecastingDL/data/Q_clean.csv", sites, nan_sites)
    Q = pl.from_pandas(Q)   

    time = Q.select("datetime")
    time = time["datetime"].str.to_datetime("%Y-%m-%d %H:%M:%S%z")

    Q = Q.select(pl.col(pl.Float64))
    
    Q, cols = feature_engineering(Q,time)

    # build lagged dataset

    in_stations = np.array([i for i in range(len(Q.columns))])
    out_stations = np.array([cols["08166200"]])

    time_window = 32        # 15*4*64 hours of context 
    horizons = (4*np.array([2, 4, 8, 16, 32]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance

    logger.info(f"Time Window: {time_window} | Horizons: {horizons}")

    Q_cpu = Q.to_numpy()

    X, Y, Y_idx = build_multi_horizon_dataset(Q_cpu, in_stations, out_stations, time_window, horizons)
    train, val, test, times = create_train_val_test(X, Y, time)

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=1000,
        max_to_keep=3,
        enable_async_checkpointing=True,
        create=True
    )

    in_features = X.shape[-1]
    out_features = Y.shape[-2]
    hidden_size = 64
    quantiles = jnp.array([0.05, 0.5, 0.95])
    
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, num=jax.local_device_count()) 
    x = jnp.zeros((batch_size, time_window, in_features))

    model = LTCNRegressor(
        features=out_features, 
        quantiles=len(quantiles), 
        hidden_size=hidden_size,
        dt=0.25
    )

    params = model.init(key, x)

        # Training setup
    steps_per_epoch = len(train["x"]) // batch_size
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 500

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,          # very small start
        peak_value=8e-5,          # max LR (after warmup)
        warmup_steps=warmup_steps,
        decay_steps=total_steps-warmup_steps,  # decay until end of training
        end_value=1e-5            # LR at final step (lower = steeper decay)
    )
    
    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-5)
    state = ModelTrainState.create(apply_fn=model.apply, params=params,tx=tx)
    state = jax.tree_util.tree_map(ocp_utils.fully_replicated_host_local_array_to_global_array, state) 

    # 1. Define your checkpoint directory and options for async saving
    path = "/work/jovillalobos/hidrologia/DischargeForecastingDL/checkpoints/LTCN/model_2025_08_30"
    with ocp.CheckpointManager(
        path,
        options=options
    ) as mngr:

        mngr.restore(0,args=ocp.args.StandardRestore(state))
