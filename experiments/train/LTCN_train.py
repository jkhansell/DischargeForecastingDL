import numpy as np
import polars as pl
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
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
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


import orbax.checkpoint as ocp
from orbax.checkpoint import Checkpointer, utils as ocp_utils
from orbax.checkpoint import StandardCheckpointHandler
from flax.training import orbax_utils

# We'll be studying the Guadalupe River in Kerr County, TX the USGS sites are the following
# Previously exploring the data some stations report the discharge variable in another column of the data frame

sites = ["08165300", "08165500", "08166000", "08166140", "08166200"]
nan_sites = ["08166140", "08166200"]

def train_model():
    Q, cols, time = feature_engineering("./data/Q_clean.csv", sites, nan_sites)

    in_stations = np.array([i for i in range(Q.shape[1])])
    out_stations = np.array([cols["08166200"]])

    time_window = 32        # 15*4*64 hours of context 
    horizons = (4*np.array([2, 4, 8, 16, 32]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance

    logger.info(f"Time Window: {time_window} | Horizons: {horizons}")

    X, Y, Y_idx = build_multi_horizon_dataset(Q, in_stations, out_stations, time_window, horizons)
    train, val, test, times = create_train_val_test(X, Y, time)

    # initialize distributed environment
    visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]          

    jax.distributed.initialize(
        local_device_ids=visible_devices
    )

    logger.info(f"[JAX] ProcID: {jax.process_index()}")
    logger.info(f"[JAX] Local devices: {jax.local_devices()}")
    logger.info(f"[JAX] Global devices: {jax.devices()}")
    
    # Devices
    n_local_devices = jax.local_device_count()   # 4 GPUs per host
    n_total_devices = jax.device_count()         # 8 total
    logger.info(f"Host {jax.process_index()} sees {n_local_devices} local devices")

    per_device_batch_size = 64
    batch_size = per_device_batch_size * jax.device_count()
    for split in [train, val, test]:
        split["x"] = trim_to_batches(split["x"], per_device_batch_size)
        split["y"] = trim_to_batches(split["y"], per_device_batch_size)

    num_epochs = 10

    in_features = train["x"].shape[-1]
    out_features = train["y"].shape[-2]
    hidden_size = 64
    quantiles = jnp.array([0.05, 0.5, 0.95])
    
    key = jax.random.PRNGKey(123)
    x = jnp.zeros((batch_size, time_window, in_features))

    model = LTCNRegressor(
        features=out_features, 
        quantiles=len(quantiles), 
        hidden_size=hidden_size,
        dt=0.25
    )

    params = model.init(key, x)
    logger.info(f"Model size {sum(x.size for x in tree_leaves(params))}.")
    logger.info(f"Model memory {sum(x.size * x.dtype.itemsize for x in tree_leaves(params))/1024**2}.")

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

    horizon_weights = jnp.array([1.0, 1.1, 1.2, 1.5, 1.7]) 
    horizon_weights /= jnp.mean(horizon_weights)
    loss_fn = lambda x,y: quantile_loss_complex(
        x, y, quantiles, horizon_weights, crossing_penalty_coef=0.25
    )

    mesh = Mesh(jax.devices(), ('batch',))

    # Partition specs
    param_spec = P()        # fully replicated
    in_spec = P('batch', None, None)                    # shard batch
    out_spec = P('batch', None, None)   

    in_batch_sharding = NamedSharding(mesh, in_spec)
    out_batch_sharding = NamedSharding(mesh, out_spec)
    param_sharding = NamedSharding(mesh, param_spec)

    p_train_step = jax.jit(
        train_step(loss_fn),
        in_shardings=(param_sharding, in_batch_sharding, param_sharding),  # state, batch, rng
        out_shardings=(param_sharding, param_sharding)
    )

    p_eval_step = jax.jit(
        eval_step(loss_fn),
        in_shardings=(param_sharding, in_batch_sharding),
        out_shardings=(param_sharding, out_batch_sharding)         # loss, preds
    )

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        
        # Regular batch iterator
        train_gen = batch_iterator(train, batch_size=batch_size, shuffle=True)
        train_prefetch = prefetch_batches(train_gen, prefetch_size=32)

        # Regular batch iterator
        val_gen = batch_iterator(val, batch_size=batch_size, shuffle=True)
        val_prefetch = prefetch_batches(val_gen, prefetch_size=32)

        # Training phase
        train_loss = []
        val_loss = []

        for batch in train_prefetch:
            state, loss = p_train_step(state, batch, key)
            train_loss.append(loss)

        # Validation phase
        for batch in val_prefetch:
            loss, _ = p_eval_step(state, batch)
            val_loss.append(loss)

        # Compute epoch averages

        loss_train = np.mean(train_loss)
        loss_val = np.mean(val_loss)

        train_losses.append(loss_train)
        val_losses.append(loss_val)
        
        logger.info(f"Loss {jax.process_index()}: {train_loss[0]}, Name: {socket.gethostname()}")

        if jax.process_index() == 0:
            logger.info(f"Epoch {epoch+1}")
            logger.info(f"Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")


    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25, which="both")
    ax.set_yscale("log")
    fig.legend()
    fig.savefig("LSTM_Loss.png")
    plt.close()


    async_checkpointer = ocp.AsyncCheckpointer(
        ocp.StandardCheckpointHandler(), timeout_secs=60
    )

    fmt = '%Y_%m_%d'
    checkpoint_dir = os.path.abspath("./checkpoints")
    checkpoint_dir = os.path.join(checkpoint_dir, f"LTCN/model_{datetime.today().strftime(fmt)}")
   
    options = ocp.CheckpointManagerOptions(max_to_keep=2)
    
    with ocp.CheckpointManager(checkpoint_dir, async_checkpointer, options) as mngr:
        ckpt = state.params
        custom_save_args = orbax_utils.save_args_from_target(ckpt)
        mngr.save(0, ckpt, save_kwargs={'save_args': custom_save_args})
        mngr.wait_until_finished()


    jax.distributed.shutdown()
