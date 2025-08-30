import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import socket

from utils.datautils import (
    get_data, build_multi_horizon_dataset, 
    get_discharges, create_train_val_test,
    batch_iterator, prefetch_batches, shard_batch, reshape_fn
)

from utils.trainingutils import (
    quantile_loss_complex, cosine_annealing, 
    train_step, eval_step, 
    ModelTrainState
)

from models.LSTM import LSTMRegressor

# import DL libraries
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
import flax.linen as nn 
from flax.jax_utils  import prefetch_to_device
import optax


# logging
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger()


# Define a mapping from window suffix to temporal length (in increasing order)
temporal_order = {
    "": 0,      # original column
    "2h": 1,
    "6h" : 2,
    "12h": 3,
    "1D": 4,
    "3D": 5, 
    "1W": 6,
}

# 15-min resolution -> integer number of rows
windows = {
    "2h": 8,
    "6h" : 24,
    "12h": 48,
    "1D": 96,
    "3D": 96*3, 
    "1W": 672,
#    "2W": 1344,
#    "1M": 2880,
#    "3M": 8640,
#    "6M": 17280
}

def sort_key(col):
    # Match the station name and optional MA suffix
    m = re.match(r"(\d+)(?:_MA_)?(.*)", col)
    station, suffix = m.groups()
    # Map suffix to temporal order
    return (station, temporal_order.get(suffix, 99))

## We'll be studying the Guadalupe River in Kerr County, TX the USGS sites are the following
# Previously exploring the data some stations report the discharge variable in another column of the data frame

sites = ["08165300", "08165500", "08166000", "08166140", "08166200"]
nan_sites = ["08166140", "08166200"]

import polars as pl

def train_model():

    Q = get_data("./data/Q_clean.csv", sites, nan_sites)
    Q = pl.from_pandas(Q)   

    time = Q.select("datetime")
    Q = Q.select(pl.col(pl.Float64))

    # Create a new DataFrame for all new features
    ma_features = []
    cols = {}
    for i,col in enumerate(Q.columns):
        for key, value in windows.items():
            ma_features.append(
                pl.col(col).rolling_median(window_size=value, min_samples=1).alias(f"{col}_MA_{key}")
            )
        cols[col] = i

    # Add rolling means as new columns
    Q = Q.with_columns(ma_features)
    Q = Q.select(sorted(Q.columns, key=sort_key))

    # build lagged dataset

    in_stations = np.array([i for i in range(len(Q.columns))])
    out_stations = np.array([cols["08166200"]])

    time_window = 128        # 15*4*64 hours of context 
    horizons = (4*np.array([2, 4, 8, 16, 32]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance

    logger.info(f"Time Window: {time_window} | Horizons: {horizons}")

    Q_cpu = Q.to_numpy()

    X, Y, Y_idx = build_multi_horizon_dataset(Q_cpu, in_stations, out_stations, time_window, horizons)
    X = np.log10(X)
    Y = np.log10(Y)

    # initialize distributed environment
    visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]          

    jax.distributed.initialize(
        local_device_ids=visible_devices
    )

    logger.info(f"[JAX] ProcID: {jax.process_index()}")
    logger.info(f"[JAX] Local devices: {jax.local_devices()}")
    logger.info(f"[JAX] Global devices: {jax.devices()}")
    
    num_devices = len(jax.local_devices())
    per_device_batch_size = 128  # batch size per device
    batch_size = per_device_batch_size * num_devices  # total batch across all devices

    total_rows = X.shape[0]

    # number of full batches we can create
    num_full_batches = total_rows // batch_size

    # optionally trim X so it contains only full batches
    valid_rows = num_full_batches * batch_size
    rows_per_device = valid_rows // jax.process_count()

    X = X[:valid_rows]
    Y = Y[:valid_rows]

    # Devices
    n_local_devices = jax.local_device_count()   # 4 GPUs per host
    n_total_devices = jax.device_count()         # 8 total
    logger.info(f"Host {jax.process_index()} sees {n_local_devices} local devices")

    # Each host keeps only its slice
    host_index = jax.process_index()
    start = host_index * rows_per_device
    end   = (host_index + 1) * rows_per_device
    
    X_local = X[start:end]
    Y_local = Y[start:end]
    
    train, val, test, times = create_train_val_test(X_local, Y_local, time)
    num_epochs = 20

    in_features = X.shape[-1]
    out_features = Y.shape[-2]
    hidden_size = 64
    quantiles = jnp.array([0.05, 0.5, 0.95])
    quantiles_b = jnp.broadcast_to(quantiles, (jax.device_count(),) + quantiles.shape)
    
    key = jax.random.PRNGKey(123)
    x = jnp.zeros((batch_size, time_window, in_features))

    model = LSTMRegressor(features=out_features, quantiles=len(quantiles), hidden_size=hidden_size)
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

    state = LSTMTrainState.create(apply_fn=model.apply, params=params,tx=tx)

    horizon_weights = jnp.array([1.0, 1.1, 1.2, 1.5, 1.7]) 
    horizon_weights /= jnp.mean(horizon_weights)
    loss_fn = lambda x,y: quantile_loss_complex(
        x, y, quantiles, horizon_weights, crossing_penalty_coef=0.25
    )

    Qtrain_step = train_step(loss_fn)
    Qeval_step = eval_step(loss_fn)

    # training using data parallelism
    p_train_step = jax.pmap(Qtrain_step, axis_name="batch")
    p_eval_step = jax.pmap(Qeval_step, axis_name="batch")
    state = jax.device_put_replicated(state, jax.local_devices())

    train_losses = []
    val_losses = []

    logger.info("Initializing Training Loop...")

    for epoch in range(num_epochs):
        
        # Regular batch iterator
        train_gen = batch_iterator(train, batch_size=batch_size, shuffle=True)
        train_prefetch = prefetch_batches(train_gen, prefetch_size=2)

        # Regular batch iterator
        val_gen = batch_iterator(val, batch_size=batch_size, shuffle=True)
        val_prefetch = prefetch_batches(val_gen, prefetch_size=2)

        # Training phase
        train_loss = []
        val_loss = []

        for batch in train_prefetch:
            batch = shard_batch(batch)
            state, loss = p_train_step(state, batch)
            train_loss.append(loss)

        # Validation phase
        for batch in val_prefetch:
            batch = shard_batch(batch)
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

    # Testing phase
    test_gen = batch_iterator(test, batch_size=batch_size, shuffle=False)
    test_prefetch = prefetch_batches(test_gen, prefetch_size=32)

    test_loss = []
    medians = []
    lows = []
    highs = []
    truths = []
        
    for batch in test_prefetch:
        batch = shard_batch(batch)
        loss, preds = p_eval_step(state, batch)
        test_loss.append(np.mean(loss))
        
        # append for graphing
        truths.append(batch["y"].reshape(-1, batch["y"].shape[2]))
        lows.append(preds[..., 0].reshape(-1, preds.shape[2]))
        medians.append(preds[..., 1].reshape(-1, preds.shape[2]))
        highs.append(preds[..., 2].reshape(-1, preds.shape[2]))

    medians = np.concatenate(medians, axis=0).astype(np.float32)
    lows = np.concatenate(lows, axis=0).astype(np.float32)
    highs = np.concatenate(highs, axis=0).astype(np.float32)
    truths = np.concatenate(truths, axis=0).astype(np.float32)
    test_loss = np.array(test_loss).flatten().astype(np.float32)

    # plot losses

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

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(test_loss, label="Test Loss Histogram")
    fig.legend()
    fig.savefig("LSTM_Test_Loss.png")
    plt.close()

    # Multi horizon prediction [2, 4, 8, 16, 32] h in advance

    for i in range(medians.shape[1]):

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(10**truths[:,i],  label="Ground Truth")
        ax.plot(10**medians[:,i], label="Median", linewidth=2)
        ax.fill_between(np.arange(lows[:,i].shape[0]), 10**lows[:,i], 10**highs[:,i], alpha=0.25, label="Quantiles = [0.05,0.95]")
        ax.set_xlabel("Time point")
        ax.set_ylabel("Flow Discharge [cfs]")
        ax.grid(alpha=0.25)
        ax.set_yscale("log")
        fig.suptitle(f"LSTM Forecast - Horizon {horizons[i]//4}h")
        fig.legend()
        fig.savefig(f"LSTMpredictions_{i}.png")
        plt.close()
    
    jax.distributed.shutdown()
    