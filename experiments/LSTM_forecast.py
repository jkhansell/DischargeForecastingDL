import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

from utils.datautils import (
    get_data, build_multi_horizon_dataset, 
    get_discharges, create_train_val_test,
    batch_iterator, prefetch_batches, shard_batch, reshape_fn
)
from utils.trainingutils import quantile_loss_complex, cosine_annealing

from models.LSTM import LSTMRegressor, LSTMTrainState, LSTMtrain_step, LSTMeval_step

# import DL libraries
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
import flax.linen as nn 
from flax.jax_utils  import prefetch_to_device
import optax

# Define a mapping from window suffix to temporal length (in increasing order)
temporal_order = {
    "": 0,      # original column
    "2h": 1,
    "6h" : 2,
    "12h": 3,
    "1D": 4,
    "1W": 5,
}

# 15-min resolution -> integer number of rows
windows = {
    "2h": 8,
    "6h" : 24,
    "12h": 48,
    "1D": 96,
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

    time_window = 64        # 16 hours of context 
    horizons = (15*4*np.array([2, 4, 8, 16, 32]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance

    Q_cpu = Q.to_numpy()

    X, Y, Y_idx = build_multi_horizon_dataset(Q_cpu, in_stations, out_stations, time_window, horizons)
    X = np.log10(X)
    Y = np.log10(Y)

    # initialize distributed environment
    visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]          

    jax.distributed.initialize(
        local_device_ids=visible_devices
    )

    print(f"[JAX] ProcID: {jax.process_index()}")
    print(f"[JAX] Local devices: {jax.local_devices()}")
    print(f"[JAX] Global devices: {jax.devices()}")
    
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
    print(f"Host {jax.process_index()} sees {n_local_devices} local devices")

    # Each host keeps only its slice
    host_index = jax.process_index()
    start = host_index * rows_per_device
    end   = (host_index + 1) * rows_per_device
    
    X_local = X[start:end]
    Y_local = Y[start:end]
    
    train, val, test, times = create_train_val_test(X_local, Y_local, time)
   
    print(X_local.shape)
    num_epochs = 35

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
    warmup_steps = 50

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,          # very small start
        peak_value=1e-4,          # max LR (after warmup)
        warmup_steps=warmup_steps,
        decay_steps=total_steps-warmup_steps,  # decay until end of training
        end_value=1e-6            # LR at final step (lower = steeper decay)
    )
    
    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-5)

    state = LSTMTrainState.create(apply_fn=model.apply, params=params,tx=tx)

    loss_fn = lambda x,y: quantile_loss_complex(
        x, y, quantiles, crossing_penalty_coef=0.2
    )

    train_step = LSTMtrain_step(loss_fn)
    eval_step = LSTMeval_step(loss_fn)

    # training using data parallelism
    p_LSTMtrain_step = jax.pmap(train_step, axis_name="batch")
    p_LSTMeval_step = jax.pmap(eval_step, axis_name="batch")
    state = jax.device_put_replicated(state, jax.local_devices())

    train_losses = []
    val_losses = []

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
            state, loss = p_LSTMtrain_step(state, batch)
            train_loss.append(loss)

        # Validation phase
        for batch in val_prefetch:
            batch = shard_batch(batch)
            loss, _ = p_LSTMeval_step(state, batch)
            val_loss.append(loss)

        # Compute epoch averages

        loss_train = np.mean(train_loss)
        loss_val = np.mean(val_loss)

        train_losses.append(loss_train)
        val_losses.append(loss_val)

        if jax.process_index() == 0:
            print(f"Epoch {epoch+1}")
            print(f"Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")

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
        loss, preds = p_LSTMeval_step(state, batch)
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
        ax.plot(10**medians[:,i], label="Median")
        ax.fill_between(np.arange(lows[:,i].shape[0]), 10**lows[:,i], 10**highs[:,i], alpha=0.25, label="Quantiles = [0.05,0.95]")
        ax.set_xlabel("Time point")
        ax.set_ylabel("Flow Discharge [m^3/s]")
        ax.grid(alpha=0.25)
        ax.set_yscale("log")
        fig.legend()
        fig.savefig(f"LSTMpredictions_{i}.png")
        plt.close()
    
    jax.distributed.shutdown()
    