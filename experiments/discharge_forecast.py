import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from utils.datautils import (
    get_data, build_multi_horizon_dataset, 
    get_discharges, create_train_val_test,
    batch_iterator, prefetch_batches, shard_batch, reshape_fn
)

from models.LSTM import LSTMRegressor, LSTMTrainState, LSTMtrain_step, LSTMeval_step


# import DL libraries
import jax
import jax.numpy as jnp
import flax.linen as nn 
import optax


# Define a mapping from window suffix to temporal length (in increasing order)
temporal_order = {
    "": 0,      # original column
    "12h": 1,
    "1D": 2,
    "1W": 3,
    "2W": 4,
    "1M": 5,
    "3M": 6,
    "6M": 7
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

    # Feature engineering, augmenting dataset with
    # 1 week MA
    # 2 week MA
    # 1 month MA
    # 3 month MA
    # 6 month MA

    # 15-min resolution -> integer number of rows
    windows = {
        "12h": 48,
        "1D": 96,
        "1W": 672,
        "2W": 1344,
        "1M": 2880,
        "3M": 8640,
        "6M": 17280
    }

    time = Q.select("datetime")
    Q = Q.select(pl.col(pl.Float64))
    
    # Create a new DataFrame for all new features
    ma_features = []
    cols = {}
    for i,col in enumerate(Q.columns):
        for key, value in windows.items():
            ma_features.append(
                pl.col(col).rolling_mean(window_size=value, min_periods=1).alias(f"{col}_MA_{key}")
            )
        cols[col] = i
    # Add rolling means as new columns
    Q = Q.with_columns(ma_features)
    Q = Q.select(sorted(Q.columns, key=sort_key))

    # build lagged dataset

    in_stations = jnp.array([i for i in range(len(Q.columns))])
    out_stations = jnp.array([cols["08166200"]])

    time_window = 32        # 8 hours of context 
    horizons = (15*4*jnp.array([2, 4, 8, 16, 32]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance
    
    device_cpu = jax.devices("cpu")[0]
    Q_cpu = jax.device_put(Q.to_numpy(), device=device_cpu)

    # compiling function for CPU
    X, Y, Y_idx = build_multi_horizon_dataset(Q_cpu, in_stations, out_stations, time_window, horizons)
    time = time.to_numpy()[Y_idx]

    X = jnp.log10(X)
    Y = jnp.log10(Y)

    # make batches 
    batch_size = 512
    
    assert batch_size % jax.local_device_count() == 0

    train, val, test, times = create_train_val_test(X, Y, time)
    
    num_epochs = 30

    in_features = X.shape[-1]
    out_features = Y.shape[-2]
    hidden_size = 64
    quantiles = jnp.array([0.1, 0.5, 0.9])
    quantiles_b = jnp.broadcast_to(quantiles, (jax.local_device_count(),) + quantiles.shape)
    
    key = jax.random.PRNGKey(123)
    x = jnp.zeros((batch_size, time_window, in_features))

    model = LSTMRegressor(features=out_features, quantiles=len(quantiles), hidden_size=hidden_size)
    params = model.init(key, x)
    tx = optax.adamw(learning_rate=5e-3, weight_decay=1e-4)
    state = LSTMTrainState.create(apply_fn=model.apply, params=params,tx=tx)

    print(jax.local_devices())
    # training using data parallelism
    p_LSTMtrain_step = jax.pmap(LSTMtrain_step, axis_name="batch")
    p_LSTMeval_step = jax.pmap(LSTMeval_step, axis_name="batch")
    state = jax.device_put_replicated(state, jax.local_devices())

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
            batch = shard_batch(batch)
            state, loss = p_LSTMtrain_step(state, batch, quantiles_b)
            train_loss.append(loss)

        # Validation phase
        for batch in val_prefetch:
            batch = shard_batch(batch)
            loss, _ = p_LSTMeval_step(state, batch, quantiles_b)
            val_loss.append(loss)

        # Compute epoch averages

        loss_train = np.mean(train_loss)
        loss_val = np.mean(val_loss)

        train_losses.append(loss_train)
        val_losses.append(loss_val)

        print(f"Epoch {epoch+1}, Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")

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
        loss, preds = p_LSTMeval_step(state, batch, quantiles_b)
        test_loss.append(np.mean(loss))
        
        # append for graphing
        truths.append(batch["y"].reshape(-1, batch["y"].shape[2]))
        lows.append(preds[..., 0].reshape(-1, preds.shape[2]))
        medians.append(preds[..., 1].reshape(-1, preds.shape[2]))
        highs.append(preds[..., 2].reshape(-1, preds.shape[2]))

    medians = np.concatenate(medians, axis=0)
    lows = np.concatenate(lows, axis=0)
    highs = np.concatenate(highs, axis=0)
    truths = np.concatenate(truths, axis=0)

    test_loss = np.array(test_loss).flatten()

    # plot losses

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25, which="both")
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
        ax.plot(10**medians[:,i], label="Median")
        ax.fill_between(np.arange(lows[:,i].shape[0]), 10**lows[:,i], 10**highs[:,i], alpha=0.25, label="Quantiles = [0.1,0.9]")
        ax.plot(10**truths[:,i], label="Ground Truth")
        ax.set_xlabel("Time point")
        ax.set_ylabel("Flow Discharge [m^3/s]")
        ax.grid(alpha=0.25)
        ax.set_yscale("log")
        fig.legend()
        fig.savefig(f"predictions_{i}.png")
        plt.close()

    