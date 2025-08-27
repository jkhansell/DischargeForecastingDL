import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from utils.datautils import (
    get_data, build_multi_horizon_dataset, 
    get_discharges, create_train_val_test,
    batch_iterator, prefetch_batches, shard_batch, reshape_fn,
)

from utils.trainingutils import cosine_annealing

from models.QATN import (
    QATNRegressor, QATNTrainState, 
    QATNtrain_step, QATNeval_step
)

# import DL libraries
import jax
import jax.numpy as jnp
import flax.linen as nn 
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

    # Feature engineering, augmenting dataset with
    # 1 week MA
    # 2 week MA
    # 1 month MA
    # 3 month MA
    # 6 month MA

    # 15-min resolution -> integer number of rows
    windows = {
        "3h": 12,
        "6h": 24,
        "12h": 48,
        "1D": 96,
        "1W": 672,
        "2W": 1344,
    }

    time = Q.select("datetime")
    Q = Q.select(pl.col(pl.Float64))
    
    # Create a new DataFrame for all new features
    ma_features = []
    cols = {}
    for i,col in enumerate(Q.columns):
        #for key, value in windows.items():
        #    ma_features += [
        #        pl.col(col).rolling_std(window_size=value, min_periods=1).alias(f"{col}_STD_{key}"),
        #        pl.col(col).rolling_min(window_size=value, min_periods=1).alias(f"{col}_MIN_{key}"),
        #        pl.col(col).rolling_max(window_size=value, min_periods=1).alias(f"{col}_MAX_{key}"),
        #        pl.col(col).rolling_mean(window_size=value, min_periods=1).alias(f"{col}_MEAN_{key}"),
        #        #(pl.col(col) - pl.col(col).shift(value)).alias(f"{col}_SLOPE_{key}"),
        #        #(pl.col(col) - pl.col(col).rolling_mean(window_size=value, min_periods=1)).alias(f"{col}_RESID_{key}"),
        #    ]
        cols[col] = i
    # Add rolling means as new columns
    #Q = Q.with_columns(ma_features)
    #Q = Q.select(sorted(Q.columns, key=sort_key))
    #Q = Q.drop_nulls()
    print(Q)

    # build lagged dataset

    in_stations = jnp.array([i for i in range(len(Q.columns))])
    out_stations = jnp.array([cols["08166200"]])

    time_window = 32        # 32 hours of context 
    horizons = (15*4*jnp.array([2, 4, 8, 16, 32]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance
    
    device_cpu = jax.devices("cpu")[0]
    Q_cpu = jax.device_put(Q.to_numpy(), device=device_cpu)


    # compiling function for CPU
    X, Y, Y_idx = build_multi_horizon_dataset(Q_cpu, in_stations, out_stations, time_window, horizons)
    time = time.to_numpy()[Y_idx]

    #lam = -1.5
    X = jnp.log10(X+1)
    Y = jnp.log10(Y+1)

    #Xmax = X.max(axis=1, keepdims=True)
    #Ymax = Y.max(axis=1, keepdims=True)
    
    # make batches 
    batch_size = 128
    assert batch_size % jax.local_device_count() == 0

    train, val, test, times = create_train_val_test(X, Y, time)
    
    num_epochs = 20

    in_features = X.shape[-1]
    out_features = Y.shape[-2]
    hidden_size = 64
    quantiles = jnp.array([0.05, 0.5, 0.95])
    quantiles_b = jnp.broadcast_to(quantiles, (jax.local_device_count(),) + quantiles.shape)
    
    key = jax.random.PRNGKey(123)
    x = jnp.zeros((batch_size, time_window, in_features))

    model = QATNRegressor(
        features=out_features, 
        quantiles=len(quantiles), 
        hidden_size=hidden_size,
        depth=4, 
        n_heads=4,
        causal=True, 
        dropout=0.0 
    )

    params = model.init(key, x)

    # Training setup
    steps_per_epoch = len(train["x"]) // batch_size
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 150

    #schedule = optax.warmup_cosine_decay_schedule(
    #    init_value=1e-5,          # very small start
    #    peak_value=8e-4,          # max LR (after warmup)
    #    warmup_steps=warmup_steps,
    #    decay_steps=total_steps-warmup_steps,  # decay until end of training
    #    end_value=1e-6            # LR at final step (lower = steeper decay)
    #)

    schedule = lambda step: cosine_annealing(
        step,
        base_lr=1e-3,
        min_lr=1e-4,
        steps_per_cycle=total_steps//2,
        m_mul=0.5,
        t_mul=1.0
    )

    tx = optax.noisy_sgd(learning_rate=schedule)#, weight_decay=5e-5)

    state = QATNTrainState.create(apply_fn=model.apply, params=params,tx=tx)

    print(jax.local_devices())
    # training using data parallelism
    p_QATNtrain_step = jax.pmap(QATNtrain_step, axis_name="batch")
    p_QATNeval_step = jax.pmap(QATNeval_step, axis_name="batch")
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
            state, loss = p_QATNtrain_step(state, batch, quantiles_b)
            train_loss.append(loss)

        # Validation phase
        for batch in val_prefetch:
            batch = shard_batch(batch)
            loss, _ = p_QATNeval_step(state, batch, quantiles_b)
            val_loss.append(loss)

        # Compute epoch averages

        loss_train = np.mean(train_loss)
        loss_val = np.mean(val_loss)

        train_losses.append(loss_train)
        val_losses.append(loss_val)

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
        loss, preds = p_QATNeval_step(state, batch, quantiles_b)
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
    fig.savefig("QATN_Loss.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(test_loss, label="Test Loss Histogram")
    fig.legend()
    fig.savefig("QATN_Test_Loss.png")
    plt.close()

    # Multi horizon prediction [2, 4, 8, 16, 32] h in advance

    #decoded_medians = 10**( medians) #boxcox_decode(medians, lam)
    #decoded_lows = 10**(lows) #boxcox_decode(lows, lam)
    #decoded_highs = 10**(highs) #boxcox_decode(highs, lam)


    for i in range(medians.shape[1]):
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(10**(medians[:,i]-1), label="Median")
        ax.fill_between(np.arange(lows[:,i].shape[0]), 10**(lows[:,i]-1), 10**(highs[:,i]-1), alpha=0.25, label="Quantiles = [0.05,0.95]")
        ax.plot(10**truths[:,i], label="Ground Truth")
        ax.set_xlabel("Time point")
        ax.set_ylabel("Flow Discharge [m^3/s]")
        ax.grid(alpha=0.25)
        ax.set_yscale("log")
        fig.legend()
        fig.savefig(f"QATNpredictions_{i}.png")
        plt.close()