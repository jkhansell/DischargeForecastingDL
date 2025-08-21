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

    time_window = 64        # 16 hours of context 
    horizons = (15*4*jnp.array([2, 4, 8, 16, 32]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance
    
    device_cpu = jax.devices("cpu")[0]
    Q_cpu = jax.device_put(Q.to_numpy(), device=device_cpu)

    # compiling function for CPU
    X, Y, Y_idx = build_multi_horizon_dataset(Q_cpu, in_stations, out_stations, time_window, horizons)
    time = time.to_numpy()[Y_idx]

    X = jnp.log10(X)
    Y = jnp.log10(Y)

    # make batches 
    batch_size = 256
    
    assert batch_size % jax.local_device_count() == 0

    train, val, test, times = create_train_val_test(X, Y, time)
    
    num_epochs = 30

    in_features = X.shape[-1]
    out_features = Y.shape[-2]
    hidden_size = 32
    quantiles = jnp.array([0.1, 0.5, 0.9])
    quantiles_b = jnp.broadcast_to(quantiles, (jax.local_device_count(),) + quantiles.shape)
    
    key = jax.random.PRNGKey(123)
    x = jnp.zeros((batch_size, time_window, in_features))

    model LTCN