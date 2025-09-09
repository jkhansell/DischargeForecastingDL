import dataretrieval.nwis as nwis
import re

import numpy as np
import pandas as pd
import polars as pl

import jax
import jax.numpy as jnp

import threading
import queue


# Constants
CFS2M3S = 0.028316846592        # cfs to m^3/s

# As some time series might be incomplete we use brownian bridges to interpolate in between the gaps

def brownian_bridge(start, end, n_steps, step_std=1.0):
    """Generate Brownian bridge from start to end with n_steps points."""
    increments = np.random.normal(0, step_std, n_steps)
    walk = np.cumsum(increments)
    t = np.linspace(0, 1, n_steps)
    bridge = start + (walk - t * walk[-1]) + t * (end - start)
    return bridge

def fill_nans_with_random_walk(s, step_std=1.0, threshold=0.1):
    """Fill NaNs in df[col] with Brownian bridge random walks."""
    n = len(s)
    series = s.copy()
    series[series < threshold] = np.nan
    series[series < 0] = np.nan
    i = 0
    while i < n:
        if pd.isna(series.iloc[i]):
            # Start of NaN block
            start_idx = i - 1
            while i < n and pd.isna(series.iloc[i]):
                i += 1
            end_idx = i
            if start_idx >= 0 and end_idx < n:
                start_val = series.iloc[start_idx]
                end_val = series.iloc[end_idx]
                n_steps = end_idx - start_idx + 1  # include start and end
                bridge = brownian_bridge(start_val, end_val, n_steps, step_std)
                series.iloc[start_idx:end_idx + 1] = bridge
        else:
            i += 1
    return series.clip(lower=threshold)

def get_discharges(sites, nan_sites, service="iv", start_date="2005-01-01", end_date="2025-08-15"):

    df = nwis.get_record(sites=sites, service=service, start=start_date, end=end_date)
    Q = pd.DataFrame()
    if service == "iv":
        for site in sites:
            station_df = df.loc[site]
            if site in nan_sites:
                data = station_df["00060"]
            else:
                data = station_df["00060_15-minute update"]

            Q[site] = data
    else:
        for site in sites:
            station_df = df.loc[site]
            data = station_df["00060_Mean"]
            Q[site] = data
    return Q
    # bear in mind that the points are within 15 minute intervals

def get_data(path, sites, nan_sites):
    try: 
        Q = pd.read_csv(path)
    except:
        Q = get_discharges(sites, nan_sites, service="iv", start_date="2005-01-01", end_date="2025-08-29")        
        Q.to_csv("./data/Q_raw.csv")

    Q_filled = pd.DataFrame()
    for col in Q.columns:
        if col == "datetime":
            Q_filled[col] = Q[col]
        else:
            data = Q[col].interpolate(method="linear")
            data = fill_nans_with_random_walk(data, 0.5, 0.1)
            Q_filled[col] = data
    
    Q_filled = Q_filled.dropna()
    return Q, Q_filled

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
}

def sort_key(col):
    # Match the station name and optional MA suffix
    m = re.match(r"(\d+)(?:_MA_)?(.*)", col)
    station, suffix = m.groups()
    # Map suffix to temporal order
    return (station, temporal_order.get(suffix, 99))

def feature_engineering(path, sites, nan_sites):
    Q_raw, Q = get_data(path, sites, nan_sites)
    Q = pl.from_pandas(Q)

    time = Q.select("datetime")
    time = time["datetime"].str.to_datetime("%Y-%m-%d %H:%M:%S%z")

    Q = Q.select(pl.col(pl.Float64))
    Q = Q.select([
        pl.col(c).log10().alias(c) #removed log10
        for c in Q.columns 
    ])

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

    # Add temporal sinusoidal encoding to data 
    Q = Q.with_columns([
        (2 * np.pi * time.dt.hour() / 24).sin().alias("hour_sin"),
        (2 * np.pi * time.dt.hour() / 24).cos().alias("hour_cos"),
        (2 * np.pi * time.dt.weekday() / 7).sin().alias("dow_sin"),
        (2 * np.pi * time.dt.weekday() / 7).cos().alias("dow_cos"),
        (2 * np.pi * time.dt.ordinal_day() / 365).sin().alias("doy_sin"),
        (2 * np.pi * time.dt.ordinal_day() / 365).cos().alias("doy_cos"),
    ])

    return Q.to_numpy(), cols, time

def build_multi_horizon_dataset(Q, in_stations, out_stations, p, horizons):
    """
    Build input matrix X and multi-horizon target Y for training.

    Parameters:
    - Q: np.array of shape (time_steps, stations)
    - in_stations: list of input station indices
    - out_stations: list of output station indices
    - p: int, number of lags (time steps)
    - horizons: list of prediction horizons (steps ahead)
    - time: optional array-like of length time_steps, datetime or numeric

    Returns:
    - X: np.array of shape (num_samples, p, len(in_stations))
    - Y: np.array of shape (num_samples, len(horizons))
    - T: np.array of length num_samples with timestamps for the last horizon (optional)
    """
    time_steps, num_stations = Q.shape

    max_start = time_steps - p - np.max(horizons) + 1
    start_indices = np.arange(max_start)

    lag_indices = np.arange(p)
    horizons = np.array(horizons)

    X_idx = start_indices[:, None] + lag_indices[None, :]
    Y_idx = X_idx[:, -1][:, None] + horizons[None, :]

    X = Q[X_idx][:,:,in_stations]
    Y = Q[Y_idx][:,:,out_stations]

    return X, Y, Y_idx

def create_train_val_test(X, Y, time, train_frac=0.7, val_frac=0.15):
    """
    Create train, validation, and test sets for 
    X: (Nt, N_stations, window_size) 
    Y: (Nt, N_targets)

    Returns:
        train, val, test as tuples of dicts:
        ({'x': X_train, 'y': Y_train}, {'x': X_val, 'y': Y_val}, {'x': X_test, 'y': Y_test})
    """
    Nt = X.shape[0]
    train_end = int(Nt * train_frac)
    val_end = int(Nt * (train_frac + val_frac))

    traintimes = time[:train_end]
    valtimes = time[train_end:val_end]
    testtimes = time[val_end:] 

    train = {
        "x": X[:train_end],
        "y": Y[:train_end],
    }
    val = {
        "x": X[train_end:val_end],
        "y": Y[train_end:val_end],
    }
    test = {
        "x": X[val_end:],
        "y": Y[val_end:],
    }

    return train, val, test, (traintimes, valtimes, testtimes)

def trim_to_batches(arr, global_batch_size):
    """
    Trim array so that it is divisible by the global batch size.
    """
    n = arr.shape[0]
    n_batches = n // global_batch_size
    valid_rows = n_batches * global_batch_size
    arr = arr[:valid_rows]

    return arr
    




def batch_iterator(data, batch_size, shuffle=True):
    """
    Generator yielding minibatches from a dataset dict.

    Args:
        data: dict with keys "x" and "y" (NumPy arrays)
        batch_size: int, number of samples per batch
        shuffle: bool, shuffle indices before iterating

    Yields:
        batch: dict with "x" and "y" as jnp arrays (moved to device)
    """
    Nt = data["x"].shape[0]
    indices = np.arange(Nt)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, Nt, batch_size):
        idx = indices[start_idx:start_idx + batch_size]
        batch = {
            "x": jnp.array(data["x"][idx]),
            "y": jnp.array(data["y"][idx]),
        }
        yield batch

def prefetch_batches(generator, prefetch_size=2):
    """
    Wrap a batch generator to prefetch `prefetch_size` batches to device.
    """
    q = queue.Queue(maxsize=prefetch_size)

    def producer():
        for batch in generator:
            # Move batch to device
            batch_device = {k: jax.device_put(v) for k, v in batch.items()}
            q.put(batch_device)
        q.put(None)  # Sentinel to indicate end

    threading.Thread(target=producer, daemon=True).start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch

def reshape_fn(x):
    n_devices = jax.local_device_count()
    n = x.shape[0]
    remainder = n % n_devices
    if remainder > 0:
        # repeat first few examples to pad
        pad_width = n_devices - remainder
        x = jnp.concatenate([x, x[:pad_width]], axis=0)
    return x.reshape(n_devices, -1, *x.shape[1:])

def shard_batch(batch):
    return jax.tree_util.tree_map(reshape_fn, batch)

