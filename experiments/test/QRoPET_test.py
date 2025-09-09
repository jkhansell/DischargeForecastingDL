# global imports

# Data analysis libraries
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

# System handling libraries
import os
import sys
import socket

# Deep Learning and accelerated computing libraries
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from jax.experimental import multihost_utils

# Optimization libraries
import optax

# Checkpointing libraries
import orbax.checkpoint as ocp
from flax.training import orbax_utils

# Local utilities
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

# Model implementation
from models.QRoPET import QRoPETRegressor


# We'll be studying the Guadalupe River in Kerr County, TX the USGS sites are the following
# Previously exploring the data some stations report the discharge variable in another column of the data frame

sites = ["08165300", "08165500", "08166000", "08166140", "08166200"]
nan_sites = ["08166140", "08166200"]

def test_model():
    Q, cols, time = feature_engineering("./data/Q_raw.csv", sites, nan_sites)

    in_stations = np.array([i for i in range(Q.shape[1])])
    out_stations = np.array([cols["08166200"]])

    time_window = 128        # 15*4*64 hours of context 
    horizons = (4*np.array([2, 4, 8, 12, 24]))  # Multi horizon prediction [2, 4, 8, 16, 32] h in advance


    X, Y, Y_idx = build_multi_horizon_dataset(Q, in_stations, out_stations, time_window, horizons)
    train, val, test, times = create_train_val_test(X, Y, time)
    print(f"Time Window: {time_window} | Horizons: {horizons}")

    # initialize distributed environment
    visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]          

    jax.distributed.initialize(
        local_device_ids=visible_devices
    )

    print(f"[JAX] ProcID: {jax.process_index()}")
    print(f"[JAX] Local devices: {jax.local_devices()}")
    print(f"[JAX] Global devices: {jax.devices()}")
    
    # Devices
    n_local_devices = jax.local_device_count()   # 4 GPUs per host
    n_total_devices = jax.device_count()         # 8 total
    print(f"Host {jax.process_index()} sees {n_local_devices} local devices")

    per_device_batch_size = 64
    batch_size = per_device_batch_size * jax.device_count()
    for split in [train, val, test]:
        split["x"] = trim_to_batches(split["x"], per_device_batch_size)
        split["y"] = trim_to_batches(split["y"], per_device_batch_size)

    num_epochs = 15

    in_features = train["x"].shape[-1]
    out_features = train["y"].shape[-2]
    hidden_size = 64
    quantiles = jnp.array([0.05, 0.5, 0.95])
    
    key = jax.random.PRNGKey(123)
    x = jnp.zeros((batch_size, time_window, in_features))

    model = QRoPETRegressor(
        d_model = hidden_size, 
        num_heads = 4, 
        mlp_dim = 128, 
        num_layers = 4,
        out_features = out_features, 
        n_quantiles = len(quantiles)
    )

    params = model.init(key, x)
    
    steps_per_epoch = len(train["x"]) // batch_size
    total_steps = num_epochs * steps_per_epoch
    fractions = [0.2, 0.3, 0.5]  # adjust as needed
    decay_steps = [int(f * total_steps) for f in fractions]

    cosine_kwargs = [
        dict(init_value=1e-6, peak_value=1e-3, warmup_steps=500,
            decay_steps=decay_steps[0], end_value=1e-4),
        dict(init_value=1e-6, peak_value=5e-4, warmup_steps=200,
            decay_steps=decay_steps[1], end_value=1e-5),
        dict(init_value=1e-6, peak_value=5e-5, warmup_steps=0,
            decay_steps=decay_steps[2], end_value=5e-6),
    ]

    schedule = optax.sgdr_schedule(cosine_kwargs)

    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-5)
    
    async_checkpointer = ocp.AsyncCheckpointer(
        ocp.StandardCheckpointHandler(), timeout_secs=60
    )

    mesh = Mesh(jax.devices(), ('batch',))

    # Partition specs
    param_spec = P()        # fully replicated
    param_sharding = NamedSharding(mesh, param_spec)

    fmt = '%Y_%m_%d'
    checkpoint_dir = os.path.abspath("./checkpoints")
    checkpoint_dir = os.path.join(checkpoint_dir, f"QRoPET/model_{datetime.today().strftime(fmt)}")
    #checkpoint_dir="/work/jovillalobos/hidrologia/DischargeForecastingDL/checkpoints/QRoPET/model_2025_09_05/"

    options = ocp.CheckpointManagerOptions(max_to_keep=2)

    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, params)

    def set_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        return x.update(sharding=param_sharding)

    changed_params = jax.tree.map(set_sharding, abstract_state)

    with ocp.CheckpointManager(checkpoint_dir, async_checkpointer, options) as mngr:
        params = mngr.restore(0, args=ocp.args.StandardRestore(changed_params))
        mngr.wait_until_finished()

    state = ModelTrainState.create(apply_fn=model.apply, params=params,tx=tx)

    horizon_weights = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]) 
    horizon_weights /= jnp.mean(horizon_weights)
    loss_fn = lambda x,y: quantile_loss_complex(
        x, y, quantiles, horizon_weights, 
        crossing_penalty_coef=0.2, cov_weight=0.75, k=100, mae_coef=1.0
    )
    
    # Mesh
    mesh = Mesh(jax.devices(), ('batch',))

    # Partition specs
    param_spec = P()        # fully replicated
    in_spec = P('batch', None, None)                    # shard batch
    out_spec = P('batch', None)   

    in_batch_sharding = NamedSharding(mesh, in_spec)
    out_batch_sharding = NamedSharding(mesh, out_spec)
    param_sharding = NamedSharding(mesh, param_spec)

    p_eval_step = jax.jit(
        eval_step(loss_fn),
        in_shardings=(param_sharding, in_batch_sharding),
        out_shardings=(param_sharding, out_batch_sharding)         # loss, preds
    )

    # Testing phase
    test_gen = batch_iterator(test, batch_size=batch_size, shuffle=False)
    test_prefetch = prefetch_batches(test_gen, prefetch_size=32)

    test_loss = []
    medians = []
    lows = []
    highs = []
    truths = []

    for batch in test_prefetch:
        loss, preds = p_eval_step(state, batch)
        test_loss.append(np.mean(loss))
        
        # append for graphing

        truths.append(np.asarray(batch["y"].reshape(-1, batch["y"].shape[-2])))
        
        lows.append(np.asarray(preds[..., 0].reshape(-1, preds.shape[1])))
        medians.append(np.asarray(preds[..., 1].reshape(-1, preds.shape[1])))
        highs.append(np.asarray(preds[..., 2].reshape(-1, preds.shape[1])))

    # Test data analysis

    medians = np.concatenate(medians, axis=0)
    lows = np.concatenate(lows, axis=0)
    highs = np.concatenate(highs, axis=0)


    truths = np.concatenate(truths, axis=0)
    test_loss = np.array(test_loss)

    medians = 10**medians
    lows = 10**lows
    highs = 10**highs
    truths = 10**truths

    #nquant = np.quantile(test_loss, 0.99)
    #test_loss = test_loss[test_loss < nquant] 

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(test_loss, bins=15, label="Test Loss Histogram")
    fig.legend()
    fig.savefig("images/QRoPET/QRoPET_Test_Loss.png")
    plt.close()

    num_horizons = medians.shape[1]

    for i in range(num_horizons):
        y_med = medians[:, i]
        y_low = lows[:, i]
        y_high = highs[:, i]
        y_true = truths[:, i]

        # ----- Median metrics -----
        rmse = np.sqrt(np.mean((y_med - y_true)**2))
        mae = np.mean(np.abs(y_med - y_true))
        r2 = 1 - np.sum((y_true - y_med)**2) / np.sum((y_true - np.mean(y_true))**2)
        mbe = np.mean(y_med - y_true)
        
        # ----- Uncertainty metrics -----
        inside_interval = (y_true >= y_low) & (y_true <= y_high)
        picp = inside_interval.mean()
        mpiw = np.mean(y_high - y_low)
        crps_proxy = np.mean(np.abs(y_true - y_med))  # simple CRPS proxy
        
        # ----- Print nicely -----
        print(f"Horizon {horizons[i]//4}h:")
        print(f"  Median Metrics -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, MBE: {mbe:.3f}")
        print(f"  Uncertainty Metrics -> PICP: {picp*100:.2f}%, MPIW: {mpiw:.3f}")
        print("-"*60)

        fig, ax = plt.subplots(figsize=(8,8))
        
        ax.scatter(y_true, y_med, alpha=0.4, color='tab:blue')
    
        # Ideal diagonal
        min_val = min(y_true.min(), y_med.min())
        max_val = max(y_true.max(), y_med.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--')
        ax.set_xlabel("True Discharge [m^3/s]")
        ax.set_ylabel("Forecasted Discharge [m^3/s]")
        ax.grid(alpha=0.3)
        ax.set_title(f"Horizon {horizons[i]//4}h")
        ax.set_aspect('equal')

        fig.savefig(f"images/QRoPET/ScatterPlots_{i}.png", dpi=250)
        plt.close()

    print("here")

    for i in range(num_horizons):
        y_med = medians[:, i]
        y_low = lows[:, i]
        y_high = highs[:, i]
        y_true = truths[:, i]
        fig, ax = plt.subplots(figsize=(14,5))

        # Ground truth
        ax.plot(y_true[70000:90000], label="Ground Truth", linewidth=2.5, linestyle="--", color="black")

        # Prediction median
        ax.plot(y_med[70000:90000], label="Prediction (Mean)", linewidth=1, color="red")

        # Uncertainty band
        ax.fill_between(
            np.arange(y_med[70000:90000].shape[0]),
            y_low[70000:90000], y_high[70000:90000],
            alpha=0.25, color="red", label="90% Prediction Interval"
        )

        ax.set_xlabel("Time point")
        ax.set_ylabel("Flow Discharge [m^3/s]")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)

        fig.suptitle(f"QRoPET Forecast - Horizon {horizons[i]//4}h")
        fig.legend(loc="upper right")
        fig.tight_layout()

        fig.savefig(f"images/QRoPET/predictions_{i}.png", dpi=300)
        plt.close()