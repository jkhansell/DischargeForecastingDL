import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from datetime import datetime

from utils.datautils import get_data, build_multi_horizon_dataset, fill_nans_with_random_walk


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


# Define a mapping from window suffix to temporal length (in increasing order)
temporal_order = {
    "": 0,      # original column
    "2h": 1,
    "6h" : 2,
    "12h": 3,
    "1D": 4,
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

def EDA():
    Q_raw, Q_filled = get_data("./data/Q_raw.csv", sites, nan_sites)
    for col in sites: 

        fig, ax = plt.subplots(figsize=(14,5))
        Q_raw[col].plot(ax=ax)
        ax.set_yscale("log")
        fig.savefig(f"RawQ{col}.png")
        plt.close()
            
        fig, ax = plt.subplots(figsize=(14,5))
        Q_filled[col].plot(ax=ax)
        ax.set_yscale("log")
        fig.savefig(f"InterpolatedQ{col}.png")
        plt.close()

    print(Q_filled[Q_filled.isnull().any(axis=1)])
    Q_filled = Q_filled.dropna()
    print(Q_filled)

def OLDEDA(): 
    Q = pl.from_pandas(Q)

    # Feature engineering, augmenting dataset with
    # 1 week MA
    # 2 week MA
    # 1 month MA
    # 3 month MA
    # 6 month MA

    # 15-min resolution -> integer number of rows
    windows = {
        "2h": 8,
        "6h" : 24,
        "12h": 48,
        "1D": 96,
    #    "1W": 672,
    #    "2W": 1344,
    #    "1M": 2880,
    #    "3M": 8640,
    #    "6M": 17280
    }

    time = Q.select("datetime")
    Q = Q.select(pl.col(pl.Float64))

    diff_Q = Q.select([
        pl.col(c).diff().alias(c).abs()  # difference for each column
        for c in Q.columns
    ])

    diff_abs = diff_Q.select([
        pl.when(pl.col(c) != 0).then(pl.col(c).abs()).otherwise(None).alias(c)
        for c in diff_Q.columns
    ])

    sensor_estimate = diff_abs.select([
        pl.col(c).drop_nulls().quantile(0.25).alias(c)
        for c in diff_abs.columns
    ])

    resolutions = {}    

    for c in diff_Q.columns:
        sensor_threshold = sensor_estimate[c]  # a float

        # drop nulls first
        series = diff_abs[c].drop_nulls()

        # create a boolean Series mask
        mask = series < sensor_threshold

        # use .filter() with the boolean Series
        data = series.filter(mask).to_numpy()

        counts, bin_edges = np.histogram(data, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = bin_centers > 0.005

        # among those bins, find the one with max counts
        idx = np.argmax(counts[mask])
        best_bin_center = bin_centers[mask][idx]
        best_bin_count = counts[mask][idx]
        print(f"Station {c} Most common error > 0.005:", best_bin_center, "with count", best_bin_count)
        resolutions[c] = best_bin_center
        
        plt.figure()
        plt.hist(data, bins=50, edgecolor='k')
        plt.title(f"Histogram of consecutive differences: {c}")
        plt.xlabel("Difference")
        plt.ylabel("Count")
        plt.savefig(f"images/resolution_hist{c}.png")
        plt.close()

    # Assume your DataFrame is `df` and you want to dequantize all columns

    Q = Q.with_columns([
        (pl.col(c) + pl.Series(np.random.uniform(0, resolutions[c], size=Q.height))).alias(c)
        for c in Q.columns
    ])

    # Create a new DataFrame for all new features
    ma_features = []
    diff_features = []
    cols = {}
    for i,col in enumerate(Q.columns):
        for key, value in windows.items():
            ma_features.append(
                pl.col(col).rolling_median(window_size=value, min_samples=1).alias(f"{col}_MA_{key}")
            )
        cols[col] = i


    # Add rolling means as new columns0
    Q = Q.with_columns(ma_features)
    Q = Q.select(sorted(Q.columns, key=sort_key))

    time = time["datetime"].str.to_datetime("%Y-%m-%d %H:%M:%S%z")

    for site in sites:
        plt.figure(figsize=(14, 6))
        
        # Find all columns for this site
        site_cols = [c for c in Q.columns if c.startswith(site)]
        
        for col in site_cols:
            plt.plot(time[-6000:-4500], Q[col][-6000:-4500], label=col)


        # Tell matplotlib to put major ticks at each year
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M"))
        plt.title(f"Site {site} Discharge with Rolling Medians")
        plt.xlabel("Time")
        plt.ylabel("Discharge (cfs)")
        plt.legend(loc="upper right", ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"images/rolling_{site}.png")
        plt.close()

    # --- Plot each rolling window across all sites ---
    for win in temporal_order.keys():
        if win == "":  # skip raw
            continue
        plt.figure(figsize=(14, 6))
        for site in sites:
            col = f"{site}_MA_{win}"
            if col in Q.columns:
                plt.plot(time[::1000], Q[col][::1000], label=site)
        
        plt.title(f"Rolling Median {win} Across All Sites")
        plt.xlabel("Time")
        plt.ylabel("Discharge (cfs)")
        plt.legend(loc="upper right")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"images/rolling_sites_{win}.png")
        plt.close()

if __name__ == "__main__":
    EDA()