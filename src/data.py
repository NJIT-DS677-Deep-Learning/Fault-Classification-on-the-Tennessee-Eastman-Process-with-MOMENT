"""Loading, labeling, splitting and windowing for the extended TEP dataset (Rieth et al., 2017).

Key facts about the data:
- 52 sensors: xmeas_1..xmeas_41 and xmv_1..xmv_11, sampled every 3 minutes.
- 500 simulation runs per fault (faults 1-20) plus 500 fault-free runs.
- Training runs have 500 samples, with the fault switched on after sample 20.
- Testing runs have 960 samples, with the fault switched on after sample 160.
- Runs with the same simulationRun number share a random seed across files,
  so splits must always be made by run number, the same way in every file.
"""

"""Load and prepare the extended TEP dataset (Rieth et al., 2017)."""

from pathlib import Path
import numpy as np
import pandas as pd

SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
# The fault is switched on after this sample in every run
FAULT_START = {"train": 20, "test": 160}

FILES = {
    "train": ("TEP_FaultFree_Training.csv", "TEP_Faulty_Training.csv"),
    "test": ("TEP_FaultFree_Testing.csv", "TEP_Faulty_Testing.csv"),
}

def read_runs(path, runs):
    """Read only the chosen simulation runs from a large CSV, in chunks."""
    parts = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        parts.append(chunk[chunk["simulationRun"].isin(runs)])
    return pd.concat(parts, ignore_index=True)

def load_split(data_dir, split, runs):
    """Load fault-free and faulty runs for 'train' or 'test' and add labels.

    label = 0 before the fault starts, otherwise the fault number.
    """
    normal_file, faulty_file = FILES[split]
    df = pd.concat([
        read_runs(Path(data_dir) / normal_file, runs),
        read_runs(Path(data_dir) / faulty_file, runs),
    ], ignore_index=True)

    df["label"] = np.where(df["sample"] > FAULT_START[split], df["faultNumber"], 0)
    return df

def split_runs(seed=0, n_val=50):
    """Shuffle run numbers 1-500 and hold some out for validation."""
    rng = np.random.default_rng(seed)
    runs = rng.permutation(np.arange(1, 501))
    return runs[n_val:], runs[:n_val]  # train_pool, val_runs

def subset(train_pool, pct):
    """Take the first pct% of the pool, so smaller subsets sit inside larger ones."""
    k = max(1, round(len(train_pool) * pct / 100))
    return train_pool[:k]

def fit_scaler(df):
    """Mean and std of each sensor, using normal data only."""
    normal = df.loc[df["label"] == 0, SENSORS]
    return normal.mean().to_numpy(), normal.std().replace(0, 1).to_numpy()

def make_windows(df, mean, std, window=64, stride=8):
    """Cut each run into windows that end at time t. Each window gets the label of its last sample."""
    X, y = [], []
    for _, run in df.groupby(["faultNumber", "simulationRun"]):
        run = run.sort_values("sample")
        values = ((run[SENSORS].to_numpy() - mean) / std).astype(np.float32)
        labels = run["label"].to_numpy()
        for end in range(window - 1, len(run), stride):
            X.append(values[end - window + 1:end + 1].T)  # shape: (52 sensors, window)
            y.append(labels[end])
    return np.stack(X), np.array(y)
