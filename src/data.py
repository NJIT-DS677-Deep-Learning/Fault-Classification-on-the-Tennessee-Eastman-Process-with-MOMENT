"""Loading, labeling, splitting and windowing for the extended TEP dataset (Rieth et al., 2017).

Key facts about the data:
- 52 sensors: xmeas_1..xmeas_41 and xmv_1..xmv_11, sampled every 3 minutes.
- 500 simulation runs per fault (faults 1-20) plus 500 fault-free runs.
- Training runs have 500 samples, with the fault switched on after sample 20.
- Testing runs have 960 samples, with the fault switched on after sample 160.
- Runs with the same simulationRun number share a random seed across files,
  so splits must always be made by run number, the same way in every file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
ID_COLS = ["faultNumber", "simulationRun", "sample"]
N_RUNS = 500
N_CLASSES = 21  # 0 = normal, 1-20 = faults
FAULT_ONSET = {"train": 20, "test": 160}  # last normal sample in each run

FILES = {
    ("train", "normal"): "TEP_FaultFree_Training.csv",
    ("train", "faulty"): "TEP_Faulty_Training.csv",
    ("test", "normal"): "TEP_FaultFree_Testing.csv",
    ("test", "faulty"): "TEP_Faulty_Testing.csv",
}


def _read_filtered(path: Path, runs: set[int], faults: set[int] | None,
                   chunksize: int = 500_000) -> pd.DataFrame:
    """Read only the requested runs (and faults) from a large CSV, in chunks."""
    dtypes = {c: np.float32 for c in SENSORS}
    dtypes.update({"faultNumber": np.int16, "simulationRun": np.int16, "sample": np.int16})
    parts = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize):
        mask = chunk["simulationRun"].isin(runs)
        if faults is not None:
            mask &= chunk["faultNumber"].isin(faults)
        if mask.any():
            parts.append(chunk.loc[mask])
    if not parts:
        return pd.DataFrame(columns=ID_COLS + SENSORS)
    return pd.concat(parts, ignore_index=True)


def load_split(data_dir: str | Path, split: str, runs: Iterable[int],
               faults: Iterable[int] = range(N_CLASSES)) -> pd.DataFrame:
    """Load the given runs for the given faults from the train or test files.

    Adds a `label` column: the fault number after the fault starts, 0 before it.
    """
    if split not in FAULT_ONSET:
        raise ValueError("split must be 'train' or 'test'")
    data_dir = Path(data_dir)
    runs = set(int(r) for r in runs)
    faults = set(int(f) for f in faults)

    frames = []
    if 0 in faults:
        frames.append(_read_filtered(data_dir / FILES[(split, "normal")], runs, {0}))
    faulty = faults - {0}
    if faulty:
        frames.append(_read_filtered(data_dir / FILES[(split, "faulty")], runs, faulty))

    df = pd.concat(frames, ignore_index=True)
    onset = FAULT_ONSET[split]
    df["label"] = np.where(df["sample"] > onset, df["faultNumber"], 0).astype(np.int16)
    return df.sort_values(ID_COLS, ignore_index=True)


def split_runs(seed: int = 0, n_val: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle training run numbers once and hold out `n_val` runs for validation.

    Returns (train_pool, val_runs). The same run numbers are used for every fault.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(1, N_RUNS + 1))
    return order[n_val:], order[:n_val]


def nested_subset(train_pool: Sequence[int], pct: float) -> np.ndarray:
    """First `pct` percent of the shuffled pool, so smaller subsets sit inside larger ones."""
    k = max(1, int(round(len(train_pool) * pct / 100)))
    return np.asarray(train_pool[:k])


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Scaler":
        """Fit on normal-operation rows only (label 0)."""
        x = df.loc[df["label"] == 0, SENSORS].to_numpy(np.float64)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        return cls(x.mean(axis=0).astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def make_windows(df: pd.DataFrame, scaler: Scaler, window: int = 64,
                 stride: int = 8) -> dict[str, np.ndarray]:
    """Cut each run into sliding windows that end at time t (no peeking ahead).

    Each window gets the label of its last sample.
    Returns X with shape (n_windows, 52, window) plus labels and run metadata.
    """
    xs, ys, faults, runs, ends = [], [], [], [], []
    for (fault, run), g in df.groupby(["faultNumber", "simulationRun"], sort=True):
        x = scaler.transform(g[SENSORS].to_numpy(np.float32))
        lab = g["label"].to_numpy()
        smp = g["sample"].to_numpy()
        for end in range(window - 1, len(g), stride):
            xs.append(x[end - window + 1:end + 1].T)
            ys.append(lab[end])
            faults.append(fault)
            runs.append(run)
            ends.append(smp[end])
    if not xs:
        raise ValueError("No windows created: runs are shorter than the window length")
    return {
        "X": np.stack(xs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "fault": np.asarray(faults, dtype=np.int16),
        "run": np.asarray(runs, dtype=np.int16),
        "end_sample": np.asarray(ends, dtype=np.int16),
    }
