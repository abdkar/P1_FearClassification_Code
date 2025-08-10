"""Loader for the cleaned variable-trial-count dataset in `clean_simulated_data/`.

Structure expected:
clean_simulated_data/
  participants/
    HF_201/
      high_fear/*.csv
    LF_202/
      low_fear/*.csv
  control_data/ (optional controls – not yet integrated)

Usage (LOPOCV example):
    from src.data.clean_data_loader import CleanDataLoader
    loader = CleanDataLoader(root="clean_simulated_data", high_dir="high_fear", low_dir="low_fear")
    X_train, y_train, X_test, y_test = loader.get_lopocv_split("HF_201")

Each CSV is assumed to hold a 2D array (timesteps x features). All files must share the same shape.
High fear label = 1, Low fear label = 0.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

class CleanDataLoader:
    def __init__(self, root: str = "clean_simulated_data", high_dir: str = "high_fear", low_dir: str = "low_fear", include_controls: bool = True):
        self.root = Path(root)
        self.participants_dir = self.root / "participants"
        self.high_dir = high_dir
        self.low_dir = low_dir
        self.include_controls = include_controls
        self.control_dirs = [
            self.root / "control_data" / "healthy" / "C5" / "Med_C4_D",
            self.root / "control_data" / "healthy" / "C5" / "Med_C4_ND",
            self.root / "control_data" / "athletes" / "C5" / "Med_C4_D",
            self.root / "control_data" / "athletes" / "C5" / "Med_C4_ND",
        ]
        if not self.participants_dir.is_dir():
            raise FileNotFoundError(f"Participants directory not found: {self.participants_dir}")
        self.participants = sorted([p.name for p in self.participants_dir.iterdir() if p.is_dir()])
        # Track how many control samples were appended in the last split (for exclusion in class weights)
        self._last_control_count: int = 0

    def list_participants(self) -> List[str]:
        return self.participants

    def _load_trials_dir(self, path: Path) -> np.ndarray:
        if not path.is_dir():
            return np.empty((0,))
        csvs = sorted([f for f in path.iterdir() if f.suffix == ".csv"])
        trials = []
        for f in csvs:
            try:
                df = pd.read_csv(f)
                trials.append(df.values.astype("float32"))
            except Exception as e:
                print(f"Skip {f}: {e}")
        if not trials:
            return np.empty((0,))
        # Ensure consistent shapes
        base_shape = trials[0].shape
        filtered = [t for t in trials if t.shape == base_shape]
        if len(filtered) != len(trials):
            print(f"Warning: dropped {len(trials)-len(filtered)} trials with mismatched shape in {path}")
        return np.stack(filtered, axis=0)

    def _load_participant(self, participant: str) -> Tuple[np.ndarray, np.ndarray]:
        p_dir = self.participants_dir / participant
        hf_path = p_dir / self.high_dir
        lf_path = p_dir / self.low_dir
        hf = self._load_trials_dir(hf_path)
        lf = self._load_trials_dir(lf_path)
        X_list = []
        y_list = []
        if hf.size:
            X_list.append(hf); y_list.append(np.ones((hf.shape[0],), dtype="int32"))
        if lf.size:
            X_list.append(lf); y_list.append(np.zeros((lf.shape[0],), dtype="int32"))
        if not X_list:
            return np.empty((0,)), np.empty((0,))
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    def _load_control_trials(self):
        if not self.include_controls:
            return np.empty((0,))
        collected = []
        for d in self.control_dirs:
            if d.is_dir():
                arr = self._load_trials_dir(d)
                if arr.size:
                    collected.append(arr)
        if not collected:
            return np.empty((0,))
        return np.concatenate(collected, axis=0)

    def get_lopocv_split(self, test_participant: str):
        """Return train/test split for given participant.
        Control trials (if enabled) are appended ONLY to the training set as augmentation
        (label 0) and never appear in the test set.
        Tracks the number of appended control samples in `self._last_control_count`.
        """
        if test_participant not in self.participants:
            raise ValueError(f"Participant '{test_participant}' not found. Available: {self.participants[:5]} ...")
        self._last_control_count = 0
        X_train_list, y_train_list = [], []
        X_test, y_test = self._load_participant(test_participant)
        for p in self.participants:
            if p == test_participant:
                continue
            Xp, yp = self._load_participant(p)
            if Xp.size:
                X_train_list.append(Xp); y_train_list.append(yp)
        controls = self._load_control_trials()
        if controls.size:
            X_train_list.append(controls)
            y_train_list.append(np.zeros((controls.shape[0],), dtype="int32"))  # label 0 (low fear)
            self._last_control_count = controls.shape[0]
        if not X_test.size:
            print(f"Warning: test participant {test_participant} has no data")
        if X_train_list:
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
        else:
            X_train = np.empty((0,)); y_train = np.empty((0,))
        return X_train, y_train, X_test, y_test

    def compute_class_weights(self, y_train, exclude_controls: bool = False):
        """Compute inverse-frequency class weights.

        Parameters
        ----------
        y_train : np.ndarray
            Training labels (including controls if augmentation used).
        exclude_controls : bool, default False
            If True and controls were appended, exclude those samples from the
            frequency calculation so they act only as augmentation and do not
            shift the weighting.
        """
        if y_train.size == 0:
            return {0:1.0, 1:1.0}
        labels_for_weights = y_train
        if exclude_controls and self._last_control_count > 0 and self._last_control_count < y_train.size:
            labels_for_weights = y_train[:-self._last_control_count]
        unique, counts = np.unique(labels_for_weights, return_counts=True)
        total = labels_for_weights.shape[0]
        weights = {int(u): float(total/(len(unique)*c)) for u,c in zip(unique, counts)}
        return weights

    def summary(self):
        rows = []
        for p in self.participants:
            Xp, yp = self._load_participant(p)
            rows.append({
                "participant": p,
                "trials_total": int(yp.shape[0]),
                "trials_high": int((yp==1).sum()),
                "trials_low": int((yp==0).sum())
            })
        return pd.DataFrame(rows)

if __name__ == "__main__":
    loader = CleanDataLoader()
    print("Participants:", loader.list_participants()[:5], "...")
    df = loader.summary()
    print(df.head())
    Xtr, ytr, Xte, yte = loader.get_lopocv_split(loader.list_participants()[0])
    print("Train shape:", Xtr.shape, "Test shape:", Xte.shape)
