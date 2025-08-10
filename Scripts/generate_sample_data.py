"""Generate structured synthetic sample dataset mirroring full project layout.

Structure produced (for Med only by default):
  <out>/Sim_<SUBJECT_ID>/<HF|LF>/Train/Med_Train/<SUBJECT_ID>_<hop>.csv
                                   /Test/Med_Test/<SUBJECT_ID>_<hop>.csv
Optionally add Lat land type (adds Lat_Train / Lat_Test). Hop sets:
  Med hops: 1,3,5,7,9,11,13,15,17,19
  Lat hops: 0,2,4,6,8,10,12,14,16,18

Each file: 101 timesteps x 24 biomechanical features with realistic-ish
sinusoidal patterns + noise. High fear (HF_) subjects have elevated
amplitude & noise vs low fear (LF_) subjects.

Usage example (minimal subset of 4 subjects, Med only):
  python Scripts/generate_sample_data.py \
      --out clean_simulated_data/sample_sim \
      --hf 2 --lf 2 --tests 2 --land Med

Add both Med & Lat:
  python Scripts/generate_sample_data.py --land Med Lat

Include small control set:
  python Scripts/generate_sample_data.py --controls --hf 1 --lf 1

NOTE: Designed for a lightweight public sample; not the full synthetic generator.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# Constants
TIMESTEPS = 101
FEATURES = 24
MED_HOPS = [1,3,5,7,9,11,13,15,17,19]
LAT_HOPS = [0,2,4,6,8,10,12,14,16,18]

FEATURE_NAMES_MED = [
    'ankle_mom_X_Med','ankle_mom_Y_Med','ankle_mom_Z_Med',
    'foot_X_Med','foot_Y_Med','foot_Z_Med',
    'hip_mom_X_Med','hip_mom_Y_Med','hip_mom_Z_Med',
    'hip_X_Med','hip_Y_Med','hip_Z_Med',
    'knee_mom_X_Med','knee_mom_Y_Med','knee_mom_Z_med',
    'knee_X_med','knee_Y_med','knee_Z_med',
    'pelvis_X_med','pelvis_Y_med','pelvis_Z_med',
    'thorax_X_Med','thorax_Y_Med','thorax_Z_Med'
]
FEATURE_NAMES_LAT = [
    'ankle_mom_X_Lat','ankle_mom_Y_Lat','ankle_mom_Z_Lat',
    'foot_X_Lat','foot_Y_Lat','foot_Z_Lat',
    'hip_mom_X_Lat','hip_mom_Y_Lat','hip_mom_Z_Lat',
    'hip_X_Lat','hip_Y_Lat','hip_Z_Lat',
    'knee_mom_X_Lat','knee_mom_Y_Lat','knee_mom_Z_med',
    'knee_X_med','knee_Y_med','knee_Z_med',
    'pelvis_X_med','pelvis_Y_med','pelvis_Z_med',
    'thorax_X_Lat','thorax_Y_Lat','thorax_Z_Lat'
]


def _extract_ints(s: str) -> int:
    nums = re.findall(r'\d+', s)
    return int(nums[0]) if nums else 0


def generate_signal(name: str, fear_level: str):  # return ndarray (untyped for simplicity)
    # Base settings by type
    if 'mom' in name:
        amp = np.random.uniform(0.5, 2.0)
        freq = np.random.uniform(0.1, 0.3)
        noise = 0.1
    elif any(x in name for x in ['_X_', '_Y_', '_Z_']):
        amp = np.random.uniform(10, 45)
        freq = np.random.uniform(0.05, 0.2)
        noise = 2.0
    else:
        amp = np.random.uniform(1.0, 10.0)
        freq = np.random.uniform(0.1, 0.4)
        noise = 0.5
    if fear_level == 'HF':
        amp *= np.random.uniform(1.2, 1.7)
        noise *= np.random.uniform(1.4, 1.9)
    else:  # LF
        amp *= np.random.uniform(0.7, 1.05)
        noise *= np.random.uniform(0.5, 0.9)
    t = np.linspace(0,1,TIMESTEPS)
    main = amp * np.sin(2*np.pi*freq*t)
    secondary = (amp*0.3) * np.sin(4*np.pi*freq*t + np.pi/4)
    envelope = np.exp(-((t-0.5)**2)/(2*0.3**2))
    signal = (main + secondary) * envelope
    drift = np.random.uniform(-0.1,0.1)*t
    noise_arr = np.random.normal(0, noise, TIMESTEPS)
    return signal + drift + noise_arr


def generate_trial_df(land: str, fear_level: str) -> pd.DataFrame:
    names = FEATURE_NAMES_MED if land == 'Med' else FEATURE_NAMES_LAT
    data = {fname: generate_signal(fname, fear_level) for fname in names}
    return pd.DataFrame(data)


def subject_ids(n_hf: int, n_lf: int) -> List[str]:
    ids = [f'HF_{201 + i*2}' for i in range(n_hf)]
    ids += [f'LF_{202 + i*2}' for i in range(n_lf)]
    return ids


def split_hops(hops: List[int], seed: int):
    rng = np.random.default_rng(seed)
    order = hops.copy()
    rng.shuffle(order)
    k = int(len(order)*0.7)
    return order[:k], order[k:]


def write_trial(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def generate_subject(root: Path, subj: str, land_types: List[str], tests: int):
    fear = 'HF' if subj.startswith('HF_') else 'LF'
    base = root / f'Sim_{subj}' / fear
    numeric = _extract_ints(subj)
    for land in land_types:
        hops = MED_HOPS if land == 'Med' else LAT_HOPS
        for test_n in range(1, tests+1):
            train_h, test_h = split_hops(hops, seed=42 + numeric + test_n)
            train_dir = base / 'Train' / f'{land}_Train'
            test_dir = base / 'Test' / f'{land}_Test'
            for h in train_h:
                df = generate_trial_df(land, fear)
                write_trial(df, train_dir / f'{subj}_{h}.csv')
            for h in test_h:
                df = generate_trial_df(land, fear)
                write_trial(df, test_dir / f'{subj}_{h}.csv')


def generate_controls(root: Path, land_types: List[str], small: bool):
    if not small:
        return
    # Minimal control example (few files per path)
    control_sets = [
        'Control_KK_Healthy_24_1/C5',
        'Control_KK_Athletes_24_1/C5'
    ]
    for land in land_types:
        hops = MED_HOPS if land == 'Med' else LAT_HOPS
        for cset in control_sets:
            for side in ['D','ND']:
                base = root / cset / f'{land}_C4_{side}'
                base.mkdir(parents=True, exist_ok=True)
                for i in range(3):  # few illustrative files
                    hop = np.random.choice(hops)
                    df = generate_trial_df(land, 'LF')  # treat as normal
                    write_trial(df, base / f'Control_{i+1:03d}_{hop}.csv')


def summarize(root: Path):
    total = 0
    for _ in root.rglob('*.csv'):
        total += 1
    print(f'Created {total} CSV files under {root}')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('clean_simulated_data/sample_sim'))
    ap.add_argument('--hf', type=int, default=2, help='Number of HF subjects')
    ap.add_argument('--lf', type=int, default=2, help='Number of LF subjects')
    ap.add_argument('--tests', type=int, default=2, help='Test replicates per subject')
    ap.add_argument('--land', nargs='+', default=['Med'], choices=['Med','Lat'])
    ap.add_argument('--controls', action='store_true', help='Include minimal control example')
    ap.add_argument('--seed', type=int, default=123)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    root: Path = args.out
    ids = subject_ids(args.hf, args.lf)
    for s in ids:
        print(f'Generating subject {s} ...')
        generate_subject(root, s, args.land, args.tests)
    generate_controls(root, args.land, args.controls)
    summarize(root)
    print('Done.')

if __name__ == '__main__':
    main()
