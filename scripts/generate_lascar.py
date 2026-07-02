# scripts/generate_lascar.py
from pathlib import Path
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import (
    load_config, validate_dirs,
    load_synthesizer, model_exists, coerce_numeric, get_paths
)

# ── fit ───────────────────────────────────────────────────────────────────────
def fit_lascar(real_data_dir, models_dir):
    model_path    = Path(models_dir) / "lascar_synthesizer.pkl"
    metadata_path = Path(models_dir) / "lascar_metadata.json"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Fitting Lascar model from real data...")
    files = list(Path(real_data_dir).rglob("*.txt"))
    data  = pd.concat([
        pd.read_csv(f, usecols=[1, 2], names=['datetime', 'co'],
                    skiprows=1)
        for f in files
    ]).reset_index(drop=True)

    data = data[['co']]
    data = coerce_numeric(data).dropna()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.save_to_json(str(metadata_path))
    print(f"✓ Metadata saved to {metadata_path}")

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    synthesizer.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    return synthesizer

def load_lascar(models_dir):
    return load_synthesizer(Path(models_dir) / "lascar_synthesizer.pkl")

# ── filename ──────────────────────────────────────────────────────────────────
def make_lascar_filename(sensor_id, serial, start):
    """2021_11_18_1000000009_PER_009.txt"""
    ts = pd.Timestamp(start)
    return f"{ts.strftime('%Y_%m_%d')}_{serial}_PER_{sensor_id}.txt"

# ── write ─────────────────────────────────────────────────────────────────────
def write_lascar(df, sensor_id, serial, start, output_dir):
    timestamps = pd.date_range(start=start, periods=len(df), freq='60s')
    n_rows     = len(df)
    header_1   = f"{n_rows:03d},Time,CO(ppm),Serial Number,Sensor Life Expiry,Overrange Exposure"

    rows = [header_1]
    for i, (ts, co) in enumerate(zip(timestamps, df['co']), start=1):
        if i == 1:
            rows.append(
                f"{i},{ts.strftime('%Y-%m-%d %H:%M:%S')},{co:.1f},{serial},2/6/2024,No"
            )
        else:
            rows.append(f"{i},{ts.strftime('%Y-%m-%d %H:%M:%S')},{co:.1f}")

    fname    = make_lascar_filename(sensor_id, serial, start)
    out_path = Path(output_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(rows))
    return out_path

# ── edge cases ────────────────────────────────────────────────────────────────
def inject_edge_cases(df, rng):
    # negative CO (test flag)
    neg_idx = rng.integers(10, len(df) - 10)
    df.loc[neg_idx, 'co'] = -0.1

    # high CO spike (test flag)
    spike_idx = rng.integers(20, len(df) - 20)
    df.loc[spike_idx, 'co'] = 210.0

    # sustained zeros (filter obstruction)
    zero_start = rng.integers(50, len(df) - 30)
    df.loc[zero_start:zero_start+15, 'co'] = 0.0

    return df

# ── campaign ──────────────────────────────────────────────────────────────────
def generate_lascar_campaign(real_data_dir, output_dir, models_dir,
                              n_sensors=3, n_days=7,
                              start='2022-09-15 08:00:00',
                              seed=42, force_refit=False):
    """
    Generate synthetic Lascar CO data for multiple sensors.

    Parameters
    ----------
    real_data_dir : str  — path to real Lascar data for model fitting
    output_dir    : str  — path to write synthetic files
    models_dir    : str  — path to model cache directory
    n_sensors     : int  — number of synthetic sensors to generate
    n_days        : int  — deployment duration per sensor in days
    start         : str  — campaign start datetime
    seed          : int  — random seed for reproducibility
    force_refit   : bool — refit model even if cached version exists
    """
    real_data_dir, output_dir = validate_dirs(real_data_dir, output_dir,
                                               ext='.txt')
    if force_refit or not model_exists(models_dir, 'lascar'):
        synthesizer = fit_lascar(real_data_dir, models_dir)
    else:
        synthesizer = load_lascar(models_dir)

    n_records = n_days * 24 * 60   # 60s resolution
    outputs   = []

    for i in range(n_sensors):
        sensor_id = f"{i+1:03d}"
        serial    = f"{1000000000 + i + 1:010d}"
        rng       = np.random.default_rng(seed + i)
        synthetic = synthesizer.sample(num_rows=n_records)
        synthetic = coerce_numeric(synthetic)
        synthetic['co'] = synthetic['co'].clip(lower=0.0).round(1)
        synthetic = inject_edge_cases(synthetic, rng)

        out = write_lascar(synthetic, sensor_id, serial,
                           start=pd.Timestamp(start),
                           output_dir=output_dir)
        outputs.append(out)
        print(f"✓ Sensor {i+1}/{n_sensors}: {out.name}")

    print(f"\n✓ Generated {n_sensors} sensors × {n_days} days "
          f"({n_records} records each)")
    return outputs

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    paths  = get_paths(config)
    cfg    = config['campaign']
    inst   = config['instruments']['lascar']

    if not inst['enabled']:
        print("Lascar disabled in config — skipping")
    else:
        generate_lascar_campaign(
            real_data_dir = paths['real_data'] / 'lascar',
            output_dir    = paths['output']    / 'lascar',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['force_refit'],
        )