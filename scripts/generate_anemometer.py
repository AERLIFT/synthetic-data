# scripts/generate_anemometer.py
from pathlib import Path
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from utils import (
    load_config, validate_dirs, find_data_start,
    load_synthesizer, model_exists, coerce_numeric, get_paths
)

# ── fit ───────────────────────────────────────────────────────────────────────
def fit_anemometer(real_data_dir, models_dir):
    model_path    = Path(models_dir) / "anemometer_synthesizer.pkl"
    metadata_path = Path(models_dir) / "anemometer_metadata.json"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Fitting anemometer model from real data...")
    files = list(Path(real_data_dir).rglob("*.txt"))
    data  = pd.concat([
        pd.read_csv(f, skiprows=6, header=None,
                    usecols=[1], names=['air_flow'])
        for f in files
    ]).reset_index(drop=True)
    data = coerce_numeric(data)

    metadata = Metadata.detect_from_dataframe(data)
    metadata.save_to_json(str(metadata_path))
    print(f"✓ Metadata saved to {metadata_path}")

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    synthesizer.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    return synthesizer

def load_anemometer(models_dir):
    return load_synthesizer(Path(models_dir) / "anemometer_synthesizer.pkl")

# ── filename ──────────────────────────────────────────────────────────────────
def make_anemometer_filename(sensor_id, start):
    """CP000001_E001_2022-09-15 8-00.txt"""
    ts = pd.Timestamp(start)
    return f"{sensor_id}_{ts.year}-{ts.month:02d}-{ts.day:02d} {ts.hour}-{ts.minute:02d}.txt"

# ── write ─────────────────────────────────────────────────────────────────────
def write_anemometer(values, sensor_id, start, output_dir):
    timestamps = pd.date_range(start=start, periods=len(values), freq='60s')
    max_val    = values.max()
    min_val    = values.min()
    mean_val   = values.mean()
    max_ts     = timestamps[values.argmax()].strftime('%d-%m-%Y,%H:%M:%S')
    min_ts     = timestamps[values.argmin()].strftime('%d-%m-%Y,%H:%M:%S')
    start_str  = timestamps[0].strftime('%d-%m-%Y,%H:%M:%S')

    header = (
        f"TXT Data File\n"
        f"StartTime: {start_str}\n"
        f"Max.: {max_val:.2f} @ {max_ts}  m/s\n"
        f"Min.: {min_val:.2f} @ {min_ts}  m/s\n"
        f"Average: {mean_val:.2f}  m/s\n"
        f"SampleRate: 60 second\n"
        f"\n"
    )
    rows = [
        f"{i}, {val:.2f}, m/s, {ts.strftime('%d-%m-%Y')},{ts.strftime('%H:%M:%S')}"
        for i, (ts, val) in enumerate(zip(timestamps, values), start=1)
    ]

    fname    = make_anemometer_filename(sensor_id, start)
    out_path = Path(output_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(header)
        f.write('\n'.join(rows))
    return out_path

# ── edge cases ────────────────────────────────────────────────────────────────
def inject_edge_cases(values, rng):
    start = rng.integers(50, len(values) - 30)
    values[start:start + 20] = 0.0
    neg_idx = rng.choice(len(values), size=2, replace=False)
    values[neg_idx] = -0.1
    return values

# ── campaign ──────────────────────────────────────────────────────────────────
def generate_anemometer_campaign(real_data_dir, output_dir, models_dir,
                                  n_sensors=3, n_days=7,
                                  start='2022-09-15 08:00:00',
                                  seed=42, force_refit=False):
    """
    Generate synthetic anemometer data for multiple sensors.

    Parameters
    ----------
    real_data_dir : str  — path to real anemometer data for model fitting
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
    if force_refit or not model_exists(models_dir, 'anemometer'):
        synthesizer = fit_anemometer(real_data_dir, models_dir)
    else:
        synthesizer = load_anemometer(models_dir)

    n_records = n_days * 24 * 60
    outputs   = []

    for i in range(n_sensors):
        sensor_id = f"CP{i+1:06d}_E{i+1:03d}"
        rng       = np.random.default_rng(seed + i)
        synthetic = synthesizer.sample(num_rows=n_records)
        synthetic = coerce_numeric(synthetic)
        values    = synthetic['air_flow'].clip(lower=0).values
        values    = inject_edge_cases(values, rng)
        out       = write_anemometer(values, sensor_id,
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
    inst   = config['instruments']['anemometer']

    if not inst['enabled']:
        print("Anemometer disabled in config — skipping")
    else:
        generate_anemometer_campaign(
            real_data_dir = paths['real_data'] / 'anemometer',
            output_dir    = paths['output']    / 'anemometer',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['force_refit'],
        )