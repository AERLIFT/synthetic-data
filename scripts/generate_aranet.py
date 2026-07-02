# scripts/generate_aranet.py
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import (
    load_config, validate_dirs,
    load_synthesizer, model_exists, coerce_numeric, get_paths
)

# ── fit ───────────────────────────────────────────────────────────────────────
def fit_aranet(real_data_dir, models_dir):
    model_path    = Path(models_dir) / "aranet_synthesizer.pkl"
    metadata_path = Path(models_dir) / "aranet_metadata.json"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Fitting Aranet4 model from real data...")
    files = list(Path(real_data_dir).rglob("*.csv"))
    data  = pd.concat([
        pd.read_csv(f, usecols=[1, 2, 3, 4],
                    names=['co2', 'temperature', 'rh', 'pressure'],
                    skiprows=1)
        for f in files
    ]).reset_index(drop=True)

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

def load_aranet(models_dir):
    return load_synthesizer(Path(models_dir) / "aranet_synthesizer.pkl")

# ── filename ──────────────────────────────────────────────────────────────────
def make_aranet_filename(sensor_id, start):
    """Aranet4 16A28_2022-10-06T15_56_44-0700.csv"""
    ts   = pd.Timestamp(start)
    code = f"{random.randint(0, 12):02d}{random.randint(0, 59):02d}"
    return f"Aranet4 {sensor_id}_{ts.strftime('%Y-%m-%dT%H_%M_%S')}-{code}.csv"

# ── write ─────────────────────────────────────────────────────────────────────
def write_aranet(df, sensor_id, start, output_dir):
    timestamps = pd.date_range(start=start, periods=len(df), freq='5min')

    out = pd.DataFrame({
        'Time(dd/mm/yyyy)':          timestamps.strftime('%d/%m/%Y %I:%M:%S %p'),
        'Carbon dioxide(ppm)':       df['co2'].round(0).astype('Int64'),
        'Temperature(°C)':           df['temperature'].round(1),
        'Relative humidity(%)':      df['rh'].round(0).astype('Int64'),
        'Atmospheric pressure(hPa)': df['pressure'].round(0).astype('Int64'),
    })

    fname    = make_aranet_filename(sensor_id, start)
    out_path = Path(output_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path

# ── edge cases ────────────────────────────────────────────────────────────────
def inject_edge_cases(df, rng):
    # co2 low — sensor startup
    start = rng.integers(0, 5)
    df.loc[start:start+3, 'co2'] = 350.0

    # NaN fill in pressure (first row known issue)
    df.loc[0, 'pressure'] = float('nan')

    # duplicate row with NaN co2
    dupe_idx = rng.integers(10, len(df) - 10)
    dupe     = df.loc[dupe_idx].copy()
    dupe['co2'] = float('nan')
    df = pd.concat([df.iloc[:dupe_idx+1],
                    pd.DataFrame([dupe]),
                    df.iloc[dupe_idx+1:]]).reset_index(drop=True)
    return df

# ── campaign ──────────────────────────────────────────────────────────────────
def generate_aranet_campaign(real_data_dir, output_dir, models_dir,
                              n_sensors=3, n_days=7,
                              start='2022-09-15 08:00:00',
                              seed=42, force_refit=False):
    """
    Generate synthetic Aranet4 data for multiple sensors.

    Parameters
    ----------
    real_data_dir : str  — path to real Aranet4 data for model fitting
    output_dir    : str  — path to write synthetic files
    models_dir    : str  — path to model cache directory
    n_sensors     : int  — number of synthetic sensors to generate
    n_days        : int  — deployment duration per sensor in days
    start         : str  — campaign start datetime
    seed          : int  — random seed for reproducibility
    force_refit   : bool — refit model even if cached version exists
    """
    real_data_dir, output_dir = validate_dirs(real_data_dir, output_dir,
                                               ext='.csv')
    if force_refit or not model_exists(models_dir, 'aranet'):
        synthesizer = fit_aranet(real_data_dir, models_dir)
    else:
        synthesizer = load_aranet(models_dir)

    n_records = n_days * 24 * 12   # 5-min resolution
    outputs   = []

    for i in range(n_sensors):
        rng_id    = np.random.default_rng(seed * 1_000_000 + i)
        sensor_id = f"{i+1:02X}{int(rng_id.integers(0, 0x1000)):03X}"
        rng       = np.random.default_rng(seed + i)
        synthetic = synthesizer.sample(num_rows=n_records)
        synthetic = coerce_numeric(synthetic)

        synthetic['co2']         = synthetic['co2'].clip(300, 5000).round(0)
        synthetic['temperature'] = synthetic['temperature'].clip(-5, 50).round(1)
        synthetic['rh']          = synthetic['rh'].clip(0, 100).round(0)
        synthetic['pressure']    = synthetic['pressure'].clip(950, 1050).round(0)

        synthetic = inject_edge_cases(synthetic, rng)

        out = write_aranet(synthetic, sensor_id,
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
    inst   = config['instruments']['aranet']

    if not inst['enabled']:
        print("Aranet disabled in config — skipping")
    else:
        generate_aranet_campaign(
            real_data_dir = paths['real_data'] / 'aranet',
            output_dir    = paths['output']    / 'aranet',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['force_refit'],
        )