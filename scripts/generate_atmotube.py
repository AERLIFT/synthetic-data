# scripts/generate_atmotube.py
import random
import string
from pathlib import Path
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import (
    load_config, validate_dirs,
    load_synthesizer, model_exists, coerce_numeric, get_paths
)

MODEL_COLS = [
    'AQS', 'PM1.0', 'PM2.5', 'PM10',
    'Temperature', 'Humidity', 'Pressure',
    'TVOC_Index', 'TVOC_ppm', 'NOx_Index', 'CO2',
]

# ── fit ───────────────────────────────────────────────────────────────────────
def fit_atmotube(real_data_dir, models_dir):
    model_path    = Path(models_dir) / "atmotube_synthesizer.pkl"
    metadata_path = Path(models_dir) / "atmotube_metadata.json"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Fitting Atmotube model from real data...")
    files = list(Path(real_data_dir).rglob("*.csv"))
    dfs   = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        # normalise column names to safe internal names
        rename = {
            'AQS':              'AQS',
            'PM1.0 (μg/m³)':   'PM1.0',
            'PM2.5 (μg/m³)':   'PM2.5',
            'PM10 (μg/m³)':    'PM10',
            'Temperature (˚C)': 'Temperature',
            'Humidity (%)':     'Humidity',
            'Pressure (hPa)':   'Pressure',
            'TVOC Index':       'TVOC_Index',
            'TVOC (ppm)':       'TVOC_ppm',
            'NOx Index':        'NOx_Index',
            'CO₂ (ppm)':        'CO2',
        }
        df = df.rename(columns=rename)
        available = [c for c in MODEL_COLS if c in df.columns]
        dfs.append(coerce_numeric(df[available]))

    data = pd.concat(dfs).reset_index(drop=True).dropna()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.save_to_json(str(metadata_path))
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    synthesizer.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    return synthesizer

def load_atmotube(models_dir):
    return load_synthesizer(Path(models_dir) / "atmotube_synthesizer.pkl")

# ── filename ──────────────────────────────────────────────────────────────────
def make_atmotube_filename(mac, start, n_days):
    """D7A6AAABC4ED_07-Jun-2026_10-Jun-2026.csv"""
    ts_start = pd.Timestamp(start)
    ts_end   = ts_start + pd.Timedelta(days=n_days)
    fmt      = '%d-%b-%Y'
    return f"{mac}_{ts_start.strftime(fmt)}_{ts_end.strftime(fmt)}.csv"

def _random_mac(rng=None):
    chars = '0123456789ABCDEF'
    if rng is not None:
        return ''.join(rng.choice(list(chars)) for _ in range(12))
    return ''.join(random.choices(chars, k=12))

# ── write ─────────────────────────────────────────────────────────────────────
def write_atmotube(df, mac, start, n_days, output_dir, utc_offset=-6.0):
    timestamps = pd.date_range(start=start, periods=len(df), freq='60s')

    sign   = '+' if utc_offset >= 0 else '-'
    offset = f"UTC{sign}{abs(int(utc_offset)):02d}:00"

    out = pd.DataFrame({
        f'Date ({offset})':  timestamps.strftime('%Y-%m-%d %H:%M:%S'),
        'AQS':               df['AQS'].round(0).astype('Int64'),
        'PM1.0 (μg/m³)':    df['PM1.0'].round(1),
        'PM2.5 (μg/m³)':    df['PM2.5'].round(1),
        'PM10 (μg/m³)':     df['PM10'].round(1),
        'Temperature (˚C)': df['Temperature'].round(1),
        'Humidity (%)':      df['Humidity'].round(0).astype('Int64'),
        'Pressure (hPa)':    df['Pressure'].round(1),
        'TVOC Index':        df['TVOC_Index'].round(0).astype('Int64'),
        'TVOC (ppm)':        df['TVOC_ppm'].round(3),
        'NOx Index':         df['NOx_Index'].round(0).astype('Int64'),
        'CO₂ (ppm)':         df['CO2'].round(0).astype('Int64'),
        'Latitude':          '',
        'Longitude':         '',
        'Altitude (m)':      '',
        'Position Error (m)': '',
        'Battery (%)':       100,
        'Charging':          '',
        'Motion':            'no',
        'Phone GPS':         '',
    })

    fname    = make_atmotube_filename(mac, start, n_days)
    out_path = Path(output_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path

# ── edge cases ────────────────────────────────────────────────────────────────
def inject_edge_cases(df, rng):
    # PM spike
    spike = rng.integers(50, len(df) - 50)
    df.loc[spike, 'PM2.5'] = rng.uniform(80, 150)
    df.loc[spike, 'PM10']  = df.loc[spike, 'PM2.5'] * rng.uniform(1.0, 2.5)

    # CO2 warmup artefact at start
    df.loc[0:5, 'CO2'] = 400.0

    # brief NaN in pressure (connectivity gap)
    gap = rng.integers(200, len(df) - 5)
    df.loc[gap:gap+2, 'Pressure'] = float('nan')

    return df

# ── campaign ──────────────────────────────────────────────────────────────────
def generate_atmotube_campaign(real_data_dir, output_dir, models_dir,
                               n_sensors=3, n_days=7,
                               start='2022-09-15 08:00:00',
                               utc_offset=-6.0,
                               seed=42, force_refit=False):
    """
    Generate synthetic Atmotube data for multiple sensors.

    Parameters
    ----------
    real_data_dir : str   — path to real Atmotube CSV data
    output_dir    : str   — path to write synthetic files
    models_dir    : str   — path to model cache directory
    n_sensors     : int   — number of synthetic sensors
    n_days        : int   — deployment duration in days
    start         : str   — campaign start datetime
    utc_offset    : float — UTC offset for timestamp column label
    seed          : int   — random seed
    force_refit   : bool  — refit model even if cached
    """
    real_data_dir, output_dir = validate_dirs(real_data_dir, output_dir,
                                               ext='.csv')
    if force_refit or not model_exists(models_dir, 'atmotube'):
        synthesizer = fit_atmotube(real_data_dir, models_dir)
    else:
        synthesizer = load_atmotube(models_dir)

    n_records = n_days * 24 * 60
    outputs   = []
    rng_mac   = random.Random(seed)

    for i in range(n_sensors):
        mac       = _random_mac(rng_mac)
        rng       = np.random.default_rng(seed + i)
        synthetic = synthesizer.sample(num_rows=n_records)
        synthetic = coerce_numeric(synthetic)

        synthetic['AQS']        = synthetic['AQS'].clip(0, 500).round(0)
        synthetic['PM1.0']      = synthetic['PM1.0'].clip(0, 500).round(1)
        synthetic['PM2.5']      = synthetic['PM2.5'].clip(0, 500).round(1)
        synthetic['PM10']       = synthetic['PM10'].clip(0, 500).round(1)
        synthetic['Temperature']= synthetic['Temperature'].clip(-20, 60).round(1)
        synthetic['Humidity']   = synthetic['Humidity'].clip(0, 100).round(0)
        synthetic['Pressure']   = synthetic['Pressure'].clip(300, 1100).round(1)
        synthetic['TVOC_Index'] = synthetic['TVOC_Index'].clip(1, 500).round(0)
        synthetic['TVOC_ppm']   = synthetic['TVOC_ppm'].clip(0, 10).round(3)
        synthetic['NOx_Index']  = synthetic['NOx_Index'].clip(1, 500).round(0)
        synthetic['CO2']        = synthetic['CO2'].clip(400, 5000).round(0)

        synthetic = inject_edge_cases(synthetic, rng)

        out = write_atmotube(synthetic, mac, start=pd.Timestamp(start),
                             n_days=n_days, output_dir=output_dir,
                             utc_offset=utc_offset)
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
    inst   = config['instruments']['atmotube']

    if not inst['enabled']:
        print("Atmotube disabled in config — skipping")
    else:
        generate_atmotube_campaign(
            real_data_dir = paths['real_data'] / 'atmotube',
            output_dir    = paths['output']    / 'atmotube',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            utc_offset    = cfg.get('utc_offset', -6.0),
            seed          = cfg['seed'],
            force_refit   = inst['force_refit'],
        )
