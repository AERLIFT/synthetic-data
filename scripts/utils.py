# utils.py
from pathlib import Path
import pandas as pd
import yaml
from sdv.utils import load_synthesizer as sdv_load_synthesizer


# ── config ────────────────────────────────────────────────────────────────────
def load_config(config_path='config/synthetic_config.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── validation ────────────────────────────────────────────────────────────────
def validate_dirs(real_data_dir, output_dir, ext):
    real_data_dir = Path(real_data_dir)
    output_dir    = Path(output_dir)

    assert real_data_dir.exists(), \
        f"Real data directory not found: {real_data_dir}"

    files = list(real_data_dir.rglob(f'*{ext}'))
    assert len(files) > 0, \
        f"No {ext} files found in {real_data_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Found {len(files)} real files in {real_data_dir}")
    print(f"✓ Output directory ready: {output_dir}")
    return real_data_dir, output_dir


# ── file parsing ──────────────────────────────────────────────────────────────
def find_data_start(filepath):
    """Return line number where data begins (line after SampleTime header)"""
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('SampleTime'):
                return i + 1
    raise ValueError(f"Could not find SampleTime header in {filepath}")


# ── model cache ───────────────────────────────────────────────────────────────
def load_synthesizer(model_path):
    model_path = Path(model_path)
    assert model_path.exists(), \
        f"No cached model found at {model_path} — run fit first"
    print(f"Loading cached model from {model_path}")
    return sdv_load_synthesizer(str(model_path))


def model_exists(models_dir, instrument, version=None):
    """Check if a cached model exists for an instrument"""
    models_dir = Path(models_dir)
    if version:
        return (models_dir / f"{instrument}_{version}_synthesizer.pkl").exists()
    return (models_dir / f"{instrument}_synthesizer.pkl").exists()


# ── data coercion ─────────────────────────────────────────────────────────────
def coerce_numeric(df):
    """Force all columns to numeric — SDV can return object dtype"""
    return df.apply(pd.to_numeric, errors='coerce').fillna(0.0)


# ── paths ─────────────────────────────────────────────────────────────────────
def get_paths(config):
    """Extract and validate path config"""
    return {
        'real_data': Path(config['paths']['real_data'].strip()),
        'output':    Path(config['paths']['output'].strip()),
        'models':    Path(config['paths']['models'].strip()),
    }