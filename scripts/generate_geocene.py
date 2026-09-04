# scripts/generate_geocene.py
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import (
    load_config, validate_dirs,
    load_synthesizer, model_exists, get_paths
)

# ── fit ───────────────────────────────────────────────────────────────────────
def fit_geocene(real_data_dir, models_dir):
    model_path    = Path(models_dir) / "geocene_synthesizer.pkl"
    metadata_path = Path(models_dir) / "geocene_metadata.json"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Fitting Geocene model from real data...")
    # Geocene data is expected to be in a single events.csv or multiple
    files = list(Path(real_data_dir).rglob("*.csv"))
    
    # We want to learn the distributions of processor_name, model_name, event_kind, and duration
    all_data = []
    for f in files:
        df = pd.read_csv(f)
        # Convert times to duration in minutes for modeling
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['stop_time'] = pd.to_datetime(df['stop_time'])
        df['duration_min'] = (df['stop_time'] - df['start_time']).dt.total_seconds() / 60.0
        
        # Keep relevant columns for modeling
        # We don't model start_time directly as a distribution across all time, 
        # but rather events per day later.
        all_data.append(df[['processor_name', 'model_name', 'event_kind', 'duration_min']])
    
    data = pd.concat(all_data).reset_index(drop=True)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.save_to_json(str(metadata_path))
    print(f"✓ Metadata saved to {metadata_path}")

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    synthesizer.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    return synthesizer

def load_geocene(models_dir):
    return load_synthesizer(Path(models_dir) / "geocene_synthesizer.pkl")

# ── filename ──────────────────────────────────────────────────────────────────
def make_geocene_filename():
    """Geocene usually has a single events.csv file in the raw dir"""
    return "events.csv"

# ── write ─────────────────────────────────────────────────────────────────────
def write_geocene(df, output_dir):
    out_path = Path(output_dir) / make_geocene_filename()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If the file already exists (from other sensors), we append
    if out_path.exists():
        df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path

# ── campaign ──────────────────────────────────────────────────────────────────
def generate_geocene_campaign(real_data_dir, output_dir, models_dir,
                              n_sensors=3, n_days=7,
                              start='2022-09-15 08:00:00',
                              seed=42, force_refit=False):
    """
    Generate synthetic Geocene events data.
    """
    real_data_dir, output_dir = validate_dirs(real_data_dir, output_dir, ext='.csv')
    
    if force_refit or not model_exists(models_dir, 'geocene'):
        synthesizer = fit_geocene(real_data_dir, models_dir)
    else:
        synthesizer = load_geocene(models_dir)

    # Clear existing events.csv if it exists to start fresh for this campaign run
    out_file = Path(output_dir) / make_geocene_filename()
    if out_file.exists():
        out_file.unlink()

    campaign_start = pd.Timestamp(start).tz_localize(None)
    campaign_end = campaign_start + pd.Timedelta(days=n_days)
    
    total_events_generated = 0
    
    for i in range(n_sensors):
        rng = np.random.default_rng(seed + i)
        sensor_id = f"{seed*1000 + i:08x}-0000-0000-0000-{random.randint(0, 0xffffffff):08x}"
        
        # Determine number of events for this sensor
        # Assuming ~2-5 events per day
        n_events = rng.integers(n_days * 1, n_days * 5)
        
        if n_events > 0:
            synthetic = synthesizer.sample(num_rows=n_events)
            
            # Assign start times randomly within the campaign
            durations = pd.to_timedelta(synthetic['duration_min'].clip(5, 180), unit='m')
            
            # Generate random start times
            start_seconds = rng.integers(0, int((campaign_end - campaign_start).total_seconds()), size=n_events)
            start_times = [campaign_start + pd.Timedelta(seconds=int(s)) for s in start_seconds]
            
            synthetic['start_time'] = start_times
            synthetic['stop_time'] = [s + d for s, d in zip(start_times, durations)]
            synthetic['mission_id'] = sensor_id
            
            # Format times as ISO8601 UTC
            synthetic['start_time'] = synthetic['start_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            synthetic['stop_time'] = synthetic['stop_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Reorder columns to match real data
            cols = ['mission_id', 'processor_name', 'model_name', 'event_kind', 'start_time', 'stop_time']
            out_df = synthetic[cols].sort_values('start_time')
            
            write_geocene(out_df, output_dir)
            total_events_generated += n_events

    print(f"✓ Generated {n_sensors} sensors with {total_events_generated} total events in {out_file}")
    return [out_file]

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    paths  = get_paths(config)
    cfg    = config['campaign']
    
    # Check if geocene is in config, default to disabled if not yet added
    inst = config['instruments'].get('geocene', {'enabled': False, 'force_refit': False})

    if not inst['enabled']:
        print("Geocene disabled in config — skipping")
    else:
        generate_geocene_campaign(
            real_data_dir = paths['real_data'] / 'geocene',
            output_dir    = paths['output']    / 'geocene',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst.get('force_refit', False),
        )
