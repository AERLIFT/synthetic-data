# scripts/generate_aulifants.py
"""
Aulifants smart-plug power monitor.

Raw file layout (no header, *-D.CSV files only — ignore *-E.CSV event logs):
  HH:MM:SS,VVV.VVolt, A.AAAmp, WW.WWatt,PF,   E.EEkWh,$C.CC ,HHhMMM
One file per calendar day, named DDMMYY-D.CSV, grouped in a deployment folder.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import (
    load_config, get_paths, load_synthesizer, model_exists, coerce_numeric
)


# ── parsing helpers ──────────────────────────────────────────────────────────
def _strip(val, suffix):
    return float(val.replace(suffix, '').strip())

def _parse_d_file(path):
    """Return DataFrame with columns [voltage, power] from a D.CSV file."""
    rows = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                voltage = _strip(parts[1], 'Volt')
                power   = _strip(parts[3], 'Watt')
                rows.append({'voltage': voltage, 'power': power})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


# ── fit ───────────────────────────────────────────────────────────────────────
def fit_aulifants(real_data_dir, models_dir):
    model_path    = Path(models_dir) / "aulifants_synthesizer.pkl"
    metadata_path = Path(models_dir) / "aulifants_metadata.json"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Fitting Aulifants model from real data...")
    # Only process *-D.CSV files; skip *-E.CSV event logs
    files = [f for f in Path(real_data_dir).rglob("*-D.CSV")
             if not f.name.upper().endswith('-E.CSV')]
    assert files, f"No *-D.CSV files found under {real_data_dir}"

    dfs = []
    for f in files:
        df = _parse_d_file(f)
        if not df.empty:
            dfs.append(df)
            print(f"  ✓ {f.name}: {len(df)} rows")

    data = pd.concat(dfs).reset_index(drop=True)
    data = coerce_numeric(data).dropna()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.save_to_json(str(metadata_path))
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    synthesizer.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    return synthesizer

def load_aulifants(models_dir):
    return load_synthesizer(Path(models_dir) / "aulifants_synthesizer.pkl")


# ── filename / folder helpers ─────────────────────────────────────────────────
def make_aulifants_folder(sensor_id, end_date):
    """036-01-Aulifant4-2026-02-02"""
    return f"{sensor_id:03d}-01-Aulifant4-{pd.Timestamp(end_date).strftime('%Y-%m-%d')}"

def make_day_filename(date):
    """190126-D.CSV"""
    ts = pd.Timestamp(date)
    return f"{ts.day:02d}{ts.month:02d}{str(ts.year)[2:]}-D.CSV"


# ── write ─────────────────────────────────────────────────────────────────────
def _fmt_duration(total_seconds):
    h, rem = divmod(int(total_seconds), 3600)
    m      = rem // 60
    return f"{h}H{m:02d}M"

def write_aulifants_day(df_day, date, output_dir, cumulative_energy=0.0,
                        cumulative_seconds=0):
    """Write one day's records to a D.CSV file. Returns (path, energy_total, secs_total)."""
    fname    = make_day_filename(date)
    out_path = Path(output_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    ts_start  = pd.Timestamp(date).replace(hour=0, minute=0, second=0)
    energy_kwh = cumulative_energy
    total_secs = cumulative_seconds

    for i, row in enumerate(df_day.itertuples(index=False)):
        ts      = ts_start + pd.Timedelta(seconds=i * 60)
        voltage = max(0.0, row.voltage)
        power   = max(0.0, row.power)
        current = (power / voltage) if voltage > 1 else 0.0
        pf      = round(min(0.99, power / (voltage * current)), 2) \
                  if current > 0.001 else 0

        energy_kwh += (power / 1000.0) * (1 / 60.0)   # kWh per minute
        total_secs += 60
        dur = _fmt_duration(total_secs)

        time_str  = ts.strftime('%-H:%M:%S')           # no leading zero for hour
        volt_str  = f"{voltage:.1f}Volt"
        amp_str   = f" {current:5.2f}Amp"
        watt_str  = f" {power:6.1f}Watt"
        pf_str    = f"{pf}" if power > 0 else "0"
        kwh_str   = f"   {energy_kwh:5.2f}kWh"
        cost_str  = f"$0.00 "

        lines.append(f"{time_str},{volt_str},{amp_str},{watt_str},"
                     f"{pf_str},{kwh_str},{cost_str},{dur}")

    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines))

    return out_path, energy_kwh, total_secs


# ── edge cases ────────────────────────────────────────────────────────────────
def inject_edge_cases(df, rng):
    # sustained zero-power stretch (device unplugged briefly)
    z = rng.integers(50, len(df) - 20)
    df.loc[z:z+10, 'power'] = 0.0

    # high-draw spike (e.g. vacuum cleaner)
    spike = rng.integers(20, len(df) - 5)
    df.loc[spike, 'power'] = rng.uniform(800, 1400)

    return df


# ── campaign ──────────────────────────────────────────────────────────────────
def generate_aulifants_campaign(real_data_dir, output_dir, models_dir,
                                n_sensors=3, n_days=7,
                                start='2022-09-15 08:00:00',
                                seed=42, force_refit=False):
    """
    Generate synthetic Aulifants smart-plug data for multiple sensors.

    Parameters
    ----------
    real_data_dir : str  — path to real Aulifants data (sub-folders with D.CSVs)
    output_dir    : str  — path to write synthetic deployment folders
    models_dir    : str  — path to model cache directory
    n_sensors     : int  — number of synthetic sensors
    n_days        : int  — deployment duration in days
    start         : str  — campaign start datetime
    seed          : int  — random seed
    force_refit   : bool — refit model even if cached
    """
    real_data_dir = Path(real_data_dir)
    output_dir    = Path(output_dir)
    assert real_data_dir.exists(), f"Real data dir not found: {real_data_dir}"

    d_files = [f for f in real_data_dir.rglob("*-D.CSV")
               if not f.name.upper().endswith('-E.CSV')]
    assert d_files, f"No *-D.CSV files found in {real_data_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Found {len(d_files)} D.CSV files in {real_data_dir}")
    print(f"✓ Output directory ready: {output_dir}")

    if force_refit or not model_exists(models_dir, 'aulifants'):
        synthesizer = fit_aulifants(real_data_dir, models_dir)
    else:
        synthesizer = load_aulifants(models_dir)

    n_records_per_day = 24 * 60
    n_records_total   = n_days * n_records_per_day
    ts_start          = pd.Timestamp(start)
    ts_end            = ts_start + pd.Timedelta(days=n_days)
    outputs           = []

    for i in range(n_sensors):
        rng       = np.random.default_rng(seed + i)
        synthetic = synthesizer.sample(num_rows=n_records_total)
        synthetic = coerce_numeric(synthetic)
        synthetic['voltage'] = synthetic['voltage'].clip(100, 260)
        synthetic['power']   = synthetic['power'].clip(lower=0)
        synthetic            = inject_edge_cases(synthetic, rng)

        sensor_id = 100 + i + 1
        folder    = make_aulifants_folder(sensor_id, ts_end)
        sensor_dir = output_dir / folder
        sensor_dir.mkdir(parents=True, exist_ok=True)

        cum_energy = 0.0
        cum_secs   = 0
        day_paths  = []

        for d in range(n_days):
            day_date   = ts_start + pd.Timedelta(days=d)
            slice_df   = synthetic.iloc[d * n_records_per_day:(d + 1) * n_records_per_day]
            slice_df   = slice_df.reset_index(drop=True)
            p, cum_energy, cum_secs = write_aulifants_day(
                slice_df, day_date, sensor_dir,
                cumulative_energy=cum_energy, cumulative_seconds=cum_secs
            )
            day_paths.append(p)

        outputs.extend(day_paths)
        print(f"✓ Sensor {i+1}/{n_sensors}: {folder} "
              f"({len(day_paths)} daily files)")

    print(f"\n✓ Generated {n_sensors} sensors × {n_days} days")
    return outputs


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    paths  = get_paths(config)
    cfg    = config['campaign']
    inst   = config['instruments']['aulifants']

    if not inst['enabled']:
        print("Aulifants disabled in config — skipping")
    else:
        generate_aulifants_campaign(
            real_data_dir = paths['real_data'] / 'aulifants',
            output_dir    = paths['output']    / 'aulifants',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['force_refit'],
        )
