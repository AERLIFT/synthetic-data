# generate_campaign.py
import sys
from pathlib import Path

# make `from utils import ...` work inside the generator modules
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

import time
import numpy as np
import pandas as pd
from scripts.utils import load_config, get_paths
from scripts.generate_anemometer import generate_anemometer_campaign
from scripts.generate_aranet     import generate_aranet_campaign
from scripts.generate_lascar     import generate_lascar_campaign
from scripts.generate_hhb        import generate_hhb_campaign
from scripts.generate_upas       import generate_upas_campaign
from scripts.generate_atmotube   import generate_atmotube_campaign
from scripts.generate_aulifants  import generate_aulifants_campaign
from scripts.generate_ogawa      import generate_ogawa_campaign


def generate_campaign_metadata(output_dir, n_sensors, seed, enabled):
    """Build campaign_metadata.csv by scanning output directories."""
    rng = np.random.default_rng(seed)
    # East Bay / Oakland residential cluster
    lats = 37.8044 + rng.uniform(-0.10, 0.10, n_sensors)
    lons = -122.2712 + rng.uniform(-0.15, 0.15, n_sensors)

    rows = []
    for i in range(n_sensors):
        row = {
            'household_id': f"HH{i+1:03d}",
            'lat':          round(float(lats[i]), 6),
            'lon':          round(float(lons[i]), 6),
        }
        rows.append(row)

    out = Path(output_dir)

    if enabled.get('anemometer'):
        files = sorted((out / 'anemometer').glob('*.txt'))
        for i, f in enumerate(files[:n_sensors]):
            rows[i]['anemometer_sensor_id'] = f.stem.split('_')[1]

    if enabled.get('aranet'):
        files = sorted((out / 'aranet').glob('*.csv'))
        for i, f in enumerate(files[:n_sensors]):
            rows[i]['aranet_sensor_id'] = f.stem.replace('Aranet4 ', '').split('_')[0]

    if enabled.get('lascar'):
        for i in range(n_sensors):
            rows[i]['lascar_sensor_id'] = str(1_000_000_000 + i + 1)

    if enabled.get('hhb'):
        for i in range(n_sensors):
            rows[i]['hhb_sensor_id'] = f"HHB{99000 + i + 1:05d}"

    if enabled.get('upas'):
        for i in range(n_sensors):
            rows[i]['upas_sensor_id'] = f"{99000 + i + 1:05d}"

    if enabled.get('atmotube'):
        files = sorted((out / 'atmotube').glob('*.csv'))
        for i, f in enumerate(files[:n_sensors]):
            rows[i]['atmotube_sensor_id'] = f.stem.split('_')[0]

    if enabled.get('aulifants'):
        all_dirs = [d for d in (out / 'aulifants').iterdir() if d.is_dir()]
        # deduplicate by sensor prefix (multiple campaign dates may coexist)
        seen = {}
        for d in all_dirs:
            prefix = d.stem.split('-Aulifant4-')[0]
            seen[prefix] = prefix
        sensor_prefixes = sorted(seen.keys())
        for i, prefix in enumerate(sensor_prefixes[:n_sensors]):
            rows[i]['aulifants_sensor_id'] = prefix

    df = pd.DataFrame(rows)
    meta_path = out / 'campaign_metadata.csv'
    df.to_csv(meta_path, index=False)
    print(f"\n✓ Metadata: {meta_path}  ({len(df)} households)")
    return meta_path

def run_campaign(config_path='config/synthetic_config.yaml'):
    config = load_config(config_path)
    paths  = get_paths(config)
    cfg    = config['campaign']
    inst   = config['instruments']

    print("=" * 60)
    print("AERLIFT synthetic data generator")
    print(f"Campaign start:  {cfg['start']}")
    print(f"Duration:        {cfg['n_days']} days")
    print(f"Sensors:         {cfg['n_sensors']} per instrument")
    print(f"Seed:            {cfg['seed']}")
    print("=" * 60)

    results = {}
    t_total = time.time()

    # ── anemometer ────────────────────────────────────────────────────────────
    if inst['anemometer']['enabled']:
        print("\n── Anemometer ───────────────────────────────────────────")
        t = time.time()
        results['anemometer'] = generate_anemometer_campaign(
            real_data_dir = paths['real_data'] / 'anemometer',
            output_dir    = paths['output']    / 'anemometer',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['anemometer']['force_refit'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── Anemometer — disabled")

    # ── aranet ────────────────────────────────────────────────────────────────
    if inst['aranet']['enabled']:
        print("\n── Aranet4 ──────────────────────────────────────────────")
        t = time.time()
        results['aranet'] = generate_aranet_campaign(
            real_data_dir = paths['real_data'] / 'aranet',
            output_dir    = paths['output']    / 'aranet',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['aranet']['force_refit'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── Aranet4 — disabled")

    # ── lascar ────────────────────────────────────────────────────────────────
    if inst['lascar']['enabled']:
        print("\n── Lascar ───────────────────────────────────────────────")
        t = time.time()
        results['lascar'] = generate_lascar_campaign(
            real_data_dir = paths['real_data'] / 'lascar',
            output_dir    = paths['output']    / 'lascar',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['lascar']['force_refit'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── Lascar — disabled")

    # ── hhb ───────────────────────────────────────────────────────────────────
    if inst['hhb']['enabled']:
        print("\n── HHB ──────────────────────────────────────────────────")
        t = time.time()
        results['hhb'] = generate_hhb_campaign(
            real_data_dir = paths['real_data'] / 'hhb',
            output_dir    = paths['output']    / 'hhb',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            utc_offset    = cfg['utc_offset'],
            firmware      = inst['hhb']['firmware'],
            seed          = cfg['seed'],
            force_refit   = inst['hhb']['force_refit'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── HHB — disabled")

    # ── upas ──────────────────────────────────────────────────────────────────
    if inst['upas']['enabled']:
        print("\n── UPAS ─────────────────────────────────────────────────")
        t = time.time()
        results['upas'] = generate_upas_campaign(
            real_data_dir = paths['real_data'] / 'upas',
            output_dir    = paths['output']    / 'upas',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            utc_offset    = cfg.get('utc_offset', -7.0),
            seed          = cfg['seed'],
            force_refit   = inst['upas']['force_refit'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── UPAS — disabled")

    # ── atmotube ──────────────────────────────────────────────────────────────
    if inst['atmotube']['enabled']:
        print("\n── Atmotube ─────────────────────────────────────────────")
        t = time.time()
        results['atmotube'] = generate_atmotube_campaign(
            real_data_dir = paths['real_data'] / 'atmotube',
            output_dir    = paths['output']    / 'atmotube',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            utc_offset    = cfg.get('utc_offset', -6.0),
            seed          = cfg['seed'],
            force_refit   = inst['atmotube']['force_refit'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── Atmotube — disabled")

    # ── aulifants ─────────────────────────────────────────────────────────────
    if inst['aulifants']['enabled']:
        print("\n── Aulifants ────────────────────────────────────────────")
        t = time.time()
        results['aulifants'] = generate_aulifants_campaign(
            real_data_dir = paths['real_data'] / 'aulifants',
            output_dir    = paths['output']    / 'aulifants',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
            force_refit   = inst['aulifants']['force_refit'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── Aulifants — disabled")

    # ── ogawa ─────────────────────────────────────────────────────────────────
    if inst['ogawa']['enabled']:
        print("\n── Ogawa ────────────────────────────────────────────────")
        t = time.time()
        results['ogawa'] = generate_ogawa_campaign(
            real_data_dir = paths['real_data'] / 'ogawa',
            output_dir    = paths['output']    / 'ogawa',
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
        )
        print(f"  Elapsed: {time.time() - t:.1f}s")
    else:
        print("\n── Ogawa — disabled")

    # ── metadata ──────────────────────────────────────────────────────────────
    generate_campaign_metadata(
        output_dir = paths['output'],
        n_sensors  = cfg['n_sensors'],
        seed       = cfg['seed'],
        enabled    = {k: v['enabled'] for k, v in inst.items()},
    )

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for instrument, outputs in results.items():
        print(f"  {instrument:<12} {len(outputs)} files → "
              f"{paths['output'] / instrument}")
    print(f"\n  Total elapsed: {time.time() - t_total:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_campaign()
