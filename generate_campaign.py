# generate_campaign.py
from pathlib import Path
import time
from scripts.utils import load_config, get_paths
from scripts.generate_anemometer import generate_anemometer_campaign
from scripts.generate_aranet     import generate_aranet_campaign
from scripts.generate_lascar     import generate_lascar_campaign
from scripts.generate_hhb        import generate_hhb_campaign

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

    # ── upas ─────────────────────────────────────────────────────────────────
    if inst['upas']['enabled']:
        print("\n── UPAS — not yet implemented")
    else:
        print("\n── UPAS — disabled")

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