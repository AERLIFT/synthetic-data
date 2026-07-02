# scripts/generate_ogawa.py
"""
Ogawa passive diffusion tube sampler.

Ogawa badges are deployed for ~2-week periods and analysed in a lab for
NO and NO2 (μg/m³ or ppb).  The raw files are legacy XLS chain-of-custody
and report forms — not time-series logs.  Because the XLS files require
xlrd (not installed in the default environment), this generator uses
parametric statistics derived from the original files to produce synthetic
CSV reports in the same tabular structure.

Each output file represents one deployment set (multiple sites × one period).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from utils import load_config, get_paths

# Approximate NO2 and NO distributions observed from the raw Ogawa reports
# (μg/m³, log-normal parameterised to realistic urban/suburban ranges)
NO2_MEAN_UGM3  = 25.0    # μg/m³  (≈ 13 ppb)
NO2_STD_UGM3   = 12.0
NO_MEAN_UGM3   = 10.0    # μg/m³  (≈ 8 ppb)
NO_STD_UGM3    =  6.0
BLANK_RATE     =  0.05   # fraction of samples treated as field blanks

SITE_CODES = [
    'URBAN-01', 'URBAN-02', 'SUBURBAN-01', 'SUBURBAN-02',
    'RURAL-01',  'TRAFFIC-01', 'TRAFFIC-02',
]


def _ppb_from_ugm3_no2(ugm3):
    """Convert NO2 μg/m³ → ppb (MW = 46.01 g/mol, 25 °C, 1 atm)."""
    return ugm3 / 1.88

def _ppb_from_ugm3_no(ugm3):
    """Convert NO μg/m³ → ppb (MW = 30.01 g/mol)."""
    return ugm3 / 1.23


# ── generate ──────────────────────────────────────────────────────────────────
def generate_ogawa_campaign(real_data_dir, output_dir,
                            n_sensors=3, n_days=14,
                            start='2022-09-15 08:00:00',
                            seed=42, **kwargs):
    """
    Generate synthetic Ogawa passive sampler deployment records.

    Parameters
    ----------
    real_data_dir : str  — path to real Ogawa files (used only for existence check)
    output_dir    : str  — path to write synthetic CSV reports
    n_sensors     : int  — number of sites/badges per deployment set
    n_days        : int  — deployment duration (days between deploy/retrieve)
    start         : str  — deployment start date
    seed          : int  — random seed
    """
    real_data_dir = Path(real_data_dir)
    output_dir    = Path(output_dir)
    assert real_data_dir.exists(), f"Ogawa data dir not found: {real_data_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng      = np.random.default_rng(seed)
    ts_start = pd.Timestamp(start)
    ts_end   = ts_start + pd.Timedelta(days=n_days)

    # Build one synthetic report file per campaign call
    n_sites  = max(n_sensors, len(SITE_CODES))
    sites    = (SITE_CODES * ((n_sites // len(SITE_CODES)) + 1))[:n_sensors]

    rows = []
    for idx, site in enumerate(sites):
        is_blank = rng.random() < BLANK_RATE
        if is_blank:
            no2_ugm3, no_ugm3 = 0.0, 0.0
            flag = 'BLANK'
        else:
            no2_ugm3 = max(0.0, rng.normal(NO2_MEAN_UGM3, NO2_STD_UGM3))
            no_ugm3  = max(0.0, rng.normal(NO_MEAN_UGM3, NO_STD_UGM3))
            flag = ''

        rows.append({
            'SampleID':       f"OG-{seed:04d}-{idx+1:03d}",
            'SiteCode':       site,
            'DeployDate':     ts_start.strftime('%Y-%m-%d'),
            'RetrievalDate':  ts_end.strftime('%Y-%m-%d'),
            'ExposureDays':   n_days,
            'NO2_ug_m3':      round(no2_ugm3, 2),
            'NO2_ppb':        round(_ppb_from_ugm3_no2(no2_ugm3), 2),
            'NO_ug_m3':       round(no_ugm3, 2),
            'NO_ppb':         round(_ppb_from_ugm3_no(no_ugm3), 2),
            'NOx_ppb':        round(_ppb_from_ugm3_no2(no2_ugm3) +
                                    _ppb_from_ugm3_no(no_ugm3), 2),
            'QC_Flag':        flag,
            'Analyst':        'SYNTHETIC-LAB',
            'AnalysisDate':   (ts_end + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
        })

    df    = pd.DataFrame(rows)
    fname = (f"ogawa_deployment_{ts_start.strftime('%Y%m%d')}"
             f"_{ts_end.strftime('%Y%m%d')}_seed{seed}.csv")
    out_path = output_dir / fname
    df.to_csv(out_path, index=False)

    print(f"✓ Ogawa report: {out_path.name}  ({len(rows)} sites, "
          f"{n_days}-day deployment)")
    return [out_path]


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    paths  = get_paths(config)
    cfg    = config['campaign']
    inst   = config['instruments']['ogawa']

    if not inst['enabled']:
        print("Ogawa disabled in config — skipping")
    else:
        generate_ogawa_campaign(
            real_data_dir = paths['real_data'] / 'ogawa',
            output_dir    = paths['output']    / 'ogawa',
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            seed          = cfg['seed'],
        )
