# aerlift-synthetic

Synthetic data generator for the AERLIFT multi-instrument personal exposure monitoring campaign. Generates format-faithful raw files for pipeline stress testing using SDV (Synthetic Data Vault) fitted on real instrument data.

---

## Instruments

| Instrument | Format | Resolution | Status |
|---|---|---|---|
| Anemometer (CP202526) | `.txt` | 60s | ✓ |
| Aranet4 | `.csv` | 5min | ✓ |
| Lascar EL-USB CO | `.txt` | 60s | ✓ |
| HHB v2 (v1 + v2 firmware) | `.csv` | 30s | ✓ |
| UPAS v2.1 | `.txt` | 30s | pending |

---

## How it works

Each instrument generator:
1. **Fits** a `GaussianCopulaSynthesizer` on real raw files — learns variable distributions and correlations
2. **Caches** the fitted model to `models/` — subsequent runs load instantly
3. **Generates** synthetic timeseries at native resolution
4. **Injects** known edge cases (negative values, sustained zeros, NaN fills, warmup artifacts)
5. **Writes** output in exact raw file format — filenames, headers, delimiters match real files exactly

Synthetic files flow through the AERLIFT pipeline identically to real data.

---

## Project Structure

```
aerlift-synthetic/
├── config/
│   └── synthetic_config.yaml    # paths, campaign params, instrument settings
├── models/                      # cached SDV models (gitignored)
├── scripts/
│   ├── utils.py                 # shared helpers
│   ├── generate_anemometer.py
│   ├── generate_aranet.py
│   ├── generate_lascar.py
│   └── generate_hhb.py
├── notebooks/
│   └── prototype.ipynb          # development notebook
├── generate_campaign.py         # top-level orchestrator
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/you/aerlift-synthetic.git
cd aerlift-synthetic

conda create -n synthetic-data python=3.12
conda activate synthetic-data
pip install -r requirements.txt
```

Update `config/synthetic_config.yaml` with your local paths:

```yaml
paths:
  real_data: '/path/to/aerlift/data/0_raw'
  output:    '/path/to/aerlift/data/0_synthetic'
  models:    'models'
```

`real_data` should match `raw_dir` in the AERLIFT pipeline `config/config.yaml`.

---

## Usage

**Run all enabled instruments:**
```bash
python generate_campaign.py
```

**Run a single instrument:**
```bash
python scripts/generate_anemometer.py
python scripts/generate_aranet.py
python scripts/generate_lascar.py
python scripts/generate_hhb.py
```

**Force model refit** (e.g. after adding new real data):
```yaml
# in synthetic_config.yaml
instruments:
  anemometer:
    force_refit: true
```

---

## Configuration

```yaml
# ── paths ─────────────────────────────────────────────────────────────────────
paths:
  real_data:  '/path/to/0_raw'
  output:     '/path/to/0_synthetic'
  models:     'models'

# ── campaign ──────────────────────────────────────────────────────────────────
campaign:
  start:          '2022-09-15 08:00:00'
  n_days:         7
  n_sensors:      3
  seed:           42
  gps_utc_offset: -7.0
  utc_offset:     -7.0

# ── instruments ───────────────────────────────────────────────────────────────
instruments:
  anemometer:
    enabled:     true
    force_refit: false
  aranet:
    enabled:     true
    force_refit: false
  lascar:
    enabled:     true
    force_refit: false
  hhb:
    enabled:     true
    firmware:    'v1'    # 'v1' = older firmware (40 cols), 'v2' = newer (101 cols)
    force_refit: false
  upas:
    enabled:     false   # pending
    force_refit: false
```

---

## Outputs

Synthetic files are written to `output/{instrument}/` with filenames matching real instrument conventions:

| Instrument | Example filename |
|---|---|
| Anemometer | `CP000001_E001_2022-09-15 8-00.txt` |
| Aranet4 | `Aranet4 01ABC_2022-09-15T08_00_00-0700.csv` |
| Lascar | `2022_09_15_1000000001_PER_001.txt` |
| HHB | `HHB99001_LOG_2022-09-15T08_00UTC.csv` |

Synthetic sensor IDs are clearly distinguishable from real ones:

| Instrument | Real ID format | Synthetic ID format |
|---|---|---|
| Anemometer | `CP202526_E111` | `CP000001_E001` |
| Aranet4 | `16A28` | random hex e.g. `01ABC` |
| Lascar | `000028223` | `1000000001` |
| HHB | `HHB00029` | `HHB99001` |

---

## Edge Cases Injected

| Instrument | Edge cases |
|---|---|
| Anemometer | Sustained zero run (low flow), negative values |
| Aranet4 | CO₂ low startup, NaN pressure first row, duplicate with NaN |
| Lascar | Negative CO, high CO spike, sustained zeros |
| HHB | Negative PM, low battery period, CO₂ warmup spike |

---

## Adding Real Data

More real files improve synthetic data quality. To refit after adding files:

```bash
# add files to your real data directory then
# set force_refit: true in config, run, then set back to false
python generate_campaign.py
```

---

## Requirements

```
pandas
numpy
sdv
pyyaml
```

---

## Integration with AERLIFT Pipeline

Set `data_mode: synthetic` in the AERLIFT `config/config.yaml` and point `paths.synthetic.raw_dir` to the `output/` directory of this repo. The synthetic files will flow through munge → trim → flag → merge → network identically to real data.

---

## Contact

Mark Campmier, PhD — UC Berkeley School of Public Health
Environmental Health Sciences / Pillarisetti Lab