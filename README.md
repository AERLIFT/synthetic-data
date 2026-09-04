# synthetic-data

Synthetic data generator for the AERLIFT multi-instrument personal exposure monitoring campaign. Generates format-faithful raw files and campaign metadata for pipeline stress testing and validation using SDV (Synthetic Data Vault) fitted on real instrument data.

---

## Instruments

| Instrument | Format | Resolution | Status |
|---|---|---|---|
| Anemometer (CP202526) | `.txt` | 60s | ✓ |
| Aranet4 | `.csv` | 5min | ✓ |
| Lascar EL-USB CO | `.txt` | 60s | ✓ |
| HHB v2 (v1 + v2 firmware) | `.csv` | 30s | ✓ |
| UPAS v2.1 | `.txt` | 30s | ✓ |
| Atmotube PRO | `.csv` | 60s | ✓ |
| Aulifants Smart Plug | `.CSV` | 60s | ✓ |
| Geocene Dots / SUMs | `.csv` | Event-based | ✓ |
| Ogawa Passive Sampler | `.csv` | Integrated (14-day) | ✓ (Parametric) |

---

## How it works

Each instrument generator:
1. **Fits** a `GaussianCopulaSynthesizer` on real raw files — learns variable distributions, covariance, and correlations. For legacy non-timeseries formats (e.g. Ogawa), parametric sampling models are used.
2. **Caches** the fitted model to `models/` — subsequent runs load instantly without re-processing raw data.
3. **Generates** synthetic time series and records at native temporal resolution matching campaign duration and sensor count.
4. **Injects** realistic field edge cases (negative values, sustained zeros, NaN fills, warmup artifacts, spikes, disconnected intervals).
5. **Writes** output in exact raw file formats — filenames, metadata headers, subheaders, delimiters, and directory structures match physical instruments precisely.
6. **Compiles** `campaign_metadata.csv` mapping simulated households to randomized geographic clusters and sensor serial numbers.

Synthetic files flow through the AERLIFT pipeline identically to real data.

---

## Project Structure

```
synthetic-data/
├── config/
│   └── synthetic_config.yaml    # paths, campaign params, instrument settings
├── models/                      # cached SDV models and metadata (gitignored)
├── scripts/
│   ├── utils.py                 # shared configuration, validation, and caching helpers
│   ├── generate_anemometer.py   # Anemometer time-series generator
│   ├── generate_aranet.py       # Aranet4 CO2/T/RH/P generator
│   ├── generate_atmotube.py     # Atmotube PRO multi-pollutant generator
│   ├── generate_aulifants.py    # Aulifants smart-plug power logger generator
│   ├── generate_geocene.py      # Geocene stove use monitor (SUMs) event generator
│   ├── generate_hhb.py          # HHB v2 personal exposure monitor generator
│   ├── generate_lascar.py       # Lascar CO time-series generator
│   ├── generate_ogawa.py        # Ogawa NO/NO2 passive sampler report generator
│   └── generate_upas.py         # UPAS v2.1 personal PM sampler generator
├── generate_campaign.py         # top-level orchestrator & metadata compiler
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/you/synthetic-data.git
cd synthetic-data

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

**Run all enabled instruments and generate campaign metadata:**
```bash
python generate_campaign.py
```

**Run a single instrument:**
```bash
python scripts/generate_anemometer.py
python scripts/generate_aranet.py
python scripts/generate_atmotube.py
python scripts/generate_aulifants.py
python scripts/generate_geocene.py
python scripts/generate_hhb.py
python scripts/generate_lascar.py
python scripts/generate_ogawa.py
python scripts/generate_upas.py
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
  n_sensors:      50
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
    enabled:     true
    force_refit: false

  atmotube:
    enabled:     true
    force_refit: false

  aulifants:
    enabled:     true
    force_refit: false

  geocene:
    enabled:     true
    force_refit: false

  ogawa:
    enabled:     false   # parametric report generator (disabled by default)
```

---

## Outputs

Synthetic files are written to `output/{instrument}/` with filenames and directory structures matching real instrument conventions:

| Instrument | Output Structure / Example Filename |
|---|---|
| Anemometer | `anemometer/CP000001_E001_2022-09-15 8-00.txt` |
| Aranet4 | `aranet/Aranet4 01ABC_2022-09-15T08_00_00-0700.csv` |
| Lascar | `lascar/2022_09_15_1000000001_PER_001.txt` |
| HHB | `hhb/HHB99001_LOG_2022-09-15T08_00UTC.csv` |
| UPAS | `upas/PSP99001_LOG_2022-09-15T08_00_00UTC_SYNTHETIC______C0000_____.txt` |
| Atmotube | `atmotube/D7A6AAABC4ED_15-Sep-2022_22-Sep-2022.csv` |
| Aulifants | `aulifants/101-01-Aulifant4-2022-09-22/150922-D.CSV` |
| Geocene | `geocene/events.csv` |
| Ogawa | `ogawa/ogawa_deployment_20220915_20220929_seed42.csv` |
| Campaign Metadata | `campaign_metadata.csv` |

Synthetic sensor IDs are clearly distinguishable from real ones while preserving format:

| Instrument | Real ID Format | Synthetic ID Format |
|---|---|---|
| Anemometer | `CP202526_E111` | `CP000001_E001` |
| Aranet4 | `16A28` | random hex e.g. `01ABC` |
| Lascar | `000028223` | `1000000001` |
| HHB | `HHB00029` | `HHB99001` |
| UPAS | `PSP00054` | `99001` / `PSP99001` |
| Atmotube | `D7A6AAABC4ED` (MAC) | random 12-char hex MAC |
| Aulifants | `036-01-Aulifant4-YYYY-MM-DD` | `101-01-Aulifant4-YYYY-MM-DD` |
| Geocene | UUID e.g. `0000a410-...` | Deterministic UUID per sensor |
| Ogawa | `OG-xxxx-xxx` | `OG-0042-001` |

---

## Edge Cases Injected

| Instrument | Injected Edge Cases & Anomalies |
|---|---|
| Anemometer | Sustained zero runs (low/blocked flow), negative flow readings |
| Aranet4 | CO₂ low startup artifacts, NaN atmospheric pressure in header row, duplicate timestamps with NaN values |
| Lascar | Negative CO baseline readings, high-exposure spikes, sustained zero runs |
| HHB | Negative optical PM readings, low battery decay periods, CO₂ warmup spikes |
| UPAS | Negative PM values, high particulate smoke spikes, flow rate drops during filter loading |
| Atmotube | Multi-pollutant particulate spikes, CO₂ sensor warmup periods, transient pressure connection dropouts (NaNs) |
| Aulifants | Sustained zero-power unplugged intervals, high-wattage appliance spikes |
| Geocene | Realistic event duration distributions and multi-event stove usage missions |
| Ogawa | Parametric urban/suburban NO/NO2 distributions, laboratory field blanks (`QC_Flag: BLANK`) |

---

## Campaign Metadata

Running `python generate_campaign.py` automatically produces `campaign_metadata.csv` at the root of the output directory. This links simulated household IDs (`HH001`, `HH002`, ...) with:
- Randomized spatial locations (latitude and longitude clustered in the target study area)
- Corresponding synthetic sensor IDs across all active instruments

---

## Adding Real Data

More real files improve synthetic data fidelity and correlation accuracy. To refit after adding files:

```bash
# 1. Add raw instrument files to your real_data directory
# 2. Set force_refit: true in config/synthetic_config.yaml
python generate_campaign.py
# 3. Set force_refit: false after completion
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

Set `data_mode: synthetic` in the AERLIFT `config/config.yaml` and point `paths.synthetic.raw_dir` to the `output/` directory of this repo. The synthetic files will flow through `munge → trim → flag → merge → network` identically to real campaign data.

---

## Contact

Mark Campmier, PhD — UC Berkeley School of Public Health  
Environmental Health Sciences / Pillarisetti Lab
