# scripts/generate_hhb.py
from pathlib import Path
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import (
    load_config, validate_dirs, find_data_start,
    load_synthesizer, model_exists, coerce_numeric, get_paths
)

# ── column definitions ────────────────────────────────────────────────────────
COMMON_COLS = [
    'AccelX', 'AccelY', 'AccelZ', 'Vbus', 'M.Vsupply', 'M.5V0', 'M.3V3',
    'Vbattery', 'Battery_Temp', 'M.BMP581_Press', 'M.BMP581_Temp',
    'SEN55_PM1.0', 'SEN55_PM2.5', 'SEN55_PM4.0', 'SEN55_PM10',
    'SEN55_RH', 'SEN55_Temp', 'SEN55_RawVOC', 'SEN55_RawNOx',
    'G.5V0', 'G.3V3', 'G.BMP581_Press', 'G.BMP581_Temp',
    'G.SCD30_CO2', 'G.SCD30_Temp', 'G.SCD30_RH',
    'G.WE1', 'G.AUX1', 'G.WE2', 'G.AUX2'
]

OLD_FIRMWARE_COLS = [
    'A.Vsupply', 'A.3V3', 'A.BMP581Int_Press', 'A.BMP581Int_Temp',
    'A.BMP581Ext_Press', 'A.BMP581Ext_Temp', 'A.AtmoDensity',
    'G.SGP41_RawVOC', 'G.SGP41_RawNOx',
]

NEW_FIRMWARE_COLS = [
    'A.Vsupply', 'A.3V3', 'A.BMP581Int_Press', 'A.BMP581Int_Temp',
    'A.BMP581Ext_Press', 'A.BMP581Ext_Temp', 'A.AtmoDensity',
    'G.Alphasense1_Algorithm1', 'G.Alphasense1_Algorithm2',
    'G.Alphasense1_Algorithm3', 'G.Alphasense1_Algorithm4',
    'G.Alphasense2_Algorithm1', 'G.Alphasense2_Algorithm2',
    'G.Alphasense2_Algorithm3', 'G.Alphasense2_Algorithm4',
    'D.MassFlow', 'D.VolFlow', 'C.MassFlow', 'C.VolFlow',
    'A.MassFlow', 'A.VolFlow', 'B.MassFlow', 'B.VolFlow',
]

HHB_COLUMNS = {
    'v1': COMMON_COLS + OLD_FIRMWARE_COLS,
    'v2': COMMON_COLS + NEW_FIRMWARE_COLS,
}

INT_COLS = {
    'AccelX', 'AccelY', 'AccelZ',
    'SEN55_RawVOC', 'SEN55_RawNOx',
    'G.SGP41_RawVOC', 'G.SGP41_RawNOx',
    'A.Pumps', 'B.Pumps', 'A.RDAC', 'B.RDAC', 'C.RDAC', 'D.RDAC',
    'G.Alphasense1_Algorithm1', 'G.Alphasense1_Algorithm2',
    'G.Alphasense1_Algorithm3', 'G.Alphasense1_Algorithm4',
    'G.Alphasense2_Algorithm1', 'G.Alphasense2_Algorithm2',
    'G.Alphasense2_Algorithm3', 'G.Alphasense2_Algorithm4',
}

HHB_UNITS = {
    'v1': (
        "(HH:MM:SS),(YYYY-MM-DDTHH:MM:SS) (UTC date time format),"
        "(integer),(integer),(integer),"
        "(V),(V),(V),(V),(V),(C),(PaA),(C),"
        "(ug*m^-3),(ug*m^-3),(ug*m^-3),(ug*m^-3),"
        "(%),(C),(integer),(integer),"
        "(V),(V),(PaA),(C),(PaA),(C),(gL^-1),"
        "(V),(V),(PaA),(C),"
        "(ppm),(C),(%),"
        "(integer),(integer),(V),(V),(V),(V)"
    ),
    'v2': (
        "(HH:MM:SS),(YYYY-MM-DDTHH:MM:SS) (UTC date time format),"
        "(integer),(integer),(integer),"
        "(V),(V),(V),(V),(V),(C),(PaA),(C),"
        "(ug*m^-3),(ug*m^-3),(ug*m^-3),(ug*m^-3),"
        "(%),(C),(integer),(integer),"
        "(V),(V),(PaA),(C),(integer),"
        "(V),(V),(V),(PaA),(C),(g*min^-1),(L*min^-1),(L),(L),(L),"
        "(integer),(V),(V),(V),(PaA),(C),(g*min^-1),(L*min^-1),(L),(L),(L),"
        "(gL^-1),(V),(V),(integer),(integer),(V),(V),(V),(PaA),(C),(PaA),(C),"
        "(g*min^-1),(L*min^-1),(L),(L),(L),(gL^-1),"
        "(V),(V),(integer),(integer),(V),(V),(V),(PaA),(C),(PaA),(C),"
        "(g*min^-1),(L*min^-1),(L),(L),(L),(gL^-1),"
        "(V),(V),(PaA),(C),(ppm),(C),(%),"
        "(V),(V),(ppb),(ppb),(ppb),(ppb),"
        "(V),(V),(ppb),(ppb),(ppb),(ppb)"
    ),
}

# ── helpers ───────────────────────────────────────────────────────────────────
def detect_firmware(f):
    skiprows = find_data_start(f) - 1
    df       = pd.read_csv(f, skiprows=skiprows, nrows=0)
    return 'v1' if len(df.columns) <= 42 else 'v2'

def make_hhb_filename(serial, start):
    """HHB00029_LOG_2024-08-13T21_07UTC.csv"""
    ts = pd.Timestamp(start).tz_localize('UTC')
    return f"{serial}_LOG_{ts.strftime('%Y-%m-%dT%H_%M')}UTC.csv"

# ── header ────────────────────────────────────────────────────────────────────
def make_hhb_header(serial, fname, start, n_days, utc_offset=-4.0,
                    alphasense1_id='0000000001', alphasense2_id='0000000002',
                    firmware='v1'):
    ts_start  = pd.Timestamp(start).tz_localize('UTC')
    ts_end    = ts_start + pd.Timedelta(days=n_days)
    runtime   = n_days * 24
    col_names = ','.join(['SampleTime', 'DateTimeUTC'] + HHB_COLUMNS[firmware])
    units_row = HHB_UNITS[firmware]

    return (
        f"PARAMETER,VALUE,UNITS/NOTES\n"
        f"\n"
        f"HHBserial,{serial},(HHB box serial identification)\n"
        f"SEN55_Serial,SYNTHETIC000000000,(SEN55 serial identification)\n"
        f"HHBslot1,Empty,(No expansion board)\n"
        f"HHBslot2,FP00001,(A Filter pumping expansion board)\n"
        f"HHBslot3,Empty,(No expansion board)\n"
        f"HHBslot4,Empty,(No expansion board)\n"
        f"HHBslot5,Empty,(No expansion board)\n"
        f"HHBslot6,GS00001,(Gas sensor expansion board)\n"
        f"G.Alphasense1_ID,{alphasense1_id},(Alphasense #1 on gas expansion board)\n"
        f"G.Alphasense2_ID,{alphasense2_id},(Alphasense #2 on gas expansion board)\n"
        f"G.SCD30_Serial,SYNTHETIC-SCD30,(SCD30 on gas expansion board)\n"
        f"G.SGP41_Serial,SYNTHETIC41,(SGP41 on gas expansion board)\n"
        f"Firmware,HHBv2 Jan 11 2024 08:35:16,(installed firmware version)\n"
        f"\n\n\n"
        f"\n"
        f"SAMPLE IDENTIFICATION\n"
        f"\n"
        f"LogFileName,{fname},(log file filename-automatically defined)\n"
        f"SampleName,SYNTHETIC,(sample name)\n"
        f"A.FilterCID,,(A Filter cartridge ID)\n"
        f"\n\n\n"
        f"\n"
        f"MASS FLOW SENSOR CALIBRATION\n"
        f"\n"
        f"A.FilterCalDate,2024-05-16T10:00:00,(YYYY-MM-DDTHH:MM:SS) (UTC date time format)\n"
        f"A.FilterCalVoutMin,0.595875,(V)\n"
        f"A.FilterCalVoutMax,1.891500,(V)\n"
        f"A.FilterCalMFMin,0.251100,(g*min^-1)\n"
        f"A.FilterCalMFMax,3.473000,(g*min^-1)\n"
        f"A.FilterMF4,0.685189,(coefficient)\n"
        f"A.FilterMF3,-1.948710,(coefficient)\n"
        f"A.FilterMF2,1.932300,(coefficient)\n"
        f"A.FilterMF1,0.826531,(coefficient)\n"
        f"A.FilterMF0,-0.601246,(coefficient)\n"
        f"\n\n\n"
        f"\n"
        f"SETUP SUMMARY\n"
        f"\n"
        f"UTCOffset,{utc_offset:.2f},(hours offset from UTC date time)\n"
        f"ProgrammedRuntime,{n_days * 86400:.6f},(s)\n"
        f"SEN55_Runtime,{runtime:.3f},(Hr)\n"
        f"SEN55_FanRuntime,{runtime:.3f},(Hr)\n"
        f"A.FilterPumpStartingVolume,8.64,(L)\n"
        f"A.FilterPumpStartingRuntime,0.100,(Hr)\n"
        f"A.FilterCartridgeStartingVolume,8.64,(L)\n"
        f"A.FilterCartridgeStartingRuntime,0.000,(Hr)\n"
        f"A.FilterVolumetricFlowRate,0.00,(L*min^-1)\n"
        f"A.FilterDutyCycle,100.0,(%)\n"
        f"G.Alphasense1_Runtime,{runtime:.3f},(Hr)\n"
        f"G.Alphasense2_Runtime,{runtime:.3f},(Hr)\n"
        f"G.FanRuntime,{runtime:.3f},(Hr)\n"
        f"G.SCD30_Runtime,{runtime:.3f},(Hr)\n"
        f"\n\n\n"
        f"\n"
        f"SAMPLE SUMMARY\n"
        f"\n"
        f"StartDateTimeUTC,{ts_start.strftime('%Y-%m-%dT%H:%M:%S')},(YYYY-MM-DDTHH:MM:SS) (UTC date time format)\n"
        f"EndDateTimeUTC,{ts_end.strftime('%Y-%m-%dT%H:%M:%S')},(YYYY-MM-DDTHH:MM:SS) (UTC date time format)\n"
        f"HHBSampledRuntime,        {runtime:.3f},(Hr)\n"
        f"A.FilterShutdownMode,0,(0=uncontrolled 1=test finished 2=high power)\n"
        f"A.FilterSampledRunTime,          0.000,(Hr)\n"
        f"A.FilterSampledVolume,           0.00,(L)\n"
        f"A.FilterAverageVolumetricFlowRate,          0.000,(L*min^-1)\n"
        f"\n\n\n"
        f"\n"
        f"SAMPLE LOG\n"
        f"\n"
        f"{units_row}\n"
        f"{col_names}"
    )

# ── fit ───────────────────────────────────────────────────────────────────────
def fit_hhb(real_data_dir, models_dir):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    files = list(Path(real_data_dir).rglob("HHB*.csv"))
    assert len(files) > 0, f"No HHB*.csv files found in {real_data_dir}"

    v1_files = [f for f in files if detect_firmware(f) == 'v1']
    v2_files = [f for f in files if detect_firmware(f) == 'v2']
    print(f"Firmware v1: {len(v1_files)} files, v2: {len(v2_files)} files")

    models = {}
    for version, flist, extra_cols in [
        ('v1', v1_files, OLD_FIRMWARE_COLS),
        ('v2', v2_files, NEW_FIRMWARE_COLS),
    ]:
        if not flist:
            print(f"No {version} files — skipping")
            continue

        cols = COMMON_COLS + extra_cols
        dfs  = []
        for f in flist:
            try:
                skiprows  = find_data_start(f) - 1
                df        = pd.read_csv(f, skiprows=skiprows)
                available = [c for c in cols if c in df.columns]
                df        = coerce_numeric(df[available])
                dfs.append(df)
                print(f"  ✓ {f.name} ({version}): {len(df)} rows")
            except Exception as e:
                print(f"  ✗ {f.name}: {e} — skipping")

        if not dfs:
            print(f"  No valid {version} files — skipping model")
            continue

        data          = pd.concat(dfs).reset_index(drop=True).dropna()
        metadata      = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        model_path    = Path(models_dir) / f"hhb_{version}_synthesizer.pkl"
        metadata_path = Path(models_dir) / f"hhb_{version}_metadata.json"
        metadata.save_to_json(str(metadata_path))

        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(data)
        synthesizer.save(str(model_path))
        models[version] = synthesizer
        print(f"✓ HHB {version} model saved to {model_path}")

    return models

def load_hhb(models_dir, firmware='v1'):
    return load_synthesizer(
        Path(models_dir) / f"hhb_{firmware}_synthesizer.pkl"
    )

# ── write ─────────────────────────────────────────────────────────────────────
def write_hhb(df, serial, start, n_days, output_dir,
              utc_offset=-4.0, alphasense1_id='0000000001',
              alphasense2_id='0000000002', firmware='v1'):
    ts_start   = pd.Timestamp(start).tz_localize('UTC')
    timestamps = pd.date_range(start=ts_start, periods=len(df), freq='30s')
    cols       = HHB_COLUMNS[firmware]
    fname      = make_hhb_filename(serial, start)
    header     = make_hhb_header(serial, fname, start, n_days,
                                  utc_offset, alphasense1_id,
                                  alphasense2_id, firmware)
    rows = []
    for i, (ts, (_, row)) in enumerate(zip(timestamps, df.iterrows())):
        elapsed     = pd.Timedelta(seconds=i * 30)
        h, rem      = divmod(int(elapsed.total_seconds()), 3600)
        m, s        = divmod(rem, 60)
        sample_time = f"{h}:{m:02d}:{s:02d}"
        dt_utc      = ts.strftime('%Y-%m-%dT%H:%M:%S')
        values      = []
        for col in cols:
            if col not in row.index:
                values.append('0')
            elif col in INT_COLS:
                try:
                    values.append(str(int(float(row[col]))))
                except (ValueError, TypeError):
                    values.append('0')
            else:
                try:
                    values.append(f"{float(row[col]):.3f}")
                except (ValueError, TypeError):
                    values.append('0.000')
        rows.append(f"{sample_time},{dt_utc},{','.join(values)}")

    out_path = Path(output_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(header)
        f.write('\n')
        f.write('\n'.join(rows))
    return out_path

# ── edge cases ────────────────────────────────────────────────────────────────
def inject_edge_cases(df, rng):
    neg_idx    = rng.integers(10, len(df) - 10)
    batt_start = rng.integers(100, len(df) - 50)
    df.loc[neg_idx, 'SEN55_PM2.5']              = -0.1
    df.loc[batt_start:batt_start+20, 'Vbattery'] = 3.1
    df.loc[0:10, 'G.SCD30_CO2']                 = 1200.0
    return df

# ── campaign ──────────────────────────────────────────────────────────────────
def generate_hhb_campaign(real_data_dir, output_dir, models_dir,
                           n_sensors=3, n_days=7,
                           start='2022-09-15 08:00:00',
                           utc_offset=-7.0, firmware='v1',
                           seed=42, force_refit=False):
    """
    Generate synthetic HHB data for multiple sensors.

    Parameters
    ----------
    real_data_dir : str   — path to real HHB raw data for model fitting
    output_dir    : str   — path to write synthetic files
    models_dir    : str   — path to model cache directory
    n_sensors     : int   — number of synthetic sensors to generate
    n_days        : int   — deployment duration per sensor in days
    start         : str   — campaign start datetime (UTC)
    utc_offset    : float — UTC offset for local time in header
    firmware      : str   — 'v1' (older) or 'v2' (newer) firmware format
    seed          : int   — random seed for reproducibility
    force_refit   : bool  — refit model even if cached version exists
    """
    real_data_dir, output_dir = validate_dirs(real_data_dir, output_dir,
                                               ext='.csv')
    if force_refit or not model_exists(models_dir, 'hhb', version=firmware):
        models      = fit_hhb(real_data_dir, models_dir)
        synthesizer = models[firmware]
    else:
        synthesizer = load_hhb(models_dir, firmware)

    n_records = n_days * 24 * 120
    outputs   = []

    for i in range(n_sensors):
        serial    = f"HHB{99000 + i + 1:05d}"
        rng       = np.random.default_rng(seed + i)
        synthetic = synthesizer.sample(num_rows=n_records)
        synthetic = coerce_numeric(synthetic)

        synthetic['SEN55_PM1.0']  = synthetic['SEN55_PM1.0'].clip(0, 500)
        synthetic['SEN55_PM2.5']  = synthetic['SEN55_PM2.5'].clip(0, 500)
        synthetic['SEN55_PM4.0']  = synthetic['SEN55_PM4.0'].clip(0, 500)
        synthetic['SEN55_PM10']   = synthetic['SEN55_PM10'].clip(0, 500)
        synthetic['SEN55_RH']     = synthetic['SEN55_RH'].clip(0, 100)
        synthetic['SEN55_Temp']   = synthetic['SEN55_Temp'].clip(-5, 50)
        synthetic['G.SCD30_CO2']  = synthetic['G.SCD30_CO2'].clip(400, 5000)
        synthetic['G.SCD30_RH']   = synthetic['G.SCD30_RH'].clip(0, 100)
        synthetic['G.SCD30_Temp'] = synthetic['G.SCD30_Temp'].clip(-5, 50)
        synthetic = inject_edge_cases(synthetic, rng)

        out = write_hhb(synthetic, serial, start, n_days,
                        output_dir, utc_offset=utc_offset, firmware=firmware)
        outputs.append(out)
        print(f"✓ Sensor {i+1}/{n_sensors}: {out.name}")

    print(f"\n✓ Generated {n_sensors} × {firmware} sensors, "
          f"{n_days} days ({n_records} records each)")
    return outputs

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    paths  = get_paths(config)
    cfg    = config['campaign']
    inst   = config['instruments']['hhb']

    if not inst['enabled']:
        print("HHB disabled in config — skipping")
    else:
        generate_hhb_campaign(
            real_data_dir = paths['real_data'] / 'hhb',
            output_dir    = paths['output']    / 'hhb',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            utc_offset    = cfg['utc_offset'],
            firmware      = inst['firmware'],
            seed          = cfg['seed'],
            force_refit   = inst['force_refit'],
        )