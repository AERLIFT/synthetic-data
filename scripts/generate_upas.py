# scripts/generate_upas.py
"""
UPAS v2 (Personal Exposure Monitor).

Header format mirrors HHB: PARAMETER,VALUE,UNITS/NOTES metadata block,
then SAMPLE LOG section with three sub-header rows (category, column names,
units) followed by 30-second data rows.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import (
    load_config, validate_dirs,
    load_synthesizer, model_exists, coerce_numeric, get_paths
)

# ── column definitions ────────────────────────────────────────────────────────
# Columns the SDV model is trained on (core physical measurements)
MODEL_COLS = [
    'PumpingFlowRate', 'OverallFlowRate', 'FilterDP', 'BatteryCharge',
    'AtmoT', 'AtmoP', 'AtmoRH', 'AtmoDensity',
    'PM1MC', 'PM2_5MC', 'PM4MC', 'PM10MC',
    'PM0_5NC', 'PM1NC', 'PM2_5NC', 'PM4NC', 'PM10NC',
    'PMtypicalParticleSize', 'MassFlow',
]

# Full ordered output column list (everything after SampleTime/datetime cols)
DATA_COLS = [
    'PumpingFlowRate', 'OverallFlowRate', 'SampledVolume', 'FilterDP',
    'BatteryCharge',
    'AtmoT', 'AtmoP', 'AtmoRH', 'AtmoDensity', 'AtmoAlt',
    'GPSQual', 'GPSlat', 'GPSlon', 'GPSalt', 'GPSsat', 'GPSspeed', 'GPShDOP',
    'AccelX', 'AccelXVar', 'AccelXMin', 'AccelXMax',
    'AccelY', 'AccelYVar', 'AccelYMin', 'AccelYMax',
    'AccelZ', 'AccelZVar', 'AccelZMin', 'AccelZMax',
    'RotX', 'RotXVar', 'RotXMin', 'RotXMax',
    'RotY', 'RotYVar', 'RotYMin', 'RotYMax',
    'RotZ', 'RotZVar', 'RotZMin', 'RotZMax',
    'Xup', 'XDown', 'Yup', 'Ydown', 'Zup', 'Zdown', 'StepCount',
    'LUX', 'UVindex', 'HighVisRaw', 'LowVisRaw', 'IRRaw', 'UVRaw',
    'PMMeasCnt',
    'PM1MC', 'PM1MCVar', 'PM2_5MC', 'PM2_5MCVar',
    'PM4MC', 'PM4MCVar', 'PM10MC', 'PM10MCVar',
    'PM0_5NC', 'PM0_5NCVar', 'PM1NC', 'PM1NCVar',
    'PM2_5NC', 'PM2_5NCVar', 'PM4NC', 'PM4NCVar',
    'PM10NC', 'PM10NCVar',
    'PMtypicalParticleSize', 'PMtypicalParticleSizeVar', 'PM2_5SampledMass',
    'PCB1T', 'PCB2T', 'FdpT', 'AccelT', 'PT100R', 'PCB2P',
    'PumpPow1', 'PumpPow2', 'PumpV', 'MassFlow', 'MFSVout', 'BFGenergy',
    'BattVolt', 'v3_3', 'v5',
    'PumpsON', 'Dead', 'BCS1', 'BCS2', 'BC_NPG', 'FLOWCTL', 'GPSRT',
    'SD_DATAW', 'SD_HEADW', 'TPumpsOFF', 'TPumpsON',
    'CO2', 'SCDT', 'SCDRH', 'VOCRaw', 'NOXRaw',
]

CATEGORY_ROW = (
    "DateTime,DateTime,DateTime,DateTime,DateTime,"
    "FilterSample,FilterSample,FilterSample,FilterSample,Battery,"
    "Atmo,Atmo,Atmo,Atmo,Atmo,"
    "GPS,GPS,GPS,GPS,GPS,GPS,GPS,"
    + ",".join(["Motion"] * 30) + ","
    + ",".join(["Light"] * 6) + ","
    + ",".join(["PMSensor"] * 23) + ","
    + ",".join(["EngData"] * 26) + ","
    + ",".join(["GasExperi"] * 5)
)

UNITS_ROW = (
    "(HH:MM:SS),(s),(s)"
    ",(YYYY-MM-DDTHH:MM:SS) (UTC date time format)"
    ",(YYYY-MM-DDTHH:MM:SS) (Local date time format),"
    "(L*min^-1),(L*min^-1),(L),(Pa),(%),"
    "(C),(hPa),(%RH),(g*L^-1),(m ASL),"
    "(-),(decimalDegree),(decimalDegree),(m),(integer),(m*s^-1),(-),"
    + ",".join(["(mg)"] * 12) + ","
    + ",".join(["(mdeg*s^-1)"] * 12) + ","
    "(%),(%),(%),(%),(%),(%),(#),"
    "(lux),(-),(-),(-),(-),(-),"
    "(#),"
    + ",".join(["(ug*m^-3)"] * 8) + ","
    + ",".join(["(#*cm^-3)"] * 10) + ","
    "(um),(um),(ug),"
    "(C),(C),(C),(C),(ohm),(hPa),(integer),(integer)"
    ",(V),(g*min^-1),(V),(integer),(V),(V),(V),"
    "(bool),(bool),(bool),(bool),(bool),(s),(s),(s),(s),(s),(s),"
    "(ppm),(C),(%),(-),(-)"
)

COL_NAMES_ROW = (
    "SampleTime,UnixTime,UnixTimeMCU,DateTimeUTC,DateTimeLocal,"
    + ",".join(DATA_COLS)
)

# ── reading helpers ───────────────────────────────────────────────────────────
def _find_col_line(filepath):
    """Return 0-based line index of the SampleTime column-name row."""
    with open(filepath, 'r', errors='replace') as fh:
        for i, line in enumerate(fh):
            if line.startswith('SampleTime'):
                return i
    raise ValueError(f"SampleTime header not found in {filepath}")


def _read_upas_data(filepath, usecols):
    """Read UPAS data skipping the metadata header and the units row."""
    col_line = _find_col_line(filepath)
    # Skip everything before col_line, and also the units row (col_line + 1)
    skip = list(range(col_line)) + [col_line + 1]
    df = pd.read_csv(filepath, skiprows=skip, low_memory=False)
    available = [c for c in usecols if c in df.columns]
    return df[available]


# ── fit ───────────────────────────────────────────────────────────────────────
def fit_upas(real_data_dir, models_dir):
    model_path    = Path(models_dir) / "upas_synthesizer.pkl"
    metadata_path = Path(models_dir) / "upas_metadata.json"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Fitting UPAS model from real data...")
    files = list(Path(real_data_dir).rglob("PSP*.txt"))
    assert files, f"No PSP*.txt files found in {real_data_dir}"

    dfs = []
    for f in files:
        try:
            df = _read_upas_data(f, MODEL_COLS)
            df = coerce_numeric(df)
            dfs.append(df)
            print(f"  ✓ {f.name}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {f.name}: {e} — skipping")

    data = pd.concat(dfs).reset_index(drop=True).dropna()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.save_to_json(str(metadata_path))
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    synthesizer.save(str(model_path))
    print(f"✓ UPAS model saved to {model_path}")
    return synthesizer

def load_upas(models_dir):
    return load_synthesizer(Path(models_dir) / "upas_synthesizer.pkl")


# ── filename ──────────────────────────────────────────────────────────────────
def make_upas_filename(serial, start, sample_name='SYNTHETIC', cartridge='C0000'):
    """PSP00054_LOG_2022-09-06T19_37_17UTC_E102____________C3798_____.txt"""
    ts   = pd.Timestamp(start).tz_localize('UTC')
    sn   = (sample_name + '_' * 15)[:15]
    cid  = (cartridge   + '_' * 10)[:10]
    return (f"PSP{serial}_LOG_{ts.strftime('%Y-%m-%dT%H_%M_%S')}UTC"
            f"_{sn}_{cid}.txt")


# ── header ────────────────────────────────────────────────────────────────────
def make_upas_header(serial, fname, start, n_days, utc_offset=-7.0):
    ts_start = pd.Timestamp(start).tz_localize('UTC')
    ts_end   = ts_start + pd.Timedelta(days=n_days)
    duration = n_days * 24

    return (
        f"PARAMETER,VALUE,UNITS/NOTES\n"
        f"\n"
        f"UPASserial,{serial},(UPAS serial identification-numerical)\n"
        f"UPASpcbRev,0,(UPAS pcb revision number)\n"
        f"UPASexpRev,1,(UPAS expansion pcb)\n"
        f"PMserial,SYNTHETIC_PM_SERIAL,(SPS30 serial identification)\n"
        f"UPASfirmware,UPAS_v2_x-rev_00127-L476RE_SYNTHETIC,(installed firmware version)\n"
        f"LifetimeSampleCount,1,(count-total lifetime sample runs)\n"
        f"LifetimeSampleRuntime,{duration:.2f},(hrs-total lifetime cumulative sample runtime)\n"
        f"\n\n\n\n"
        f"SAMPLE IDENTIFICATION\n"
        f"\n"
        f"LogFilename,/sd/{ts_start.strftime('%Y%m%d')}/{fname},(log file filename-automatically defined)\n"
        f"SampleName,SYNTHETIC,(Sample Name-user entered into app)\n"
        f"CartridgeID,C0000_____,(Cartridge Identification-user entered into app)\n"
        f"\n\n\n\n"
        f"SETUP SUMMARY\n"
        f"\n"
        f"GPSUTCOffset,{utc_offset:.2f},(hours offset from UTC date time)\n"
        f"StartOnNextPowerUp,0,(0=no 1=yes 2=system reset)\n"
        f"ProgrammedStartTime,0,(0 = Now or Start On Next or seconds since 1/1/1970)\n"
        f"ProgrammedRuntime,{duration:.3f},(Hr)\n"
        f"SizeSelectiveInlet,PM2.5,(inlet particle size fraction)\n"
        f"FlowRateSetpoint,1.000,(L*min^-1)\n"
        f"FlowOffset,0.000000,(%)\n"
        f"FlowDutyCycle,100,(%)\n"
        f"DutyCycleWindow,30,(s)\n"
        f"GPSEnabled,0,(0=no 1=yes)\n"
        f"PMSensorInterval,1,(0=sensor disabled 1=continuous measurement)\n"
        f"RTGasSampleState,0,(0=off 1=on)\n"
        f"CO2SampleState,0,(0=off 1=on)\n"
        f"LogInterval,30,(s)\n"
        f"PowerSaveMode,0,(0=off 1=on)\n"
        f"\n\n\n\n"
        f"SAMPLE SUMMARY\n"
        f"\n"
        f"StartDateTimeUTC,{ts_start.strftime('%Y-%m-%dT%H:%M:%S')},"
        f"(YYYY-MM-DDTHH:MM:SS) (UTC date time format)\n"
        f"StartDateTimeLocal,{(ts_start + pd.Timedelta(hours=utc_offset)).strftime('%Y-%m-%dT%H:%M:%S')},"
        f"(YYYY-MM-DDTHH:MM:SS) (Local date time format)\n"
        f"EndDateTimeUTC,{ts_end.strftime('%Y-%m-%dT%H:%M:%S')},"
        f"(YYYY-MM-DDTHH:MM:SS) (UTC date time format)\n"
        f"EndDateTimeLocal,{(ts_end + pd.Timedelta(hours=utc_offset)).strftime('%Y-%m-%dT%H:%M:%S')},"
        f"(YYYY-MM-DDTHH:MM:SS) (Local date time format)\n"
        f"OverallDuration, {duration:.3f},(Hr)\n"
        f"PumpingDuration, {duration:.3f},(Hr)\n"
        f"OverallFlowRateAverage,1.000,(L*min^-1)\n"
        f"PumpingFlowRateAverage,1.000,(L*min^-1)\n"
        f"SampledVolume, {duration * 60.0:.2f},(L)\n"
        f"StartBatteryCharge,099,(%)\n"
        f"EndBatteryCharge,100,(%)\n"
        f"ShutdownMode,01,(1=user pushbutton sample stop)\n"
        f"\n\n\n\n"
        f"MASS FLOW SENSOR CALIBRATION\n"
        f"\n"
        f"MFSCalDate,2022-04-18T19:12:00,(YYYY-MM-DDTHH:MM:SS) (UTC date time format)\n"
        f"MFSCalVoutMin,0.480000,(V)\n"
        f"MFSCalVoutMax,2.084125,(V)\n"
        f"MFSCalMFMin,0.010983,(g*min^-1)\n"
        f"MFSCalMFMax,3.387900,(g*min^-1)\n"
        f"MF4,0.134946,(coefficient)\n"
        f"MF3,0.131147,(coefficient)\n"
        f"MF2,-0.948272,(coefficient)\n"
        f"MF1,2.215893,(coefficient)\n"
        f"MF0,-0.855831,(coefficient)\n"
        f"\n\n\n\n"
        f"SAMPLE LOG\n"
        f"\n"
        f"{CATEGORY_ROW}\n"
        f"{COL_NAMES_ROW}\n"
        f"{UNITS_ROW}"
    )


# ── write ─────────────────────────────────────────────────────────────────────
def write_upas(df, serial, start, n_days, output_dir, utc_offset=-7.0):
    ts_start  = pd.Timestamp(start).tz_localize('UTC')
    ts_local  = ts_start + pd.Timedelta(hours=utc_offset)
    n_records = len(df)
    fname     = make_upas_filename(serial, start)
    header    = make_upas_header(serial, fname, start, n_days, utc_offset)

    rows = []
    cum_vol = 0.0
    for i, (_, row) in enumerate(df.iterrows()):
        elapsed   = pd.Timedelta(seconds=i * 30)
        h, rem    = divmod(int(elapsed.total_seconds()), 3600)
        m, s      = divmod(rem, 60)
        stime     = f"{h}:{m:02d}:{s:02d}"

        ts_utc    = ts_start + elapsed
        ts_loc    = ts_local + elapsed
        unix_t    = int(ts_utc.timestamp())

        # Derive sampled volume from flow rate
        flow = float(row.get('PumpingFlowRate', 1.0))
        cum_vol += flow * (30.0 / 60.0)   # L per 30s interval

        def _f(col, default=0.0, fmt='.3f'):
            v = row.get(col, default)
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return str(default)

        values = [
            stime,
            str(unix_t), str(unix_t),
            ts_utc.strftime('%Y-%m-%dT%H:%M:%S'),
            ts_loc.strftime('%Y-%m-%dT%H:%M:%S'),
            _f('PumpingFlowRate'),
            _f('OverallFlowRate'),
            f"{cum_vol:.3f}",
            _f('FilterDP', 300.0),
            str(int(float(row.get('BatteryCharge', 99)))),
            _f('AtmoT', 25.0),
            _f('AtmoP', 990.0),
            _f('AtmoRH', 40.0),
            _f('AtmoDensity', 1.15),
            _f('AtmoAlt', 200.0, '.1f'),
            # GPS (no fix — all empty)
            '', '', '', '', '', '', '',
            # IMU / motion (zero-fill)
        ] + ['0'] * 30 + [
            # Light
            '0', '0', '0', '0', '0', '0',
            # PM
            '30',
            _f('PM1MC'), '0.00',
            _f('PM2_5MC'), '0.00',
            _f('PM4MC'), '0.00',
            _f('PM10MC'), '0.00',
            _f('PM0_5NC'), '0.00',
            _f('PM1NC'), '0.00',
            _f('PM2_5NC'), '0.00',
            _f('PM4NC'), '0.00',
            _f('PM10NC'), '0.00',
            _f('PMtypicalParticleSize', 0.6),
            '0.00',
            _f('PM2_5SampledMass', 0.01, '.4f'),
            # Engineering data
            '30.0', '30.0', '30.0', '28.0', '112.0', '990.0',
            '562', '0',
            '4.20',
            _f('MassFlow', 1.13),
            '1.320', '52000',
            '4.20', '3.32', '5.03',
            '1', '0', '1', '1', '1',
            '0.000', '0.000', '0.000', '0.000', '0.000', '30.000',
            # Gas experiment (CO2/VOC off by default)
            '', '', '', '', '',
        ]

        rows.append(','.join(str(v) for v in values))

    out_path = Path(output_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write(header)
        fh.write('\n')
        fh.write('\n'.join(rows))
    return out_path


# ── edge cases ────────────────────────────────────────────────────────────────
def inject_edge_cases(df, rng):
    # negative PM (calibration artefact)
    neg_idx = rng.integers(10, len(df) - 10)
    df.loc[neg_idx, 'PM2_5MC'] = -0.1

    # high PM spike (smoke event)
    spike = rng.integers(100, len(df) - 50)
    df.loc[spike, 'PM2_5MC']  = rng.uniform(120, 250)
    df.loc[spike, 'PM10MC']   = df.loc[spike, 'PM2_5MC'] * 1.3

    # flow drop (filter loading)
    drop_start = rng.integers(200, len(df) - 30)
    df.loc[drop_start:drop_start+10, 'PumpingFlowRate'] = 0.85

    return df


# ── campaign ──────────────────────────────────────────────────────────────────
def generate_upas_campaign(real_data_dir, output_dir, models_dir,
                           n_sensors=3, n_days=7,
                           start='2022-09-15 08:00:00',
                           utc_offset=-7.0,
                           seed=42, force_refit=False):
    """
    Generate synthetic UPAS v2 data for multiple sensors.

    Parameters
    ----------
    real_data_dir : str   — path to real UPAS .txt files for model fitting
    output_dir    : str   — path to write synthetic files
    models_dir    : str   — path to model cache directory
    n_sensors     : int   — number of synthetic sensors
    n_days        : int   — deployment duration in days
    start         : str   — campaign start datetime (UTC)
    utc_offset    : float — UTC offset written into file header
    seed          : int   — random seed
    force_refit   : bool  — refit model even if cached
    """
    real_data_dir, output_dir = validate_dirs(real_data_dir, output_dir,
                                               ext='.txt')
    if force_refit or not model_exists(models_dir, 'upas'):
        synthesizer = fit_upas(real_data_dir, models_dir)
    else:
        synthesizer = load_upas(models_dir)

    n_records = n_days * 24 * 120   # 30-second intervals
    outputs   = []

    for i in range(n_sensors):
        serial    = f"{99000 + i + 1:05d}"
        rng       = np.random.default_rng(seed + i)
        synthetic = synthesizer.sample(num_rows=n_records)
        synthetic = coerce_numeric(synthetic)

        synthetic['PumpingFlowRate']  = synthetic['PumpingFlowRate'].clip(0.5, 2.0)
        synthetic['OverallFlowRate']  = synthetic['PumpingFlowRate']
        synthetic['FilterDP']         = synthetic['FilterDP'].clip(50, 600)
        synthetic['BatteryCharge']    = synthetic['BatteryCharge'].clip(0, 100)
        synthetic['AtmoT']            = synthetic['AtmoT'].clip(-10, 55)
        synthetic['AtmoP']            = synthetic['AtmoP'].clip(850, 1060)
        synthetic['AtmoRH']           = synthetic['AtmoRH'].clip(0, 100)
        synthetic['PM1MC']            = synthetic['PM1MC'].clip(0, 500)
        synthetic['PM2_5MC']          = synthetic['PM2_5MC'].clip(0, 500)
        synthetic['PM4MC']            = synthetic['PM4MC'].clip(0, 500)
        synthetic['PM10MC']           = synthetic['PM10MC'].clip(0, 500)
        synthetic['PMtypicalParticleSize'] = synthetic['PMtypicalParticleSize'].clip(0.1, 10)
        synthetic['MassFlow']         = synthetic['MassFlow'].clip(0.5, 2.0)

        synthetic = inject_edge_cases(synthetic, rng)

        out = write_upas(synthetic, serial, start=start, n_days=n_days,
                         output_dir=output_dir, utc_offset=utc_offset)
        outputs.append(out)
        print(f"✓ Sensor {i+1}/{n_sensors}: {out.name}")

    print(f"\n✓ Generated {n_sensors} UPAS sensors, "
          f"{n_days} days ({n_records} records each)")
    return outputs


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    paths  = get_paths(config)
    cfg    = config['campaign']
    inst   = config['instruments']['upas']

    if not inst['enabled']:
        print("UPAS disabled in config — skipping")
    else:
        generate_upas_campaign(
            real_data_dir = paths['real_data'] / 'upas',
            output_dir    = paths['output']    / 'upas',
            models_dir    = paths['models'],
            n_sensors     = cfg['n_sensors'],
            n_days        = cfg['n_days'],
            start         = cfg['start'],
            utc_offset    = cfg.get('utc_offset', -7.0),
            seed          = cfg['seed'],
            force_refit   = inst['force_refit'],
        )
