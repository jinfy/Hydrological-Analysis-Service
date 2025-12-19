import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import rasterio
from rasterio.features import shapes
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, unquote
import pandas as pd
import numpy as np
import math

## you have to set the env_result_dir for saving the temp file

def ensure_local_data(path_or_url, env_result_dir=r"D:\Work\Data\Paper_Projects\HydrologyLLM\project\hydro_service\temp"):
    s = str(path_or_url)
    # 本地文件，直接返回
    if not (s.startswith("http://") or s.startswith("https://")):
        return s

    env_result_dir = Path(env_result_dir)
    env_result_dir.mkdir(parents=True, exist_ok=True)

    # 从 URL 中解析文件名
    parsed = urlparse(s)
    filename = Path(unquote(parsed.path)).name
    if not filename:
        raise ValueError(f"Cannot determine filename from URL: {s}")

    local_path = env_result_dir / filename

    # 已存在则不重复下载
    # if local_path.exists():
    #     print("Use cached:", local_path)
    #     return str(local_path)

    print("Downloading:", s)
    urllib.request.urlretrieve(s, local_path)

    return str(local_path)



# ========= 固定配置（写死） =========
GRASS_HOME = Path(r"D:\Work\GRASS GIS 8.4")
RESULT_DIR_DEFAULT = Path(r"repository")  ## set the absolute path of the repository dictionary in this project
STREAM_THRESHOLD = 500

GRASS_CANDIDATES = [
    GRASS_HOME / "grass.bat",
    GRASS_HOME / "bin" / "grass.bat",
    GRASS_HOME / "grass84.bat",
    GRASS_HOME / "bin" / "grass84.bat",
]


# ========= 公共工具 =========
def _grass_cmd():
    for p in GRASS_CANDIDATES:
        if p.exists():
            return str(p)
    raise RuntimeError("GRASS not found")


GRASS_CMD = _grass_cmd()


def _run(cmd):
    subprocess.run(cmd, check=True)


def _kv(**kwargs):
    return [f"{k}={v}" for k, v in kwargs.items() if v is not None]


def _temp_mapset(geofile):
    tmp = Path(tempfile.mkdtemp(prefix="grass_tmp_"))
    loc = tmp / "loc"
    _run([GRASS_CMD, "-c", str(geofile), "-e", str(loc)])
    return loc / "PERMANENT", tmp


def _run_grass(mapset, module, *args):
    _run([GRASS_CMD, str(mapset), "--exec", module, *args])


# ========= 1. DEM 填洼 =========
def fill_dem(
    src_dem_tif,
    dem_filled_tif,
    env_result_dir=RESULT_DIR_DEFAULT,
):
    env_result_dir = Path(env_result_dir)
    env_result_dir.mkdir(parents=True, exist_ok=True)
    src_dem_tif = Path(ensure_local_data(src_dem_tif))
    dem_tif = env_result_dir / src_dem_tif.name

    shutil.copyfile(src_dem_tif, dem_tif)

    mapset, tmp = _temp_mapset(dem_tif)
    try:
        _run_grass(mapset, "r.import", "--overwrite",
                   *_kv(input=dem_tif, output="dem"))
        _run_grass(mapset, "g.region", *_kv(raster="dem"))
        _run_grass(mapset, "r.fill.dir", "--overwrite",
                   *_kv(input="dem", output="dem_filled", direction="dir"))

        if Path(dem_filled_tif).exists():
            Path(dem_filled_tif).unlink()

        _run_grass(mapset, "r.out.gdal", "--overwrite",
                   *_kv(input="dem_filled", output=dem_filled_tif,
                        format="GTiff", createopt="COMPRESS=DEFLATE"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ========= 2. 流向 =========
def flow_direction(
    dem_filled_tif,
    flowdir_tif,
    env_result_dir=RESULT_DIR_DEFAULT,
):
    env_result_dir = Path(env_result_dir)
    env_result_dir.mkdir(parents=True, exist_ok=True)
    dem_filled_tif = ensure_local_data(dem_filled_tif)


    mapset, tmp = _temp_mapset(dem_filled_tif)
    try:
        _run_grass(mapset, "r.import", "--overwrite",
                   *_kv(input=dem_filled_tif, output="dem_filled"))
        _run_grass(mapset, "g.region", *_kv(raster="dem_filled"))
        _run_grass(mapset, "r.watershed", "--overwrite",
                   *_kv(elevation="dem_filled", drainage="flowdir"))

        if Path(flowdir_tif).exists():
            Path(flowdir_tif).unlink()

        _run_grass(mapset, "r.out.gdal", "--overwrite",
                   *_kv(input="flowdir", output=flowdir_tif,
                        format="GTiff", createopt="COMPRESS=DEFLATE"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ========= 3. 汇流累积 =========
def flow_accumulation(
    dem_filled_tif,
    accum_tif,
    env_result_dir=RESULT_DIR_DEFAULT,
):
    env_result_dir = Path(env_result_dir)
    env_result_dir.mkdir(parents=True, exist_ok=True)
    dem_filled_tif = ensure_local_data(dem_filled_tif)


    mapset, tmp = _temp_mapset(dem_filled_tif)
    try:
        _run_grass(mapset, "r.import", "--overwrite",
                   *_kv(input=dem_filled_tif, output="dem_filled"))
        _run_grass(mapset, "g.region", *_kv(raster="dem_filled"))
        _run_grass(mapset, "r.watershed", "--overwrite",
                   *_kv(elevation="dem_filled", accumulation="accum"))

        if Path(accum_tif).exists():
            Path(accum_tif).unlink()

        _run_grass(mapset, "r.out.gdal", "--overwrite",
                   *_kv(input="accum", output=accum_tif,
                        format="GTiff", createopt="COMPRESS=DEFLATE"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ========= 4. 河网提取 =========
def extract_streams(
    dem_filled_tif,
    accum_tif,
    streams_tif,
    env_result_dir=RESULT_DIR_DEFAULT,
):
    env_result_dir = Path(env_result_dir)
    env_result_dir.mkdir(parents=True, exist_ok=True)
    
    dem_filled_tif = ensure_local_data(dem_filled_tif)
    accum_tif = ensure_local_data(accum_tif)


    mapset, tmp = _temp_mapset(dem_filled_tif)
    try:
        _run_grass(mapset, "r.import", "--overwrite",
                   *_kv(input=dem_filled_tif, output="dem_filled"))
        _run_grass(mapset, "r.import", "--overwrite",
                   *_kv(input=accum_tif, output="accum"))
        _run_grass(mapset, "g.region", *_kv(raster="dem_filled"))

        _run_grass(mapset, "r.stream.extract", "--overwrite",
                   *_kv(elevation="dem_filled",
                        accumulation="accum",
                        threshold=STREAM_THRESHOLD,
                        stream_raster="streams"))

        if Path(streams_tif).exists():
            Path(streams_tif).unlink()

        _run_grass(mapset, "r.out.gdal", "--overwrite",
                   *_kv(input="streams", output=streams_tif,
                        format="GTiff", createopt="COMPRESS=DEFLATE"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ========= 5. Raster → GeoJSON =========
def raster_to_geojson(
    raster_tif,
    geojson_path,
    env_result_dir=RESULT_DIR_DEFAULT,
):
    env_result_dir = Path(env_result_dir)
    env_result_dir.mkdir(parents=True, exist_ok=True)
    raster_tif = ensure_local_data(raster_tif)

    with rasterio.open(raster_tif) as ds:
        arr = ds.read(1)
        mask = arr == 1

        features = []
        for geom, val in shapes(arr, mask=mask, transform=ds.transform):
            if val == 1:
                features.append({
                    "type": "Feature",
                    "properties": {"value": 1},
                    "geometry": geom
                })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        if ds.crs:
            geojson["crs_wkt"] = ds.crs.to_wkt()

    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

# ========= 6. Xianjiang Model =========
"""
Xinanjiang Model - Daily Model Calculation

This module implements the daily version of the Xinanjiang hydrological model,
a conceptual rainfall-runoff model widely used in China for flood forecasting
and water resource management.

The model includes:
- Three-layer evapotranspiration calculation (upper, lower, deep layers)
- Runoff generation based on saturation excess mechanism
- Three-source separation (surface, interflow, groundwater)
- Flow routing using linear reservoir method

Author: Liuhuang
Date: 2024
"""

def calculate_daily_model(input_file: str, output_file: str) -> tuple:
    """
    Execute the Xinanjiang daily model calculation.
    
    Args:
        input_file: Path to input Excel file containing daily data
        output_file: Path to save output Excel file
        
    Returns:
        tuple: (runoff_depth_error, determination_coefficient)
    """
    # Read input data
    df = pd.read_excel(input_file)
    df = df.fillna(0)  # Handle missing values
    
    # ==================== Model Parameters ====================
    # Evapotranspiration coefficient
    K = 0.95
    
    # Soil moisture storage capacity parameters (mm)
    WUM = 20    # Upper layer storage capacity
    WLM = 60    # Lower layer storage capacity
    WDM = 40    # Deep layer storage capacity
    WM = WUM + WLM + WDM  # Total storage capacity
    b = 0.3     # Tension water capacity curve exponent
    C = 0.18    # Deep layer evapotranspiration coefficient
    FC = 2.5    # Stable infiltration rate
    WMM = WM * (1 + b)  # Maximum point storage capacity
    
    # Three-source separation parameters
    Sm = 30     # Free water storage capacity (mm)
    EX = 1.5    # Free water capacity curve exponent
    IM = 0.01   # Impervious area ratio
    KG = 0.2    # Groundwater outflow coefficient
    KI = 0.7 - KG  # Interflow outflow coefficient
    S1 = 0.5    # Initial free water storage ratio
    FR1 = 0.8   # Initial runoff producing area ratio
    SMM = Sm * (EX + 1)  # Maximum point free water capacity
    
    # Flow routing parameters
    CG = 0.99   # Groundwater recession coefficient
    CI = 0.5    # Interflow recession coefficient
    L = 1       # Lag time (days)
    CS = 0.4    # Channel routing coefficient
    Q = 0       # Initial discharge
    U = 290 / 3.6 / 24  # Unit conversion factor (km² to m³/s per mm)
    
    # ==================== Main Calculation Loop ====================
    for i in range(len(df['EP'])):
        # Initialize or update soil moisture storage
        if i == 0:
            # Initial soil moisture for three layers
            df.loc[i, ['WU']] = WUM
            df.loc[i, ['WL']] = WLM
            df.loc[i, ['WD']] = WDM
        else:
            # Calculate moisture change from previous time step
            dw = df['PE'][i-1] - df['R'][i-1]  # Net water change
            
            if dw >= 0:
                # Replenish soil moisture (from upper to lower to deep)
                if df['WU'][i-1] + dw > WUM:
                    df.loc[i, ['WU']] = WUM
                    if df['WL'][i-1] + dw - (WUM - df['WU'][i-1]) > WLM:
                        df.loc[i, ['WL']] = WLM
                        if df['WD'][i-1] + dw - (WUM - df['WU'][i-1]) - (WLM - df['WL'][i-1]) > WDM:
                            df.loc[i, ['WD']] = WDM
                        else:
                            df.loc[i, ['WD']] = df['WD'][i-1] + dw - (WUM - df['WU'][i-1]) - (WLM - df['WL'][i-1])
                    else:
                        df.loc[i, ['WL']] = df['WL'][i-1] + dw - (WUM - df['WU'][i-1])
                        df.loc[i, ['WD']] = df['WD'][i-1]
                else:
                    df.loc[i, ['WU']] = df['WU'][i-1] + dw
                    df.loc[i, ['WL']] = df['WL'][i-1]
                    df.loc[i, ['WD']] = df['WD'][i-1]
            else:
                # Deplete soil moisture due to evapotranspiration
                if df['WU'][i-1] - abs(dw) < 0:
                    df.loc[i, ['WU']] = 0
                    if df['WL'][i-1] - (abs(dw) - df['WU'][i-1]) < 0:
                        df.loc[i, ['WL']] = 0
                        if df['WD'][i-1] - (abs(dw) - df['WU'][i-1] - df['WL'][i-1]) < 0:
                            df.loc[i, ['WD']] = 0
                        else:
                            df.loc[i, ['WD']] = df['WD'][i-1] - (abs(dw) - df['WU'][i-1] - df['WL'][i-1])
                    else:
                        df.loc[i, ['WL']] = df['WL'][i-1] - (abs(dw) - df['WU'][i-1])
                        df.loc[i, ['WD']] = df['WD'][i-1]
                else:
                    df.loc[i, ['WU']] = df['WU'][i-1] - abs(dw)
                    df.loc[i, ['WL']] = df['WL'][i-1]
                    df.loc[i, ['WD']] = df['WD'][i-1]
        
        # Calculate total soil moisture
        df.loc[i, ['W']] = df['WU'][i] + df['WL'][i] + df['WD'][i]
        
        # Calculate potential evapotranspiration
        df.loc[i, ['EP']] = df['E0'][i] * K
        
        # ========== Three-layer Evapotranspiration Calculation ==========
        if df["WU"][i] + df["P"][i] >= df["EP"][i]:
            # Case 1: Upper layer can satisfy all evapotranspiration
            df.loc[i, ["EU"]] = df["EP"][i]
            df.loc[i, ["EL"]] = 0
            df.loc[i, ["ED"]] = 0
        elif df["WL"][i] >= C * WLM:
            # Case 2: Lower layer moisture is sufficient
            df.loc[i, ["EU"]] = df["WU"][i] + df["P"][i]
            df.loc[i, ["EL"]] = (df["EP"][i] - df["EU"][i]) * (df["WL"][i] / WLM)
            df.loc[i, ["ED"]] = 0
        elif C * (df["EP"][i] - df["EU"][i]) <= df["WL"][i] < C * WLM:
            # Case 3: Limited evapotranspiration from lower layer
            df.loc[i, ["EU"]] = df["WU"][i] + df["P"][i]
            df.loc[i, ["EL"]] = C * (df["EP"][i] - df["EU"][i])
            df.loc[i, ["ED"]] = 0
        else:
            # Case 4: Need to extract from deep layer
            df.loc[i, ["EU"]] = df["WU"][i] + df["P"][i]
            df.loc[i, ["EL"]] = df["WL"][i]
            df.loc[i, ["ED"]] = C * (df["EP"][i] - df["EU"][i]) - df["EL"][i]
        
        # Calculate actual evapotranspiration and net precipitation
        df.loc[i, ["E"]] = df['EU'][i] + df['EL'][i] + df['ED'][i]
        df.loc[i, ["PE"]] = df['P'][i] - df['E'][i]
        
        # ========== Runoff Generation Calculation ==========
        if df["PE"][i] > 0:
            # Calculate tension water storage curve ordinate
            a = WMM * (1 - math.pow(1 - (df['W'][i] / WM), 1 / (1 + b)))
            if a + df['PE'][i] <= WMM:
                runoff = df['PE'][i] + df['W'][i] - WM + WM * (1 - (df['PE'][i] + a) / WMM) ** (1 + b)
                df.loc[i, ["R"]] = max(runoff, 0)
            else:
                runoff = df['PE'][i] + df['W'][i] - WM
                df.loc[i, ["R"]] = max(runoff, 0)
        else:
            df.loc[i, ["R"]] = 0
        
        # ========== Runoff Producing Area Calculation ==========
        if i == 0:
            if df.loc[i, "R"] > 0:
                df.loc[i, "FR"] = min(df.loc[i, "R"] / df.loc[i, "PE"], 1)
            else:
                df.loc[i, "FR"] = FR1
        else:
            if df.loc[i, "R"] > 0:
                df.loc[i, "FR"] = min(df.loc[i, "R"] / df.loc[i, "PE"], 1)
            else:
                df.loc[i, "FR"] = df.loc[i-1, "FR"]
        
        # ========== Three-source Separation ==========
        if i == 0:
            df.loc[i, "S1"] = S1
            if df.loc[i, "PE"] > 0:
                # Calculate free water storage curve ordinate
                AU = SMM * (1 - math.pow(1 - (((df.loc[i, "S1"] * FR1) / df.loc[i, "FR"]) / Sm), 1 / (1 + EX)))
                if df.loc[i, "PE"] + AU < SMM:
                    df.loc[i, "RS"] = df.loc[i, "FR"] * (df.loc[i, "PE"] + (df.loc[i, "S1"] * FR1) / df.loc[i, "FR"] - Sm +
                                                         Sm * math.pow(1 - (df.loc[i, "PE"] + AU) / SMM, 1 + EX))
                else:
                    df.loc[i, "RS"] = df.loc[i, "FR"] * (df.loc[i, "PE"] + (df.loc[i, "S1"] * FR1) / df.loc[i, "FR"] - Sm)
                S = (df.loc[i, "S1"] * FR1) / df.loc[i, "FR"] + (df.loc[i, "R"] - df.loc[i, "RS"]) / df.loc[i, "FR"]
                df.loc[i, "RI"] = KI * S * df.loc[i, "FR"]  # Interflow
                df.loc[i, "RG"] = KG * S * df.loc[i, "FR"]  # Groundwater
                df.loc[i + 1, "S1"] = S * (1 - KI - KG)
            else:
                S = (df.loc[i, "S1"] * FR1) / df.loc[i, "FR"]
                df.loc[i + 1, "S1"] = S * (1 - KG - KI)
                df.loc[i, "RS"] = 0  # Surface runoff
                df.loc[i, "RG"] = KI * S * df.loc[i, "FR"]
                df.loc[i, "RI"] = KG * S * df.loc[i, "FR"]
        else:
            if df.loc[i, "PE"] > 0:
                AU = SMM * (1 - math.pow(1 - (((df.loc[i, "S1"] * df.loc[i-1, "FR"]) / df.loc[i, "FR"]) / Sm), 1 / (1 + EX)))
                # AU = SMM * (1 - math.pow((1 - (((df["S1"][i] * FR1) / df["FR"][i]) / Sm)), 1 / (1 + EX)))

                if df.loc[i, "PE"] + AU < SMM:
                    df.loc[i, "RS"] = df.loc[i, "FR"] * (df.loc[i, "PE"] + (df.loc[i, "S1"] * df.loc[i-1, "FR"]) / df.loc[i, "FR"] - Sm +
                                                         Sm * math.pow(1 - (df.loc[i, "PE"] + AU) / SMM, 1 + EX))
                else:
                    df.loc[i, "RS"] = df.loc[i, "FR"] * (df.loc[i, "PE"] + (df.loc[i, "S1"] * df.loc[i-1, "FR"]) / df.loc[i, "FR"] - Sm)
                S = (df.loc[i, "S1"] * df.loc[i-1, "FR"]) / df.loc[i, "FR"] + (df.loc[i, "R"] - df.loc[i, "RS"]) / df.loc[i, "FR"]
                df.loc[i, "RI"] = KI * S * df.loc[i, "FR"]
                df.loc[i, "RG"] = KG * S * df.loc[i, "FR"]
                df.loc[i + 1, "S1"] = S * (1 - KI - KG)
            else:
                S = (df.loc[i, "S1"] * df.loc[i-1, "FR"]) / df.loc[i, "FR"]
                df.loc[i + 1, "S1"] = S * (1 - KG - KI)
                df.loc[i, "RS"] = 0
                df.loc[i, "RG"] = KG * S * df.loc[i, "FR"]
                df.loc[i, "RI"] = KI * S * df.loc[i, "FR"]
        
        # ========== Flow Routing (Three Linear Reservoirs) ==========
        if i == 0:
            df.loc[i, "QS"] = df.loc[i, "RS"] * U  # Surface flow
            df.loc[i, "QI"] = 1 / 3 * Q  # Interflow
            df.loc[i, "QG"] = 1 / 3 * Q  # Groundwater flow
            df.loc[i, "QT"] = df.loc[i, "QS"] + df.loc[i, "QI"] + df.loc[i, "QG"]
        else:
            df.loc[i, "QS"] = df.loc[i, "RS"] * U
            df.loc[i, "QI"] = CI * df.loc[i-1, "QI"] + (1 - CI) * df.loc[i, "RI"] * U
            df.loc[i, "QG"] = CG * df.loc[i-1, "QG"] + (1 - CG) * df.loc[i, "RG"] * U
            df.loc[i, "QT"] = df.loc[i, "QS"] + df.loc[i, "QI"] + df.loc[i, "QG"]
        
        # Channel routing with lag
        if 0 <= i <= L:
            df.loc[i, "Qt"] = Q
        else:
            df.loc[i, "Qt"] = df.loc[i-1, "Qt"] * CS + (1 - CS) * df.loc[i - L, "QT"]
    
    # ==================== Error Calculation ====================
    # Runoff depth error (%)
    Q_error = 100 * (sum(df['Qt'][:-1]) - sum(df['Q'][:-1])) / sum(df['Q'])
    
    # Nash-Sutcliffe efficiency coefficient (Determination Coefficient)
    err_sum, off_sum = 0, 0
    avg = np.mean(df['Q'][:-1])
    for i in range(len(df['EP'][:-1])):
        err_sum += (df['Qt'][i] - df['Q'][i]) ** 2
        off_sum += (df['Q'][i] - avg) ** 2
    DC = 1 - err_sum / off_sum
    
    print(f"Runoff Depth Error: {Q_error:.2f}%")
    print(f"Nash-Sutcliffe Efficiency: {DC:.2f}")
    
    # Save results
    df = np.round(df, 2)
    df.to_excel(output_file)
    
    return Q_error, DC

def xinanjiangmodel(
    meteorological_data_table, 
    model_result_table, 
    env_result_dir=RESULT_DIR_DEFAULT
):
    local_data_table = ensure_local_data(meteorological_data_table)
    env_result_dir = Path(env_result_dir)
    env_result_dir.mkdir(parents=True, exist_ok=True)
    calculate_daily_model(local_data_table, model_result_table)
     