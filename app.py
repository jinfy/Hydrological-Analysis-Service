import os
import traceback
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# 你的 service.py（确保 app.py 与 service.py 在同一目录；否则改成正确的包路径）
import service
import utils

from urllib.parse import quote

# 你的 httpserver 地址（固定）
PUBLIC_BASE_URL = "http://127.0.0.1:8010"

def file_url(env_result_dir: str, file_path: str) -> str:
    job_tail = Path(env_result_dir).name
    fname = Path(file_path).name
    # URL 编码，避免中文/空格
    return f"{PUBLIC_BASE_URL}/{quote(job_tail)}/{quote(fname)}"

def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ok(data=None, **kwargs):
    resp = {"ok": True, "data": data}
    resp.update(kwargs)
    return jsonify(resp), 200


def _bad(message, code=400, **kwargs):
    resp = {"ok": False, "error": {"message": message}}
    resp["error"].update(kwargs)
    return jsonify(resp), code


def _safe_path(s: str) -> str:
    # 尽量规范化输出路径（便于前端/调用端展示）
    return str(Path(s).resolve())


def _default_outdir():
    # 统一输出目录：优先使用 service.py 的 RESULT_DIR_DEFAULT；否则 ./result
    base = getattr(service, "RESULT_DIR_DEFAULT", Path("./result"))
    return Path(base)


def _make_job_dir(base: Path):
    # 每次请求单独建一个目录，避免并发冲突/文件覆盖
    job_dir = base / f"job_{_now_tag()}"
    return _ensure_dir(job_dir)


def _get_mode():
    # mode=path|json
    mode = request.args.get("mode", "path").strip().lower()
    return mode if mode in ("path", "json") else "path"


def _get_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}
    # 允许 form 方式（比如 Postman x-www-form-urlencoded）
    return dict(request.form or {})


app = Flask(__name__)
# 允许跨域：默认全放开；如需限制域名再改 resources/origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)


@app.get("/health")
def health():
    return _ok({"status": "up"})


# ========= 1. DEM 填洼 =========
@app.post("/v1/dem/fill")
def api_fill_dem():
    """
    body:
      {
        "src_dem_tif": "local_path_or_url",
        "dem_filled_tif": "optional output path",
        "env_result_dir": "optional base dir"
      }
    """
    payload = _get_payload()
    try:
        src_dem_tif = payload.get("src_dem_tif")
        if not src_dem_tif:
            return _bad("Missing field: src_dem_tif")

        base_dir = Path(payload.get("env_result_dir") or _default_outdir())
        job_dir = _make_job_dir(base_dir)

        dem_filled_tif = payload.get("dem_filled_tif")
        if not dem_filled_tif:
            dem_filled_tif = str(job_dir / "dem_filled.tif")

        service.fill_dem(
            src_dem_tif=src_dem_tif,
            dem_filled_tif=dem_filled_tif,
            env_result_dir=job_dir,
        )
        return _ok({
            "dem_filled_tif": _safe_path(dem_filled_tif),
            "dem_filled_url": file_url(str(job_dir), dem_filled_tif),
            "job_dir": _safe_path(job_dir),
        })
    except Exception as e:
        return _bad(str(e), code=500, trace=traceback.format_exc())


# ========= 2. 流向 =========
@app.post("/v1/hydro/flow-direction")
def api_flow_direction():
    """
    body:
      {
        "dem_filled_tif": "local_path_or_url",
        "flowdir_tif": "optional output path",
        "env_result_dir": "optional base dir"
      }
    """
    payload = _get_payload()
    try:
        dem_filled_tif = payload.get("dem_filled_tif")
        if not dem_filled_tif:
            return _bad("Missing field: dem_filled_tif")

        base_dir = Path(payload.get("env_result_dir") or _default_outdir())
        job_dir = _make_job_dir(base_dir)

        flowdir_tif = payload.get("flowdir_tif")
        if not flowdir_tif:
            flowdir_tif = str(job_dir / "flowdir.tif")

        service.flow_direction(
            dem_filled_tif=dem_filled_tif,
            flowdir_tif=flowdir_tif,
            env_result_dir=job_dir,
        )
        return _ok({
            "flowdir_tif": _safe_path(flowdir_tif),
            "flowdir_url": file_url(str(job_dir), flowdir_tif),
            "job_dir": _safe_path(job_dir),
        })

    except Exception as e:
        return _bad(str(e), code=500, trace=traceback.format_exc())


# ========= 3. 汇流累积 =========
@app.post("/v1/hydro/flow-accumulation")
def api_flow_accumulation():
    """
    body:
      {
        "dem_filled_tif": "local_path_or_url",
        "accum_tif": "optional output path",
        "env_result_dir": "optional base dir"
      }
    """
    payload = _get_payload()
    try:
        dem_filled_tif = payload.get("dem_filled_tif")
        if not dem_filled_tif:
            return _bad("Missing field: dem_filled_tif")

        base_dir = Path(payload.get("env_result_dir") or _default_outdir())
        job_dir = _make_job_dir(base_dir)

        accum_tif = payload.get("accum_tif")
        if not accum_tif:
            accum_tif = str(job_dir / "accum.tif")

        service.flow_accumulation(
            dem_filled_tif=dem_filled_tif,
            accum_tif=accum_tif,
            env_result_dir=job_dir,
        )
        return _ok({
            "accum_tif": _safe_path(accum_tif),
            "accum_url": file_url(str(job_dir), accum_tif),
            "job_dir": _safe_path(job_dir),
        })

    except Exception as e:
        return _bad(str(e), code=500, trace=traceback.format_exc())


# ========= 4. 河网提取 =========
@app.post("/v1/hydro/streams")
def api_extract_streams():
    """
    body:
      {
        "dem_filled_tif": "local_path_or_url",
        "accum_tif": "local_path_or_url",
        "streams_tif": "optional output path",
        "env_result_dir": "optional base dir",
        "stream_threshold": 500 (optional, override global)
      }
    """
    payload = _get_payload()
    try:
        dem_filled_tif = payload.get("dem_filled_tif")
        accum_tif = payload.get("accum_tif")
        if not dem_filled_tif:
            return _bad("Missing field: dem_filled_tif")
        if not accum_tif:
            return _bad("Missing field: accum_tif")

        # 可选：临时覆盖 service.STREAM_THRESHOLD（不改你原逻辑也行）
        if payload.get("stream_threshold") is not None:
            try:
                service.STREAM_THRESHOLD = int(payload["stream_threshold"])
            except Exception:
                return _bad("stream_threshold must be int")

        base_dir = Path(payload.get("env_result_dir") or _default_outdir())
        job_dir = _make_job_dir(base_dir)

        streams_tif = payload.get("streams_tif")
        if not streams_tif:
            streams_tif = str(job_dir / "streams.tif")

        service.extract_streams(
            dem_filled_tif=dem_filled_tif,
            accum_tif=accum_tif,
            streams_tif=streams_tif,
            env_result_dir=job_dir,
        )
        return _ok({
            "streams_tif": _safe_path(streams_tif),
            "streams_url": file_url(str(job_dir), streams_tif),
            "job_dir": _safe_path(job_dir),
        })

    except Exception as e:
        return _bad(str(e), code=500, trace=traceback.format_exc())


# ========= 5. Raster → GeoJSON =========
@app.post("/v1/convert/raster-to-geojson")
def api_raster_to_geojson():
    """
    body:
      {
        "raster_tif": "local_path_or_url",
        "geojson_path": "optional output path",
        "env_result_dir": "optional base dir"
      }

    query:
      mode=path|json
    """
    payload = _get_payload()
    mode = _get_mode()

    try:
        raster_tif = payload.get("raster_tif")
        if not raster_tif:
            return _bad("Missing field: raster_tif")

        base_dir = Path(payload.get("env_result_dir") or _default_outdir())
        job_dir = _make_job_dir(base_dir)

        geojson_path = payload.get("geojson_path")
        if not geojson_path:
            geojson_path = str(job_dir / "streams.geojson")

        service.raster_to_geojson(
            raster_tif=raster_tif,
            geojson_path=geojson_path,
            env_result_dir=job_dir,
        )

        if mode == "json":
            import json
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            return _ok({
                "geojson": gj,
                "geojson_path": _safe_path(geojson_path),
                "job_dir": _safe_path(job_dir),
            })

        return _ok({
            "geojson_path": _safe_path(geojson_path),
            "geojson_url": file_url(str(job_dir), geojson_path),
            "job_dir": _safe_path(job_dir),
        })

    except Exception as e:
        return _bad(str(e), code=500, trace=traceback.format_exc())


# ========= 6. future runoff prediction =========
@app.post("/v1/hydro/future-runoff-prediction")
def api_runoff_prediction():
    """
    body:
      {
        "rainfall_data_table": "local_path_or_url",
        "future_rainoff_prediction_table": "optional output path",
        "env_result_dir": "optional base dir"
      }

    query:
      mode=path|json
    """
    payload = _get_payload()
    # print(payload)
    try:
        rainfall_data_table = payload.get("rainfall_data_table")
        if not rainfall_data_table:
            return _bad("Missing field: rainfall_data_table")

        base_dir = Path(payload.get("env_result_dir") or _default_outdir())
        job_dir = _make_job_dir(base_dir)

        future_rainoff_prediction_table = payload.get("future_rainoff_prediction_table")
        if not future_rainoff_prediction_table:
            future_rainoff_prediction_table = str(job_dir / "prediction.xlsx")

        service.xinanjiangmodel(
            meteorological_data_table=rainfall_data_table,
            model_result_table=future_rainoff_prediction_table,
            env_result_dir=job_dir,
        )

        return _ok({
            "prediction_result_path": _safe_path(future_rainoff_prediction_table),
            "prediction_result_url": file_url(str(job_dir), future_rainoff_prediction_table),
            "job_dir": _safe_path(job_dir),
        })

    except Exception as e:
        return _bad(str(e), code=500, trace=traceback.format_exc())

@app.post("/generate")
def generate():
    """
    body:
    {
        "query": "extracting the 2024 river network for Shilou county"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = data.get("query")
        # print(query)
        if not isinstance(query, str) or not query.strip():
            return jsonify({
                "issucceed": 0,
                "error": "Missing or invalid field: query"
            }), 400
        count = 0 ## debug once
        is_succeed = False
        
        while not is_succeed:
            try:
                resources = []
                standard_answer = utils.analyse_user_demand(query)
                print("standard_answer", standard_answer)
                result = utils.generate_abstract_workflow(standard_answer)
                if isinstance(result, dict):
                    resources = result
                    # print(result)
                    service_chain_info = utils.build_merged_edges(resources)
                    resources["service_chain_info"] = service_chain_info
                    bpmn_url = utils.generate_bpmn(service_chain_info)
                    resources["bpmn_url"] = bpmn_url
                    break
                else:
                    edges_service_level = utils.match_processing_services(result)
                    resources = utils.execute_service_chain(edges_service_level, standard_answer)
                    service_chain_info = utils.build_merged_edges(resources)
                    resources["service_chain_info"] = service_chain_info 

                    bpmn_url = utils.generate_bpmn(service_chain_info)
                    resources["bpmn_url"] = bpmn_url
                    print(resources)

                    if len(resources["resources"])<len(edges_service_level):
                        is_succeed = False
                    else:
                        is_succeed = True
                        break
            except Exception as e:
                print(e)
                if count>=1:
                    break
                else:
                    count = count +1
            # utils.clear_temp_dir(r"D:\Work\Data\Paper_Projects\HydrologyLLM\project\hydro_service\temp")
             
        return jsonify({
            "issucceed": 1,
            "resources": resources
        })

    except Exception as e:
        return jsonify({
            "issucceed": 0,
            "error": str(e)
        }), 500
    

if __name__ == "__main__":
    # Windows 上建议 host=0.0.0.0 方便局域网访问；不需要可改为 127.0.0.1
    app.run(host="0.0.0.0", port=8000, debug=True)
