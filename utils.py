from chat_with_llm import interact_with_llm
import json
import csv
import ast
from http import HTTPStatus
from sklearn.metrics.pairwise import cosine_similarity
import dashscope
import numpy as np
import json
from collections import defaultdict, deque
from pathlib import Path
import requests
import shutil
import os
from urllib.parse import urlparse
from collections import defaultdict, deque
import time
from datetime import datetime



EMB_API_KEY =  "..."  ## set your key to use the mebedding model (model = dashscope.TextEmbedding.Models.text_embedding_v2)
PROCESSING_EMB_CSV = r"...\processing_service_embedding.csv" ## replace to your processing_service_embedding.csv in the root dict
DATA_EMB_CSV = r"...\data_service_embedding.csv" ## replace to your data_service_embedding.csv in the root dict
MODEL_EMB_CSV = r"...\model_service_embedding.csv" ## replace to your model_service_embedding.csv in the root dict
PROCESSING_JSON = r"...\processing_service.json"  ## replace to your processing_service.json in the root dict
FLASK_BASE = "http://127.0.0.1:8000"
PUBLIC_BASE_URL = "http://127.0.0.1:8010"
LLM_MODEL = "deepseek-v3"


def match_service_by_embedding(
    query_text: str,
    emb_csv: str,
    api_key: str,
    model: str = None,
):
    """
    根据 query_text 与 xx_service_embedding.csv 中的 embedding
    进行相似度匹配，返回最匹配的 service_name。

    返回：
      {
        "service_name": str,
        "similarity": float
      }
    """
    if model is None:
        model = dashscope.TextEmbedding.Models.text_embedding_v2

    dashscope.api_key = api_key

    # ---- 1) 读取服务 embedding ----
    service_names = []
    service_embs = []

    with open(emb_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            service_names.append(row["service_name"])
            emb_str = row["embedding"]
            try:
                emb = json.loads(emb_str)
            except Exception:
                emb = ast.literal_eval(emb_str)
            service_embs.append(emb)

    if not service_names:
        raise ValueError(f"No service embeddings loaded from {emb_csv}")

    service_embs = np.array(service_embs, dtype=np.float32)

    # ---- 2) 对 query_text 做 embedding ----
    resp = dashscope.TextEmbedding.call(model=model, input=query_text)
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Embedding failed: {resp}")

    query_emb = np.array(
        resp["output"]["embeddings"][0]["embedding"],
        dtype=np.float32
    ).reshape(1, -1)

    # ---- 3) 余弦相似度匹配 ----
    sim = cosine_similarity(query_emb, service_embs)[0]
    idx = int(sim.argmax())

    return {
        "service_name": service_names[idx],
        "similarity": float(sim[idx]),
    }

def analyse_user_demand(query):
    prompt = '''
    Role:
    You are an expert in GIS and hydrological analysis and geospatial data processing, with strong capability in interpreting natural-language requests and extracting structured hydrological task information.

    Task:
    Based on the following user query, identify and extract the key elements required for hydrological analysis, including:
    - Location information (administrative region or spatial extent)
    - Time information (year, period, or temporal scope)
    - Functional requirement (e.g., watershed boundary delineation, runoff simulation, reservoir capacity calculation)
    - Expected output (e.g., watershed boundary)
    - Additional conditions or constraints

    Output Format:
    Return the extracted information strictly in the following JSON structure:
    {{
    "Location": "",
    "Time": "",
    "Requirement": "",
    "OutputType": "",
    "AdditionalInfo": "",
    "MissingInfo": ""
    }}

    Output Constraint:
    The response must contain ONLY the JSON object defined above.
    Do NOT include any explanations, reasoning steps, comments, or additional text outside the JSON.

    Guidance:
    1. Perform reasoning step by step before producing the final output.
    2. Do not fabricate administrative regions, temporal ranges, or hydrological phenomena.
    3. If spatial or temporal information is not explicitly provided, infer only when reasonable; otherwise mark the field as "Unknown".
    4. Ensure the extracted requirement corresponds to standard hydrological analysis functions.
    5. Ensure the output is consistent, machine-readable, and strictly follows the JSON schema.
    6. Avoid adding operations not explicitly requested by the user.
    7. When multiple interpretations exist, choose the most typical hydrological analysis meaning and explain it in "AdditionalInfo" if necessary.

    User Query:
    {}
    '''

    request = prompt.format(query)

    standard_answer = interact_with_llm(request)
    return(standard_answer)

def generate_abstract_workflow(standard_answer):
    prompt = ''' You are an expert in hydrological analysis. Interpret the user natural-language request and extract key task information for structured analysis. 

        Task: {} 

        Instructions: 
            1. Think step by step. 
            2. Based on the Requirement, list the functional modules needed to complete the task. Each functional module MUST be a short operation-level description that can be directly mapped to ArcGIS / arcpy geoprocessing operations. 
            The description should be concise and implementation-oriented, such as “fill sinks in DEM”, “derive flow direction raster”,Do NOT use vague or conceptual descriptions such as “define stream network”, 
            and do NOT include explanatory wording such as by, using, or for. 
            3. **Make sure each selected functional module corresponds to exactly ONE explicit arcpy geoprocessing tool (one module–one tool mapping).** Each functional module description MUST be implementation-oriented and tool-ready, 
            and MUST be directly translatable into a single arcpy function call. Composite, abstract, or multi-step semantic descriptions are NOT allowed; if a module cannot be mapped to one specific arcpy tool, it must NOT be used. 
            4. Convert these modules into an abstract workflow represented as directed tuples: (A, B). 
            5. Follow the one-shot example strictly in terms of output structure and format. 
            6. Output the results using the following structure, and ONLY output the following two parts: Functional Modules: a JSON object with a single field "nodes" (a list of Module short descr strings). 
            7. a "derive stream raster" step must be included for river network generation.
            Abstract Workflow (directed tuples): a JSON object with a single field "edges" (a list of directed pairs [A, B]). Output Format (strict)
        Functional Modules:
        ```json
            {{"nodes": "Module short descr1","Module short descr2","Module short descr3",...]]}}
        ```
        Abstract Workflow (directed tuples):
        ```json
            {{"edges": [["Module short descr1","Module short descr2"], ["Module short descr2","Module short descr3",...]]}}
        ```
        Output Constraint: 
            ONLY output the two parts in the Output Format. Do not output any other text. 

    '''
    standard_requirement = json.loads(standard_answer)["Requirement"]
    matched_model = match_service_by_embedding(
                query_text=standard_requirement,
                emb_csv=MODEL_EMB_CSV,
                api_key=EMB_API_KEY,
            )
    # print(111,matched_model["similarity"])
    if matched_model["similarity"]>=0.5:
        service = matched_model["service_name"]
        return execute_service_chain(service, standard_answer)
    else:
        request = prompt.format(
            standard_requirement
        )
        RAW_TEXT = interact_with_llm(request,"deepseek-v3")
        # print(RAW_TEXT)
        return RAW_TEXT

def extract_nodes_edges_by_anchor(raw_text: str):
    """
    假设 raw_text 中包含两段固定格式：
      ```json { "nodes": [...] }```
      ```json { "edges": [...] }```
    用 'json' 作为锚点分割，然后 json.loads 解析。
    """
    parts = raw_text.split("json")
    if len(parts) < 3:
        raise ValueError("Cannot split two JSON blocks by 'json' anchor. Check raw_text format.")

    nodes_str = parts[1].strip()
    edges_str = parts[2].strip()

    def to_json_obj(s: str):
        l = s.find("{")
        r = s.rfind("}")
        if l == -1 or r == -1 or r <= l:
            raise ValueError("JSON braces not found in anchored block.")
        return json.loads(s[l:r + 1])

    nodes_obj = to_json_obj(nodes_str)
    edges_obj = to_json_obj(edges_str)

    if "nodes" not in nodes_obj:
        raise ValueError("nodes JSON missing 'nodes' key.")
    if "edges" not in edges_obj:
        raise ValueError("edges JSON missing 'edges' key.")

    return nodes_obj["nodes"], edges_obj["edges"]

def load_service_embeddings(csv_path: str):
    """
    读取 processing_service_embedding.csv
    列：service_name, embedding_text, embedding
    embedding 可能是 JSON list 字符串，也可能是 Python list 字符串
    """
    names = []
    embs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["service_name"])
            emb_str = row["embedding"]
            try:
                emb = json.loads(emb_str)
            except Exception:
                emb = ast.literal_eval(emb_str)
            embs.append(emb)

    if not names:
        raise ValueError(f"No service embeddings loaded from {csv_path}")

    return names, np.array(embs, dtype=np.float32)

def _embed_text(text: str, model: str):
    resp = dashscope.TextEmbedding.call(model=model, input=text)
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Embedding failed: {resp}")
    return resp["output"]["embeddings"][0]["embedding"]

def _embed_nodes(nodes, model: str, verbose: bool = False):
    node_embs = []
    for n in nodes:
        e = _embed_text(n, model)
        node_embs.append(e)
        if verbose:
            print(f"[EMB] {n}")
    return np.array(node_embs, dtype=np.float32)

def build_procesing_service_graph_from_raw_text(
    raw_text: str,
    emb_csv: str,
    api_key: str,
    model: str = None,
    verbose: bool = True,
):
    """
    输入：
      - raw_text: 含两段 ```json {nodes...}``` + ```json {edges...}``` 的文本
      - emb_csv: processing_service_embedding.csv 路径
      - api_key: dashscope api_key
      - model: embedding 模型（默认 text_embedding_v2）
    输出：
      dict:
        {
          "node_to_service": [{"node","service","similarity"}, ...],
          "edges_node_level": [...],
          "edges_service_level": [...]
        }
    """
    if model is None:
        model = dashscope.TextEmbedding.Models.text_embedding_v2

    dashscope.api_key = api_key

    # 1) parse
    nodes, edges = extract_nodes_edges_by_anchor(raw_text)

    # 2) load service embeddings
    service_names, service_embs = load_service_embeddings(emb_csv)

    # 3) embed nodes
    node_embs = _embed_nodes(nodes, model=model, verbose=verbose)

    # 4) cosine similarity
    sim = cosine_similarity(node_embs, service_embs)

    # 5) node -> best service
    node_to_service = {}
    node_to_score = {}
    for i, n in enumerate(nodes):
        j = int(sim[i].argmax())
        node_to_service[n] = service_names[j]
        node_to_score[n] = float(sim[i, j])
        if verbose:
            print(f"[MATCH] {n}  ==>  {service_names[j]}  (cos={sim[i, j]:.4f})")

    # 6) edges: node-level -> service-level
    service_edges = []
    for a, b in edges:
        sa = node_to_service.get(a, a)
        sb = node_to_service.get(b, b)
        service_edges.append([sa, sb])

    # 7) 删除自循环 + 去重（保持顺序）
    seen = set()
    service_edges_optimized = []
    for sa, sb in service_edges:
        if sa == sb:
            continue
        key = (sa, sb)
        if key in seen:
            continue
        seen.add(key)
        service_edges_optimized.append([sa, sb])

    out = {
        "node_to_service": [
            {"node": n, "service": node_to_service[n], "similarity": node_to_score[n]}
            for n in nodes
        ],
        "edges_node_level": edges,
        "edges_service_level": service_edges_optimized,
    }

    # if verbose:
    #     print("\n===== RESULT JSON =====")
    #     print(json.dumps(out, ensure_ascii=False, indent=2))

    return out

def match_processing_services(RAW_TEXT):

    result = build_procesing_service_graph_from_raw_text(
        raw_text=RAW_TEXT,
        emb_csv=PROCESSING_EMB_CSV,
        api_key=EMB_API_KEY,
        verbose=True
    )
    edges_service_level = result["edges_service_level"]
    # print(edges_service_level)
    return edges_service_level

def get_api_endpoint():
    API_ENDPOINT = {
        "fill_dem": "/v1/dem/fill",
        "flow_direction": "/v1/hydro/flow-direction",
        "flow_accumulation": "/v1/hydro/flow-accumulation",
        "extract_streams": "/v1/hydro/streams",
        "raster_to_geojson": "/v1/convert/raster-to-geojson",
        "Xinanjiang model":"/v1/hydro/future-runoff-prediction"
    }
    return API_ENDPOINT

def topo_order(edges):
    nodes = set()
    indeg = defaultdict(int)
    g = defaultdict(list)

    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
        g[u].append(v)
        indeg[v] += 1
        indeg[u] += 0

    q = deque([n for n in nodes if indeg[n] == 0])
    order = []

    while q:
        n = q.popleft()
        order.append(n)
        for nxt in g[n]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(order) != len(nodes):
        raise ValueError(f"EDGES_SERVICE_LEVEL has cycle or disconnected issue. nodes={nodes}, order={order}")

    return order

def load_processing_services(json_path: str):
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"processing_service.json not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        lst = json.load(f)

    return {item["name"]: item for item in lst if "name" in item}

def call_api(api_name: str, body: dict, timeout=600):
    API_ENDPOINT = get_api_endpoint()
    endpoint = API_ENDPOINT.get(api_name)
    if not endpoint:
        raise KeyError(f"Missing endpoint mapping for api: {api_name}")
    url = FLASK_BASE + endpoint
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def add_to_pool(pool: dict, path_or_url: str):
    if not path_or_url:
        return
    if not isinstance(path_or_url, str):
        return
    if not (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        return

    if path_or_url not in pool["resources"]:
        pool["resources"].append(path_or_url)

def build_llm_prompt(api_name: str, api_meta: dict, resource_pool: dict):
    """
    让 LLM 从 resource_pool 里为 in_fields 选 URL，输出 body JSON
    """
    in_fields = api_meta.get("in_fields", [])
    input_descr = api_meta.get("input", "")
    descr = api_meta.get("descr", "")

    prompt = r"""
You are binding inputs for a REST API call in a geospatial pipeline.

API name: {api_name}
API description: {descr}
Semantic input description: {input_descr}

Required input fields (must fill all, ordered):
{in_fields}

Candidate resources (URLs only):
{resources}

Rules:
1) You MUST pick resources ONLY from the candidate resources list.
2) You MUST output a JSON object that can be used as request body.
3) Keys of the JSON must exactly match the required input fields.
4) If you cannot find a suitable resource for a field, set it to null.
5) Output STRICT JSON only. No extra text.

Output JSON example:
{{"field1":"http://.../a.tif","field2":"http://.../b.tif"}}
""".strip()

    request = prompt.format(
        api_name=api_name,
        descr=descr,
        input_descr=input_descr,
        in_fields=json.dumps(in_fields, ensure_ascii=False),
        resources=json.dumps(resource_pool.get("resources", []), ensure_ascii=False),
    )
    return request

def llm_bind_inputs(api_name: str, api_meta: dict, resource_pool: dict):
    request = build_llm_prompt(api_name, api_meta, resource_pool)
    answer = interact_with_llm(request, LLM_MODEL)

    try:
        if ("```json" in answer):
            body = answer.split("```json")[1].split("```")[0]
            body = json.loads(body)
        else:
            body = json.loads(answer)
    except Exception as e:
        raise RuntimeError(f"LLM did not return valid JSON for {api_name}.\nRaw:\n{answer}") from e

    # 完整性检查：必须包含所有 in_fields
    in_fields = api_meta.get("in_fields", [])
    for f in in_fields:
        if f not in body:
            raise RuntimeError(f"LLM output missing required field '{f}' for {api_name}. body={body}")

    # 缺参检查（你现在先严格一点：出现 null 直接报错）
    missing = [f for f in in_fields if body.get(f) in (None, "", [])]
    if missing:
        raise RuntimeError(f"Cannot bind inputs for {api_name}, missing: {missing}. LLM body={body}")

    return body

def make_timestamped_dir(base_dir: Path, name: str = "job") -> Path:
    from datetime import datetime
    """
    在 base_dir 下创建一个带时间戳的唯一子目录，并返回 Path。

    示例：
      make_timestamped_dir(Path("D:/work/tmp"), "job")
      -> D:/work/tmp/job_20251214_153012_123456
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    new_dir = base_dir / f"{name}_{ts}"
    new_dir.mkdir(parents=True, exist_ok=False)

    return new_dir

def execute_service_chain(edges_service_level, standard_answer):  ## standard_answer is used to match data service
    if isinstance(edges_service_level, list):
        API_ORDER = topo_order(edges_service_level)
    else:
        API_ORDER = [edges_service_level] ## 只用一个model service就能实现的情况 ["service"]
    svc_map = load_processing_services(PROCESSING_JSON)

    # 资源池：先简单存“所有已获得的路径/URL”（不去重之外不分类）
    resource_pool = {"resources": [],"processing_service_chain":API_ORDER,"data_flow":[],"edges_service_level":edges_service_level}

    for i, api in enumerate(API_ORDER):
        print(f"===========run service {api}=============")
        current_dataflow = {"input":[],"output":[]}
        if i == 0:
            meta = svc_map.get(api)
            query = json.loads(standard_answer)["Location"]+ " " +  meta["input"]
            INIT_DATA = match_service_by_embedding(
                query_text=query,
                emb_csv=DATA_EMB_CSV,
                api_key=EMB_API_KEY,
            )["service_name"]
            INIT_DATA = PUBLIC_BASE_URL+"/data_service/"+INIT_DATA
            # print(INIT_DATA)
            body = {meta["in_fields"][0]: INIT_DATA}

            resp = call_api(api, body)
            data = resp.get("data") or {}
            current_dataflow["input"].append({meta["in_fields"][0]: INIT_DATA})

            # 记录输出结果的“整个路径/URL”到资源池里：
            # 这里优先把返回里所有 *_path / *_tif / *_url 都收集起来（不做扩展名判断）
            for k, v in data.items():
                add_to_pool(resource_pool, v)
                if  (v.startswith("http://") or v.startswith("https://")):
                    current_dataflow["output"].append({"output_data": v})
            
            resource_pool["data_flow"].append(current_dataflow)
            if isinstance(data.get("job_dir"), str):
                add_to_pool(resource_pool, data["job_dir"])
        else:
            meta = svc_map.get(api)
            body = llm_bind_inputs(api, meta, resource_pool)
                
            current_dataflow["input"].extend(body)
            resp = call_api(api, body)
            data = resp.get("data") or {}

            for k, v in data.items():
                add_to_pool(resource_pool, v)
                if  (v.startswith("http://") or v.startswith("https://")):
                    current_dataflow["output"].append({"output_data": v})
            resource_pool["data_flow"].append(current_dataflow)

    # # 你需要的话可以在这里把 resource_pool 写盘保存
    
    final_dir = make_timestamped_dir("repository")
    with open(f"{final_dir}//resource_pool.json", "w", encoding="utf-8") as f:
        json.dump(resource_pool, f, ensure_ascii=False, indent=2)
    return resource_pool

def clear_temp_dir(temp_dir):
    """
    清空指定目录下的所有文件和子目录，但保留该目录本身。
    """
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return

    for item in temp_dir.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except Exception:
            # 如需更严格，可在这里 raise
            pass

def build_merged_edges(response: dict) -> list:
    """
    约束：
      - response["processing_service_chain"] 已经是拓扑序，格式: [[svc1, svc2, ...]]
      - data 节点必须用完整 URL
      - 对齐规则：services[i] <-> data_flow[i]
      - 返回合并后的有向边列表：
          (service, data_url)  and  (data_url, service)
    """
    def _basename(url: str) -> str:
        path = urlparse(url).path
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        return name

    def _ext(url: str) -> str:
        path = urlparse(url).path
        base = os.path.basename(path)
        _, ext = os.path.splitext(base)
        return ext.lstrip(".").lower()
    data_flow = response.get("data_flow", [])
    chain = response.get("processing_service_chain", [])
    services = chain if chain and isinstance(chain, list) else []
    n = min(len(services), len(data_flow))

    # 记录已产出数据，便于把 token（dem_filled_tif / accum_tif / raster_tif）解析成完整 URL
    produced_by_key = {}                 # (basename, ext) -> url
    produced_by_ext = defaultdict(list)  # ext -> [url...]

    def register_output_url(url: str):
        produced_by_key[(_basename(url), _ext(url))] = url
        produced_by_ext[_ext(url)].append(url)

    def resolve_input_token(token: str):
        # token: name_ext  (dem_filled_tif / accum_tif / streams_geojson / raster_tif)
        if "_" not in token:
            return None
        name, ext = token.rsplit("_", 1)
        ext = ext.lower()

        # 1) 精确匹配 name+ext
        url = produced_by_key.get((name, ext))
        if url:
            return url

        # 2) 退化：仅按 ext 取最近一次产出（raster_tif 常见）
        if produced_by_ext[ext]:
            return produced_by_ext[ext][-1]

        return None

    edges = []
    for i in range(n):
        svc = services[i]
        item = data_flow[i]

        # inputs: dict 的 value 是完整 URL；str token 解析成完整 URL
        for inp in item.get("input", []):
            if isinstance(inp, dict):
                for _, url in inp.items():
                    edges.append((url, svc))          # (data_url, service)
            elif isinstance(inp, str):
                url = resolve_input_token(inp)
                if url:
                    edges.append((url, svc))          # (data_url, service)

        # outputs: output_data 本身就是完整 URL
        for out in item.get("output", []):
            if isinstance(out, dict) and "output_data" in out:
                url = out["output_data"]
                register_output_url(url)
                edges.append((svc, url))              # (service, data_url)

    # 去重（保持顺序）
    seen = set()
    merged_edges = []
    for e in edges:
        if e not in seen:
            merged_edges.append(e)
            seen.add(e)

    return merged_edges
    
def generate_bpmn(service_chain_info):
    prompt = '''
        You are an expert in workflow modeling and BPMN 2.0 specification.
        You must generate syntactically correct BPMN 2.0 XML that can be directly imported
        into Camunda Modeler or Flowable Designer.

        You must strictly follow BPMN 2.0 semantics:
        - Use serviceTask for processing services
        - Use dataObject and dataObjectReference for data services
        - Use sequenceFlow for control flow
        - Use dataInputAssociation and dataOutputAssociation for data dependencies
        
        Given a service execution chain described as a list of directed edges:

        - Each edge is a pair [source, target]
        - Any node whose name starts with "http" represents a DATA SERVICE
        - Any other node represents a PROCESSING SERVICE

        The input format is:

        {{
        "service_chain_info": [
            [source1, target1],
            [source2, target2],
            ...
        ]
        }}

        Your task is to generate a complete BPMN 2.0 XML file that represents this service chain.

        You MUST follow these rules strictly:

        1. Node Mapping Rules
        - Each processing service MUST be represented as a <bpmn:serviceTask>
        - Each data service MUST be represented as:
            - one <bpmn:dataObject> with its URL stored in <bpmn:documentation>
            - one <bpmn:dataObjectReference> linked to that dataObject
        - Do NOT represent data services as tasks or events

        2. Control Flow Construction
        - If a processing service has only one upstream processing service, connect them directly using <bpmn:sequenceFlow>
        - If a processing service has multiple upstream processing services, insert a <bpmn:parallelGateway> to synchronize them
        - If a processing service has multiple downstream processing services, insert a <bpmn:parallelGateway> to split the flow
        - Ensure the process has exactly one <bpmn:startEvent> and one <bpmn:endEvent>

        3. Data Dependency Mapping
        - For each edge [data_service → processing_service], create a <bpmn:dataInputAssociation>
        - For each edge [processing_service → data_service], create a <bpmn:dataOutputAssociation>
        - A processing service may have multiple data inputs and outputs

        4. Graph Semantics
        - The BPMN must reflect the partial order implied by the service_chain_info
        - Parallel branches MUST be explicit using parallel gateways
        - Do NOT introduce artificial ordering not implied by the dependency graph

        5. Naming and Validity Constraints
        - All BPMN element IDs MUST be unique and XML-safe
        - Use readable names for tasks (use the original service name)
        - The output MUST be valid BPMN 2.0 XML
        - Do NOT include explanations, comments, or markdown
        - Output ONLY the BPMN XML

        Now generate the BPMN 2.0 XML for the following input:

        {}
   
    '''
    result=interact_with_llm(prompt.format(service_chain_info))
    if ("```xml" in result):
        result = result.split("```xml")[1].split("```")[0] 
    if not result or not result.strip():
        raise RuntimeError("LLM returned empty BPMN result")

    # ===== 生成文件名 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_bpmn.xml"
    save_path = os.path.join(r"...\repository", filename)  ## replace the absolote path to the repository dictionary in this project

    bpmn_url = PUBLIC_BASE_URL+"/"+filename
    # ===== 保存 BPMN 文件 =====
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(result.strip())
    
    # ===== 返回保存路径（或你可以改成 URL） =====
    return bpmn_url
    
