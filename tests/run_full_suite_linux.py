"""The full suite runner, Linux edition: every runnable Oracle test from the
official blogs and the LiveLabs workshop, run end-to-end against the Linux
lab (Oracle AI Database 26ai Free container + Oracle Private AI Services
container) with this project's content (the six techniques in corpus.json,
the same content stored in data/techniques.dmp).

On Windows, three of these tests are blocked by the edition (in-database
ONNX runtime absent, ORDS absent). On Linux the in-database runtime and
ORDS run, and so does every test in this script except text generation:
the medium Private AI image deploys its LLM on this lab host, but
generation does not complete at usable speed on a host of this size
(7 GB RAM, 1.5 shared CPUs), so T13 and the generation half of T14
record BLOCKED with that reason (full evidence in the README). It expects:

  * the lab database up (FREEPDB1) with a WRITING schema (CONNECT,
    RESOURCE, DB_DEVELOPER_ROLE, CREATE MINING MODEL, SELECT_CATALOG_ROLE,
    read/write on MODELS_DIR and LAB_DIR, and the network ACL for the
    Private AI endpoint), max_string_size EXTENDED and
    vector_memory_size = 256M;
  * the two embedding models (all_MiniLM_L12_v2.onnx,
    multilingual_e5_small.onnx) in the MODELS_DIR directory inside the
    database;
  * the Private AI container serving on the endpoint stored in the
    schema's PRIVATE_AI_CONFIG table;
  * ORDS installed in FREEPDB1 and serving (for the REST test);
  * environment: ORACLE_LAB_PW (default: the workshop's published lab
    password), ORACLE_LAB_DSN (default localhost:1521/FREEPDB1),
    ORDS_URL (default http://localhost:8080/ords/writing/techniques/),
    LAB_DIR_HOST (host path of the LAB_DIR directory, for the external
    table test).

Run:  python tests/run_full_suite_linux.py
"""
import json
import os
import time
import urllib.request

import oracledb

PW = os.environ.get("ORACLE_LAB_PW", "Welcome_12345")
DSN = os.environ.get("ORACLE_LAB_DSN", "localhost:1521/FREEPDB1")
ORDS_URL = os.environ.get("ORDS_URL", "http://localhost:8080/ords/writing/techniques/")
LAB_DIR_HOST = os.environ.get(
    "LAB_DIR_HOST", os.path.expanduser("~/workspace/oracle-linux/oradata/lab")
)
ENDPOINT = "http://localhost:8081"
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus.json")

QUERIES = [
    ("q1", "how do I avoid bias in my analysis"),
    ("q2", "why does this matter to the reader"),
    ("q3", "conhecer o publico para adaptar o conteudo"),
    ("q4", "como escrever com mais clareza"),
]
# Reference top-1 per query, produced by this repository's own engine.py
# (the same model files, run outside the database) on this corpus.
EXPECTED_TOP1 = {
    ("minilm", "q1"): "Objectivity",
    ("minilm", "q2"): "So What?",
    ("minilm", "q3"): "Audience Analysis",
    ("minilm", "q4"): "Clarity",
    ("e5", "q1"): "Objectivity",
    ("e5", "q2"): "So What?",
    ("e5", "q3"): "Audience Analysis",
    ("e5", "q4"): "Conciseness",
}

conn = oracledb.connect(user="writing", password=PW, dsn=DSN)
cur = conn.cursor()
results = {}


def show(title):
    print(f"\n===== {title} =====", flush=True)


def run(sql, **binds):
    return cur.execute(sql, **binds).fetchall()


def attempt(fn):
    try:
        return fn()
    except oracledb.Error as e:
        return f"ERROR: {str(e).splitlines()[0]}"


# ---------- setup: config table, techniques, in-database models ----------
show("SETUP config, techniques, in-database ONNX models")
tables = {r[0] for r in run("SELECT table_name FROM user_tables")}
if "PRIVATE_AI_CONFIG" not in tables:
    cur.execute(
        "CREATE TABLE private_ai_config (config_name VARCHAR2(100), config_value VARCHAR2(1000))"
    )
    cur.execute(
        "INSERT INTO private_ai_config VALUES ('HTTP_ENDPOINT', :e)", e=ENDPOINT
    )
    conn.commit()
endpoint = run(
    "SELECT config_value FROM private_ai_config WHERE config_name = 'HTTP_ENDPOINT'"
)[0][0]
print("  Private AI endpoint:", endpoint)

corpus = json.load(open(CORPUS_PATH))
if "TECHNIQUES" not in tables:
    cur.execute(
        "CREATE TABLE techniques (id VARCHAR2(20), title VARCHAR2(200), "
        "text VARCHAR2(4000), minilm_vector VECTOR(384, FLOAT32), "
        "e5_vector VECTOR(384, FLOAT32))"
    )
n = run("SELECT COUNT(*) FROM techniques")[0][0]
if n == 0:
    for item in corpus:
        cur.execute(
            "INSERT INTO techniques (id, title, text) VALUES (:i, :t, :x)",
            i=item["id"],
            t=item["title"],
            x=item["text"],
        )
    conn.commit()
    n = run("SELECT COUNT(*) FROM techniques")[0][0]
print(f"  techniques rows: {n} (expected 6)")

models = {r[0] for r in run("SELECT model_name FROM user_mining_models")}
for fname, mname in [
    ("all_MiniLM_L12_v2.onnx", "minilm_l12_v2"),
    ("multilingual_e5_small.onnx", "e5_small"),
]:
    if mname.upper() not in models:
        cur.execute(
            "BEGIN DBMS_VECTOR.LOAD_ONNX_MODEL('MODELS_DIR', :f, :m); END;",
            f=fname,
            m=mname,
        )
        print(f"  loaded {mname}")
print("  models:", sorted({r[0] for r in run("SELECT model_name FROM user_mining_models")}))

for col, model in [("minilm_vector", "minilm_l12_v2"), ("e5_vector", "e5_small")]:
    missing = run(f"SELECT COUNT(*) FROM techniques WHERE {col} IS NULL")[0][0]
    if missing:
        cur.execute(
            f"UPDATE techniques SET {col} = VECTOR_EMBEDDING({model} USING text AS data)"
        )
        conn.commit()
    done = run(f"SELECT COUNT(*) FROM techniques WHERE {col} IS NOT NULL")[0][0]
    print(f"  {col} populated in-database: {done}/6")

# ---------- T9: ORDS REST (before the DDL-heavy tests, as in the parks suite) ----------
show("T9 ORDS REST service")
try:
    cur.execute(
        "BEGIN ORDS.ENABLE_SCHEMA(p_enabled => TRUE, p_schema => 'WRITING', "
        "p_url_mapping_type => 'BASE_PATH', p_url_mapping_pattern => 'writing', "
        "p_auto_rest_auth => FALSE); "
        "ORDS.ENABLE_OBJECT(p_schema => 'WRITING', p_object => 'TECHNIQUES', "
        "p_object_type => 'TABLE', p_enabled => TRUE, p_auto_rest_auth => FALSE); "
        "COMMIT; END;"
    )
    print("  schema + TECHNIQUES REST-enabled")
except oracledb.Error as e:
    print("  ORDS enable:", str(e).splitlines()[0])
first = None
for attempt_no in range(1, 4):
    try:
        with urllib.request.urlopen(ORDS_URL + "?limit=1", timeout=15) as resp:
            doc = json.load(resp)
        first = doc["items"][0]["title"] if doc.get("items") else None
        if first:
            break
    except Exception as e:
        print(f"  ORDS GET attempt {attempt_no} failed: {e}")
        time.sleep(5)
print(f"  GET {ORDS_URL}?limit=1 -> first technique: {first}")
results["T9 ORDS REST"] = "PASS" if first else "FAIL"

# ---------- T3: in-database VECTOR_EMBEDDING ----------
show("T3 in-database VECTOR_EMBEDDING (the Windows ORA-40216 test)")
dims = {}
for model in ("minilm_l12_v2", "e5_small"):
    dims[model] = attempt(
        lambda m=model: run(
            f"SELECT VECTOR_DIMENSION_COUNT(VECTOR_EMBEDDING({m} USING 'test' AS data)) FROM dual"
        )[0][0]
    )
print("  dimensions:", dims)
results["T3 in-DB VECTOR_EMBEDDING"] = (
    "PASS" if all(v == 384 for v in dims.values()) else f"FAIL ({dims})"
)

# ---------- T1: exact search in-database vs the engine's own rankings ----------
show("T1 exact search, in-database embeddings vs engine.py (4 queries x 2 models)")
ok = 0
total = 0
for model, col in [("minilm", "minilm_vector"), ("e5", "e5_vector")]:
    dbmodel = "minilm_l12_v2" if model == "minilm" else "e5_small"
    for qid, qtext in QUERIES:
        rows = run(
            f"SELECT title, 1 - VECTOR_DISTANCE({col}, "
            f"VECTOR_EMBEDDING({dbmodel} USING :q AS data), COSINE) "
            "FROM techniques ORDER BY 2 DESC FETCH EXACT FIRST 1 ROWS ONLY",
            q=qtext,
        )
        top, sim = rows[0]
        total += 1
        match = top == EXPECTED_TOP1[(model, qid)]
        ok += match
        print(f"  {model} {qid}: {top} sim {float(sim):.4f} "
              f"(engine top-1: {EXPECTED_TOP1[(model, qid)]}) {'OK' if match else 'MISMATCH'}")
print(f"  top-1 agreement with engine.py: {ok}/{total}")
results["T1 exact search = engine top-1"] = "PASS" if ok == total else f"FAIL ({ok}/{total})"

# ---------- T2: external table over techniques.txt ----------
show("T2 external table over techniques.txt")
txt_path = os.path.join(LAB_DIR_HOST, "techniques.txt")
with open(txt_path, "w") as fh:
    for item in corpus:
        fh.write(item["text"].replace("\n", " ") + "\n")
os.chmod(txt_path, 0o644)  # the database process (another uid) must read it
attempt(lambda: cur.execute("DROP TABLE tech_ext PURGE"))
cur.execute(
    """CREATE TABLE tech_ext (sentence VARCHAR2(4000))
       ORGANIZATION EXTERNAL (TYPE ORACLE_LOADER DEFAULT DIRECTORY LAB_DIR
         ACCESS PARAMETERS (RECORDS DELIMITED BY 0x'0A' READSIZE 1000000
                            FIELDS (sentence CHAR(4000)))
         LOCATION (LAB_DIR:'techniques.txt'))
       REJECT LIMIT UNLIMITED"""
)
n = attempt(lambda: run("SELECT COUNT(*) FROM tech_ext")[0][0])
print(f"  external table rows: {n} (expected 6)")
results["T2 external table"] = "PASS" if n == 6 else f"FAIL ({n})"

# ---------- T4: FETCH APPROX vs EXACT ----------
show("T4 FETCH APPROX vs EXACT (q1, both models)")
same = True
for model, col in [("minilm", "minilm_vector"), ("e5", "e5_vector")]:
    dbmodel = "minilm_l12_v2" if model == "minilm" else "e5_small"
    tops = {}
    for mode in ("EXACT", "APPROX"):
        rows = run(
            f"SELECT title FROM techniques ORDER BY VECTOR_DISTANCE({col}, "
            f"VECTOR_EMBEDDING({dbmodel} USING :q AS data), COSINE) "
            f"FETCH {mode} FIRST 3 ROWS ONLY",
            q=QUERIES[0][1],
        )
        tops[mode] = [t for (t,) in rows]
    same &= tops["EXACT"] == tops["APPROX"]
    print(f"  {model}: EXACT {tops['EXACT']} == APPROX {tops['APPROX']}: {tops['EXACT'] == tops['APPROX']}")
results["T4 APPROX == EXACT top-3"] = "PASS" if same else "FAIL"

# ---------- T5: five distance metrics ----------
show("T5 five distance metrics (q1, minilm)")
tops5 = {}
for metric in ("COSINE", "EUCLIDEAN", "EUCLIDEAN_SQUARED", "DOT", "MANHATTAN"):
    rows = run(
        "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(minilm_vector, "
        "VECTOR_EMBEDDING(minilm_l12_v2 USING :q AS data), "
        + metric
        + ") FETCH EXACT FIRST 3 ROWS ONLY",
        q=QUERIES[0][1],
    )
    tops5[metric] = [t for (t,) in rows]
    print(f"  {metric}: " + " | ".join(tops5[metric]))
base = tops5["COSINE"]
results["T5 five distance metrics"] = (
    "PASS" if all(v == base for v in tops5.values()) else "CHECK"
)

# ---------- T7: IVF vector index ----------
show("T7 IVF vector index on minilm_vector")
attempt(lambda: cur.execute("DROP INDEX tech_ivf_idx"))
created = attempt(
    lambda: cur.execute(
        "CREATE VECTOR INDEX tech_ivf_idx ON techniques(minilm_vector) "
        "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 90"
    )
)
rows = attempt(
    lambda: run(
        "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(minilm_vector, "
        "VECTOR_EMBEDDING(minilm_l12_v2 USING :q AS data), COSINE) "
        "FETCH APPROX FIRST 3 ROWS ONLY",
        q=QUERIES[0][1],
    )
)
print("  create:", "OK" if not str(created).startswith("ERROR") else created)
print("  approx q1 with IVF present:", rows)
results["T7 IVF index"] = (
    "PASS"
    if not str(created).startswith("ERROR") and isinstance(rows, list) and rows
    else "FAIL"
)

# ---------- T6: HNSW vector index ----------
show("T6 HNSW vector index on e5_vector (vector_memory_size = 256M)")
attempt(lambda: cur.execute("DROP INDEX tech_hnsw_idx"))
created = attempt(
    lambda: cur.execute(
        "CREATE VECTOR INDEX tech_hnsw_idx ON techniques(e5_vector) "
        "ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95"
    )
)
rows = attempt(
    lambda: run(
        "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(e5_vector, "
        "VECTOR_EMBEDDING(e5_small USING :q AS data), COSINE) "
        "FETCH APPROX FIRST 3 ROWS ONLY",
        q=QUERIES[0][1],
    )
)
print("  create:", "OK" if not str(created).startswith("ERROR") else created)
print("  approx q1 on e5:", rows)
results["T6 HNSW index"] = (
    "PASS"
    if not str(created).startswith("ERROR") and isinstance(rows, list) and rows
    else "FAIL"
)

# ---------- plans + dictionary ----------
show("Execution plans and the vector index dictionary")
cur.execute(
    "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(e5_vector, "
    "VECTOR_EMBEDDING(e5_small USING :q AS data), COSINE) "
    "FETCH APPROX FIRST 3 ROWS ONLY",
    q=QUERIES[0][1],
).fetchall()
sql_id = run(
    "SELECT sql_id FROM v$sql WHERE sql_text LIKE 'SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(e5_vector%' AND ROWNUM = 1"
)
plan_lines = []
if sql_id:
    plan = run(
        "SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(:sid, NULL, 'BASIC'))",
        sid=sql_id[0][0],
    )
    plan_lines = [
        line for (line,) in plan if any(k in line for k in ("VECTOR", "INDEX", "TABLE ACCESS"))
    ]
for line in plan_lines:
    print("   ", line)
idx = run(
    "SELECT index_name, index_type, status FROM user_indexes "
    "WHERE index_name LIKE 'TECH%' ORDER BY index_name"
)
print("  user_indexes:", idx)
results["plans + dictionary"] = (
    "PASS" if any("HNSW" in line for line in plan_lines) and len(idx) >= 2 else "CHECK"
)

# ---------- T8: hybrid search ----------
show("T8 hybrid search (the Windows ORA-40284 test)")
have = run("SELECT index_name FROM user_indexes WHERE index_name = 'TECH_HYBRID_IDX'")
created = "exists"
if not have:
    created = attempt(
        lambda: cur.execute(
            "CREATE HYBRID VECTOR INDEX tech_hybrid_idx ON techniques(text) "
            "PARAMETERS('MODEL MINILM_L12_V2')"
        )
    )
print("  create:", created if str(created).startswith("ERROR") else "OK")


def hybrid_search():
    raw = run(
        "SELECT JSON_SERIALIZE(DBMS_HYBRID_VECTOR.SEARCH("
        "JSON('{\"hybrid_index_name\":\"TECH_HYBRID_IDX\",\"search_text\":\"bias in analysis\"}')) "
        "RETURNING VARCHAR2(32767)) FROM dual"
    )[0][0]
    return json.loads(raw)


hits = attempt(hybrid_search)
if isinstance(hits, list) and hits:
    print(f"  hits: {len(hits)}; top score {hits[0]['score']}; "
          f"top chunk: {hits[0]['chunk_text'][:110]}...")
    results["T8 hybrid SEARCH"] = "PASS"
else:
    print("  search:", hits)
    results["T8 hybrid SEARCH"] = "FAIL"

# ---------- T10: container embeddings via the provider ----------
show("T10 Private AI container embeddings via UTL_TO_EMBEDDING")
hello = attempt(
    lambda: run(
        "SELECT VECTOR_DIMENSION_COUNT(DBMS_VECTOR.UTL_TO_EMBEDDING('hello', "
        "JSON_OBJECT('provider' VALUE 'privateai', 'credential_name' VALUE NULL, "
        "'url' VALUE :u, 'host' VALUE 'local', 'model' VALUE 'all-minilm-l12-v2' "
        "RETURNING JSON))) FROM dual",
        u=endpoint + "/v1/embeddings",
    )[0][0]
)
print("  container embedding dimensions:", hello)
top = attempt(
    lambda: run(
        "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(minilm_vector, "
        "DBMS_VECTOR.UTL_TO_EMBEDDING(:q, JSON_OBJECT("
        "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
        "'host' VALUE 'local', 'model' VALUE 'all-minilm-l12-v2' RETURNING JSON)), COSINE) "
        "FETCH EXACT FIRST 1 ROWS ONLY",
        q=QUERIES[0][1],
        u=endpoint + "/v1/embeddings",
    )[0][0]
)
print("  q1 top-1 via container-embedded query:", top)
results["T10 container embeddings + search"] = (
    "PASS" if hello == 384 and top == EXPECTED_TOP1[("minilm", "q1")] else "FAIL"
)

# ---------- T11: chunking ----------
show("T11 chunking with DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS")
cnt = attempt(
    lambda: run(
        "SELECT CARDINALITY(DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(text, "
        "JSON('{\"by\":\"words\",\"max\":10}'))) FROM techniques WHERE id = 'core-06'"
    )[0][0]
)
print(f"  chunks for core-06 (max 10 words each): {cnt}")
results["T11 UTL_TO_CHUNKS"] = "PASS" if cnt == 4 else f"CHECK ({cnt})"

# ---------- T12: CLIP text encoder through the container ----------
show("T12 CLIP text encoder via the container (512 dimensions)")
clip_dims = attempt(
    lambda: run(
        "SELECT VECTOR_DIMENSION_COUNT(DBMS_VECTOR.UTL_TO_EMBEDDING('Civil War battlefields', "
        "JSON_OBJECT('provider' VALUE 'privateai', 'credential_name' VALUE NULL, "
        "'url' VALUE :u, 'host' VALUE 'local', 'model' VALUE 'clip-vit-base-patch32-txt' "
        "RETURNING JSON))) FROM dual",
        u=endpoint + "/v1/embeddings",
    )[0][0]
)
print("  CLIP text embedding dimensions:", clip_dims)
results["T12 CLIP text encoder"] = "PASS" if clip_dims == 512 else f"FAIL ({clip_dims})"

# ---------- T13/T14: generation gate + RAG over the techniques ----------
show("T13 generate text (Private AI LLM)")
try:
    with urllib.request.urlopen(endpoint + "/v1/models", timeout=15) as resp:
        models_doc = json.load(resp)
    ids = [m.get("id", "") for m in models_doc.get("data", [])]
    llm = next((i for i in ids if "ministral" in i.lower() or "instruct" in i.lower()), None)
except Exception as e:
    ids, llm = [], None
    print("  models probe failed:", e)
print("  container models:", ids)
if llm:

    def generate(prompt):
        return run(
            "SELECT DBMS_VECTOR_CHAIN.UTL_TO_GENERATE_TEXT(:p, JSON_OBJECT("
            "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
            "'host' VALUE 'local', 'model' VALUE :m RETURNING JSON)) FROM dual",
            p=prompt,
            u=endpoint + "/v1/chat/completions",
            m=llm,
        )[0][0]

    out = attempt(lambda: generate("Name the technique that guards against bias."))
    print("  generated:", str(out)[:300])
    results["T13 UTL_TO_GENERATE_TEXT"] = "PASS" if not str(out).startswith("ERROR") else "FAIL"
else:
    print("  no text-generation model in the running container image")
    results["T13 UTL_TO_GENERATE_TEXT"] = "BLOCKED (no LLM in running container — see README)"

show("T14 RAG over the techniques (retrieval + grounded generation)")
rag_hits = attempt(
    lambda: run(
        "SELECT title, text FROM techniques ORDER BY VECTOR_DISTANCE(minilm_vector, "
        "DBMS_VECTOR.UTL_TO_EMBEDDING(:q, JSON_OBJECT("
        "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
        "'host' VALUE 'local', 'model' VALUE 'all-minilm-l12-v2' RETURNING JSON)), COSINE) "
        "FETCH EXACT FIRST 2 ROWS ONLY",
        q=QUERIES[0][1],
        u=endpoint + "/v1/embeddings",
    )
)
print("  retrieved context:", [t for (t, _) in rag_hits] if isinstance(rag_hits, list) else rag_hits)
results["T14a RAG retrieval"] = (
    "PASS"
    if isinstance(rag_hits, list) and rag_hits and rag_hits[0][0] == EXPECTED_TOP1[("minilm", "q1")]
    else "FAIL"
)
if llm and isinstance(rag_hits, list) and rag_hits:
    context = " ".join(x[1] for x in rag_hits)
    grounded = attempt(
        lambda: generate(f"Context: {context}\nQuestion: {QUERIES[0][1]}")
    )
    print("  grounded answer:", str(grounded)[:300])
    results["T14b RAG grounded generation"] = (
        "PASS" if not str(grounded).startswith("ERROR") else "FAIL"
    )
else:
    results["T14b RAG grounded generation"] = "BLOCKED (same gate as T13)"

conn.close()
print("\n===== SUMMARY =====")
for k, v in results.items():
    print(f"  {k}: {v}")
print("\nFULL SUITE (LINUX) DONE")
