"""The full suite runner: every runnable Oracle test from the official
tutorials and workshop, run with this project's content (see the README's
"The complete test suite"). Expects the TECHNIQUES table imported from
data/techniques.dmp with both vector columns populated (run_database_test.py
does that part), the MiniLM model at F:\\OracleAI\\models, the e5 vectors at
C:\\Temp\\e5_vectors.json, the schema password in C:\\Temp\\vtest_pw.txt, and
vector_memory_size set (256M) for the HNSW index.
"""
import array
import json

import numpy as np
import onnxruntime as ort
import oracledb
from onnxruntime_extensions import get_library_path

PW = open(r"C:\Temp\vtest_pw.txt").read().strip()
Q1 = "how do I avoid bias in my analysis"

conn = oracledb.connect(user="vtest", password=PW, dsn="localhost:1521/FREEPDB1")
cur = conn.cursor()


def show(title):
    print(f"\n===== {title} =====")


def run(sql, **binds):
    return cur.execute(sql, **binds).fetchall()


def first_error(fn):
    try:
        fn()
        return "OK (no error)"
    except oracledb.Error as e:
        return str(e).splitlines()[0]


# ---------- T2: external table over the .txt (getting-started pattern) ----------
show("T2 external table over techniques.txt")
try:
    cur.execute("DROP TABLE tech_ext PURGE")
except oracledb.Error:
    pass
cur.execute(
    """CREATE TABLE tech_ext (sentence VARCHAR2(4000))
       ORGANIZATION EXTERNAL (TYPE ORACLE_LOADER DEFAULT DIRECTORY DATA_PUMP_DIR
         ACCESS PARAMETERS (RECORDS DELIMITED BY 0x'0A' READSIZE 1000000
                            FIELDS (sentence CHAR(4000)))
         LOCATION (DATA_PUMP_DIR:'techniques.txt'))
       REJECT LIMIT UNLIMITED"""
)
n = run("SELECT COUNT(*) FROM tech_ext")[0][0]
print(f"external table rows: {n} (expected 6)")
for (s,) in run("SELECT sentence FROM tech_ext WHERE ROWNUM <= 2"):
    print("  ", s[:70], "...")

# ---------- query vectors (MiniLM on this PC, e5 from JSON) ----------
opts = ort.SessionOptions()
opts.register_custom_ops_library(get_library_path())
sess = ort.InferenceSession(
    r"F:\OracleAI\models\all_MiniLM_L12_v2.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)


def embed(batch):
    (out,) = sess.run(None, {"input": np.array(batch, dtype=object)})
    out = np.asarray(out, dtype=np.float64)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


q_minilm = array.array("f", [float(x) for x in embed([Q1])[0]])
e5 = json.load(open(r"C:\Temp\e5_vectors.json"))
q_e5 = array.array("f", e5["e5_query_vectors"]["q1"])

# ---------- T4: FETCH APPROX (getting-started + workshop pattern) ----------
show("T4 FETCH APPROX vs EXACT (q1)")
for model, col, qv in [("minilm", "minilm_vector", q_minilm), ("e5", "e5_vector", q_e5)]:
    for mode in ("EXACT", "APPROX"):
        rows = run(
            f"SELECT title, VECTOR_DISTANCE({col}, :qv, COSINE) FROM techniques "
            f"ORDER BY 2 FETCH {mode} FIRST 3 ROWS ONLY",
            qv=qv,
        )
        tops = " | ".join(f"{t} {1.0 - float(d):.4f}" for t, d in rows)
        print(f"  {model} {mode}: {tops}")

# ---------- T5: other distance metrics (their ORDS service used EUCLIDEAN) ----------
show("T5 distance metrics on q1 (minilm)")
for metric in ("COSINE", "EUCLIDEAN", "EUCLIDEAN_SQUARED", "DOT", "MANHATTAN"):
    rows = run(
        f"SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(minilm_vector, :qv, {metric}) "
        "FETCH EXACT FIRST 3 ROWS ONLY",
        qv=q_minilm,
    )
    print(f"  {metric}: " + " | ".join(t for (t,) in rows))

# ---------- T3: in-database VECTOR_EMBEDDING (formal attempt) ----------
show("T3 VECTOR_EMBEDDING with the in-database model (formal attempt)")
print(
    "  result:",
    first_error(
        lambda: run(
            "SELECT VECTOR_EMBEDDING(MINILM_L12_V2 USING 'test' AS data) FROM dual"
        )
    ),
)

# ---------- T8: HYBRID VECTOR INDEX (formal attempt) + SEARCH ----------
show("T8 CREATE HYBRID VECTOR INDEX (formal attempt)")
print(
    "  result:",
    first_error(
        lambda: cur.execute(
            "CREATE HYBRID VECTOR INDEX tech_hybrid_idx ON techniques(text) "
            "PARAMETERS('MODEL MINILM_L12_V2')"
        )
    ),
)

# ---------- T7: IVF vector index ----------
show("T7 IVF vector index")
try:
    cur.execute("DROP INDEX tech_ivf_idx")
except oracledb.Error:
    pass
print(
    "  create:",
    first_error(
        lambda: cur.execute(
            "CREATE VECTOR INDEX tech_ivf_idx ON techniques(minilm_vector) "
            "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 90"
        )
    ),
)
rows = run(
    "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(minilm_vector, :qv, COSINE) "
    "FETCH APPROX FIRST 3 ROWS ONLY",
    qv=q_minilm,
)
print("  approx q1 with IVF present:", " | ".join(t for (t,) in rows))

# ---------- T6 retry: HNSW with the enlarged vector pool ----------
show("T6 HNSW vector index on e5_vector (after vector_memory_size = 256M)")
try:
    cur.execute("DROP INDEX tech_hnsw_idx")
except oracledb.Error:
    pass
print(
    "  create:",
    full_error(
        lambda: cur.execute(
            "CREATE VECTOR INDEX tech_hnsw_idx ON techniques(e5_vector) "
            "ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95"
        )
    ),
)
rows = run(
    "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(e5_vector, :qv, COSINE) "
    "FETCH APPROX FIRST 3 ROWS ONLY",
    qv=q_e5,
)
print("  approx q1 on e5:", " | ".join(t for (t,) in rows))

# ---------- dictionary: V$VECTOR_INDEX + the pool ----------
show("V$VECTOR_INDEX and V$VECTOR_MEMORY_POOL")
try:
    cur2 = conn.cursor()
    cur2.execute("SELECT * FROM v$vector_index WHERE index_name LIKE 'TECH%'")
    cols = [d[0] for d in cur2.description]
    for r in cur2.fetchall():
        keep = {
            c: v
            for c, v in zip(cols, r)
            if c
            in (
                "INDEX_NAME",
                "INDEX_ORGANIZATION",
                "INDEX_TYPE",
                "NUM_VECTORS",
                "DISTANCE_METRIC",
                "TARGET_ACCURACY",
            )
        }
        print("  ", keep or dict(zip(cols, r)))
except oracledb.Error as e:
    print("  v$vector_index error:", str(e).splitlines()[0])
try:
    rows = run("SELECT * FROM v$vector_memory_pool")
    print("  pool rows:", len(rows))
except oracledb.Error as e:
    print("  pool error:", str(e).splitlines()[0])

# ---------- plans: plain approx vs hinted index scan ----------
show("Execution plans, approx q1 on e5")
def plan_for(sql_text, tag):
    cur.execute(sql_text, qv=q_e5).fetchall()
    rows = run(
        "SELECT sql_id FROM v$sql WHERE sql_text LIKE :t AND ROWNUM = 1",
        t=sql_text[:60] + "%",
    )
    if not rows:
        print(f"  {tag}: sql_id not found")
        return
    plan = run(
        "SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(:sid, NULL, 'BASIC'))",
        sid=rows[0][0],
    )
    ops = [
        line
        for (line,) in plan
        if any(k in line for k in ("VECTOR", "INDEX", "TABLE ACCESS", "SORT"))
    ]
    print(f"  {tag}:")
    for line in ops:
        print("   ", line)

plan_for(
    "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(e5_vector, :qv, COSINE) "
    "FETCH APPROX FIRST 3 ROWS ONLY",
    "plain approx",
)
plan_for(
    "SELECT /*+ VECTOR_INDEX_SCAN(techniques tech_hnsw_idx) */ title FROM techniques "
    "ORDER BY VECTOR_DISTANCE(e5_vector, :qv, COSINE) FETCH APPROX FIRST 3 ROWS ONLY",
    "hinted VECTOR_INDEX_SCAN",
)

# ---------- dictionary: V$VECTOR_INDEX + the pool ----------
show("V$VECTOR_INDEX and V$VECTOR_MEMORY_POOL")
try:
    cur2 = conn.cursor()
    cur2.execute("SELECT * FROM v$vector_index WHERE index_name LIKE 'TECH%'")
    cols = [d[0] for d in cur2.description]
    for r in cur2.fetchall():
        keep = {
            c: v
            for c, v in zip(cols, r)
            if c
            in (
                "INDEX_NAME",
                "INDEX_ORGANIZATION",
                "INDEX_TYPE",
                "NUM_VECTORS",
                "DISTANCE_METRIC",
                "TARGET_ACCURACY",
            )
        }
        print("  ", keep or dict(zip(cols, r)))
except oracledb.Error as e:
    print("  v$vector_index error:", str(e).splitlines()[0])
try:
    rows = run("SELECT * FROM v$vector_memory_pool")
    print("  pool rows:", len(rows))
except oracledb.Error as e:
    print("  pool error:", str(e).splitlines()[0])

# ---------- plans: plain approx vs hinted index scan ----------
show("Execution plans, approx q1 on e5")
def plan_for(sql_text, tag):
    cur.execute(sql_text, qv=q_e5).fetchall()
    rows = run(
        "SELECT sql_id FROM v$sql WHERE sql_text LIKE :t AND ROWNUM = 1",
        t=sql_text[:60] + "%",
    )
    if not rows:
        print(f"  {tag}: sql_id not found")
        return
    plan = run(
        "SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(:sid, NULL, 'BASIC'))",
        sid=rows[0][0],
    )
    ops = [
        line
        for (line,) in plan
        if any(k in line for k in ("VECTOR", "INDEX", "TABLE ACCESS", "SORT"))
    ]
    print(f"  {tag}:")
    for line in ops:
        print("   ", line)

plan_for(
    "SELECT title FROM techniques ORDER BY VECTOR_DISTANCE(e5_vector, :qv, COSINE) "
    "FETCH APPROX FIRST 3 ROWS ONLY",
    "plain approx",
)
plan_for(
    "SELECT /*+ VECTOR_INDEX_SCAN(techniques tech_hnsw_idx) */ title FROM techniques "
    "ORDER BY VECTOR_DISTANCE(e5_vector, :qv, COSINE) FETCH APPROX FIRST 3 ROWS ONLY",
    "hinted VECTOR_INDEX_SCAN",
)

# ---------- T11: chunk count via CARDINALITY ----------
show("T11 chunk count (CARDINALITY)")
def chunk_count():
    rows = run(
        "SELECT CARDINALITY(DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(text, "
        "JSON('{\"by\":\"words\",\"max\":10}'))) FROM techniques WHERE id = 'core-06'"
    )
    print("  chunks for core-06 (max 10 words each):", rows[0][0])
print("  result:", full_error(chunk_count))


# ---------- T9: ORDS presence ----------
show("T9 ORDS presence")
rows = run("SELECT username FROM dba_users WHERE username LIKE 'ORDS%'")
print("  ORDS users:", [r[0] for r in rows] or "none")
rows = run(
    "SELECT COUNT(*) FROM dba_objects WHERE object_name = 'ORDS_ADMIN' AND object_type = 'PACKAGE'"
)
print("  ORDS_ADMIN package count:", rows[0][0])

conn.close()
print("\nPHASE 1 DONE")
