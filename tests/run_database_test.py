"""The database test runner: embeds the TECHNIQUES table (imported from
data/techniques.dmp) with both models and answers the four test queries
inside the database with VECTOR_DISTANCE. See database-test.md.
Expects: Python 3.12 with oracledb, onnxruntime, onnxruntime-extensions;
the MiniLM model at F:\OracleAI\models; e5 vectors at C:\Temp\e5_vectors.json;
the schema password in C:\Temp\vtest_pw.txt (never printed).
"""
import array
import json

import numpy as np
import onnxruntime as ort
import oracledb
from onnxruntime_extensions import get_library_path

PW = open(r"C:\Temp\vtest_pw.txt").read().strip()
QUERIES = [
    ("q1", "how do I avoid bias in my analysis"),
    ("q2", "why does this matter to the reader"),
    ("q3", "conhecer o publico para adaptar o conteudo"),
    ("q4", "como escrever com mais clareza"),
]

conn = oracledb.connect(user="vtest", password=PW, dsn="localhost:1521/FREEPDB1")
cur = conn.cursor()

for col in ("minilm_vector", "e5_vector"):
    try:
        cur.execute(f"ALTER TABLE techniques ADD ({col} VECTOR(384, FLOAT32))")
        print(f"column {col} added")
    except oracledb.Error as e:
        print(f"column {col}: {e}".splitlines()[0])

rows = cur.execute("SELECT id, text FROM techniques ORDER BY id").fetchall()
print(f"rows in TECHNIQUES (from the .dmp): {len(rows)}")
ids = [r[0] for r in rows]
texts = [r[1] for r in rows]


def embed(sess, batch):
    (out,) = sess.run(None, {"input": np.array(batch, dtype=object)})
    out = np.asarray(out, dtype=np.float64)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


opts = ort.SessionOptions()
opts.register_custom_ops_library(get_library_path())
sess = ort.InferenceSession(
    r"F:\OracleAI\models\all_MiniLM_L12_v2.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)
doc_minilm = embed(sess, texts)
q_minilm = embed(sess, [q for _, q in QUERIES])
for i, tid in enumerate(ids):
    cur.execute(
        "UPDATE techniques SET minilm_vector = :v WHERE id = :i",
        v=array.array("f", [float(x) for x in doc_minilm[i]]),
        i=tid,
    )
conn.commit()
n = cur.execute(
    "SELECT COUNT(*) FROM techniques WHERE minilm_vector IS NOT NULL"
).fetchone()[0]
print(f"minilm_vector populated: {n}/{len(rows)}")

e5 = json.load(open(r"C:\Temp\e5_vectors.json"))
for tid in ids:
    cur.execute(
        "UPDATE techniques SET e5_vector = :v WHERE id = :i",
        v=array.array("f", e5["e5_doc_vectors"][tid]),
        i=tid,
    )
conn.commit()
n = cur.execute(
    "SELECT COUNT(*) FROM techniques WHERE e5_vector IS NOT NULL"
).fetchone()[0]
print(f"e5_vector populated: {n}/{len(rows)}")

print()
for model, col, qvecs in [
    ("minilm", "minilm_vector", {qid: q_minilm[k] for k, (qid, _) in enumerate(QUERIES)}),
    ("e5", "e5_vector", {qid: e5["e5_query_vectors"][qid] for qid, _ in QUERIES}),
]:
    print(f"== {model} (ranking computed IN the database) ==")
    for qid, qtext in QUERIES:
        qv = qvecs[qid]
        qv = array.array("f", [float(x) for x in qv])
        got = cur.execute(
            f"SELECT title, VECTOR_DISTANCE({col}, :qv, COSINE) "
            "FROM techniques ORDER BY 2 FETCH FIRST 3 ROWS ONLY",
            qv=qv,
        ).fetchall()
        tops = " | ".join(f"{t} sim {1.0 - float(d):.4f}" for t, d in got)
        print(f"  {qid} ({qtext}): {tops}")
    print()
