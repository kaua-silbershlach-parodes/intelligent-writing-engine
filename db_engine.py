"""The database engine: the same six techniques and the same two models as
engine.py, searched with the database's full standard stack instead of a
local ONNX session. The corpus is the TECHNIQUES table (the content of
corpus.json / data/techniques.dmp); both vector columns are populated
inside the database by VECTOR_EMBEDDING, and every search mode below is
one of the patterns exercised by tests/run_full_suite_linux.py:

  search   exact search: the query is embedded inside the database with
           VECTOR_EMBEDDING and ranked with VECTOR_DISTANCE (FETCH EXACT)
  approx   approximate search (FETCH APPROX) over the vector indexes
           (HNSW on e5_vector, IVF on minilm_vector)
  hybrid   DBMS_HYBRID_VECTOR.SEARCH over the hybrid vector index
           (vector + text fusion; MiniLM model, fixed by the index)
  provider the query is embedded by the Private AI container, called from
           SQL with DBMS_VECTOR.UTL_TO_EMBEDDING, then ranked exactly
           (the container serves all-minilm-l12-v2; it does not serve
           multilingual-e5-small, so this mode is MiniLM-only)
  rag      retrieval for RAG: exact top-2 as context, with the top text
           split by DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS (grounded generation
           additionally needs an LLM endpoint; see the README)
  rest     the corpus served over ORDS REST (the TECHNIQUES table,
           REST-enabled), no SQL connection involved

Requirements: the lab schema prepared by tests/run_full_suite_linux.py
(models loaded in-database, TECHNIQUES populated, indexes created,
PRIVATE_AI_CONFIG holding the Private AI endpoint, ORDS serving for the
rest mode). Connection via environment: ORACLE_LAB_USER (default
"writing"), ORACLE_LAB_PW (default: the workshop's published lab
password), ORACLE_LAB_DSN (default localhost:1521/FREEPDB1), ORDS_URL
(default http://localhost:8080/ords/writing/techniques/).

Usage:
  python db_engine.py "how do I avoid bias in my analysis"
  python db_engine.py "como escrever com mais clareza" --model e5
  python db_engine.py "bias" --mode hybrid
  python db_engine.py "how do I avoid bias in my analysis" --mode rag
  python db_engine.py --mode rest
"""
import argparse
import json
import os
import sys
import urllib.request

import oracledb

USER = os.environ.get("ORACLE_LAB_USER", "writing")
PW = os.environ.get("ORACLE_LAB_PW", "Welcome_12345")
DSN = os.environ.get("ORACLE_LAB_DSN", "localhost:1521/FREEPDB1")
ORDS_URL = os.environ.get("ORDS_URL", "http://localhost:8080/ords/writing/techniques/")

MODELS = {
    "minilm": ("minilm_vector", "minilm_l12_v2", "all-minilm-l12-v2"),
    "e5": ("e5_vector", "e5_small", None),  # the container does not serve e5-small
}


def connect():
    return oracledb.connect(user=USER, password=PW, dsn=DSN)


def print_ranking(rows):
    for i, (tid, title, sim) in enumerate(rows, 1):
        print(f"{i}. [{tid}] {title} — {float(sim):.4f}")


def mode_search(cur, query, model, top, fetch):
    col, dbmodel, _ = MODELS[model]
    rows = cur.execute(
        f"SELECT id, title, 1 - VECTOR_DISTANCE({col}, "
        f"VECTOR_EMBEDDING({dbmodel} USING :q AS data), COSINE) "
        f"FROM techniques ORDER BY 3 DESC FETCH {fetch} FIRST :n ROWS ONLY",
        q=query,
        n=top,
    ).fetchall()
    print_ranking(rows)


def mode_hybrid(cur, query, top):
    raw = cur.execute(
        "SELECT JSON_SERIALIZE(DBMS_HYBRID_VECTOR.SEARCH("
        "JSON('{\"hybrid_index_name\":\"TECH_HYBRID_IDX\",\"search_text\":"
        + json.dumps(query)
        + "}')) RETURNING VARCHAR2(32767)) FROM dual"
    ).fetchall()[0][0]
    hits = json.loads(raw)[:top]
    titles = {
        text: (tid, title)
        for tid, title, text in cur.execute(
            "SELECT id, title, text FROM techniques"
        ).fetchall()
    }
    for i, hit in enumerate(hits, 1):
        chunk = hit["chunk_text"]
        owner = next(
            (v for text, v in titles.items() if text.startswith(chunk[:40])),
            ("?", "?"),
        )
        print(
            f"{i}. [{owner[0]}] {owner[1]} — score {hit['score']:.2f} "
            f"(vector {hit['vector_score']:.2f}, text {hit['text_score']:.2f})"
        )
        print(f"   {chunk[:110]}...")


def mode_provider(cur, query, model, top):
    col, _, provider_model = MODELS[model]
    if provider_model is None:
        print(
            f"provider mode: the running Private AI container does not serve "
            f"a model for --model {model}; use --model minilm",
            file=sys.stderr,
        )
        sys.exit(2)
    endpoint = cur.execute(
        "SELECT config_value FROM private_ai_config WHERE config_name = 'HTTP_ENDPOINT'"
    ).fetchall()[0][0]
    rows = cur.execute(
        f"SELECT id, title, 1 - VECTOR_DISTANCE({col}, "
        "DBMS_VECTOR.UTL_TO_EMBEDDING(:q, JSON_OBJECT("
        "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
        "'host' VALUE 'local', 'model' VALUE :m RETURNING JSON)), COSINE) "
        "FROM techniques ORDER BY 3 DESC FETCH EXACT FIRST :n ROWS ONLY",
        q=query,
        u=endpoint + "/v1/embeddings",
        m=provider_model,
        n=top,
    ).fetchall()
    print_ranking(rows)


def mode_rag(cur, query, model):
    col, dbmodel, _ = MODELS[model]
    rows = cur.execute(
        f"SELECT id, title, text FROM techniques ORDER BY VECTOR_DISTANCE({col}, "
        f"VECTOR_EMBEDDING({dbmodel} USING :q AS data), COSINE) "
        "FETCH EXACT FIRST 2 ROWS ONLY",
        q=query,
    ).fetchall()
    print("Retrieved context:")
    for tid, title, _ in rows:
        print(f"  [{tid}] {title}")
    top_id, top_title, _ = rows[0]
    chunks = cur.execute(
        "SELECT CARDINALITY(DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(text, "
        "JSON('{\"by\":\"words\",\"max\":10}'))) FROM techniques WHERE id = :i",
        i=top_id,
    ).fetchall()[0][0]
    print(
        f"Top text ({top_title}) split for the prompt: {chunks} chunks "
        "at a 10-word maximum (UTL_TO_CHUNKS)."
    )
    print(
        "Grounded generation runs through DBMS_VECTOR_CHAIN.UTL_TO_GENERATE_TEXT "
        "when an LLM endpoint is configured; on the current lab host it is "
        "blocked (see the README's Linux suite section)."
    )


def mode_rest(top):
    with urllib.request.urlopen(ORDS_URL + f"?limit={top}", timeout=15) as resp:
        doc = json.load(resp)
    for i, item in enumerate(doc.get("items", []), 1):
        print(f"{i}. [{item['id']}] {item['title']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--model", choices=["minilm", "e5"], default="e5")
    ap.add_argument(
        "--mode",
        choices=["search", "approx", "hybrid", "provider", "rag", "rest"],
        default="search",
    )
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    if args.mode == "rest":
        mode_rest(args.top)
        return
    if not args.query:
        ap.error("a query is required for every mode except rest")

    conn = connect()
    cur = conn.cursor()
    if args.mode == "search":
        mode_search(cur, args.query, args.model, args.top, "EXACT")
    elif args.mode == "approx":
        mode_search(cur, args.query, args.model, args.top, "APPROX")
    elif args.mode == "hybrid":
        mode_hybrid(cur, args.query, args.top)
    elif args.mode == "provider":
        mode_provider(cur, args.query, args.model, args.top)
    elif args.mode == "rag":
        mode_rag(cur, args.query, args.model)
    conn.close()


if __name__ == "__main__":
    main()
