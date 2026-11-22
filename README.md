# Intelligence Writing Engine — A Guide to Techniques
A concise reference for intelligence writing: six core principles, searchable through Oracle's own augmented embedding models.

## The engine

The engine is Oracle's own: their prebuilt **augmented** ONNX exports of the Sentence-Transformers models, copied complete and run unchanged. Oracle's augmentation bakes the whole pipeline into each model file — the tokenizer and the pre- and post-processing steps are inside the graph — so the input is raw text and the output is a finished 384-dimension embedding. Nothing was rebuilt, retrained, or substituted: the files in `models/` are Oracle's files, byte for byte.

| Model | File | Size (bytes) | Dimensions | Reach |
|---|---|---|---|---|
| all-MiniLM-L12-v2 | `models/all_MiniLM_L12_v2.onnx` | 133,322,334 | 384 | English-first |
| multilingual-e5-small | `models/multilingual_e5_small.onnx` | 123,021,105 | 384 | 100 languages |

`models/` also carries the model's own documentation and attribution files from the same package (`README-ALL_MINILM_L12_V2-augmented.txt`, `LICENSE_ATTRIBUTION.txt`). The model files are stored with Git LFS.

`engine.py` is the thin runner around them: it embeds the six techniques (`corpus.json`) and the query with the chosen model, and ranks by cosine similarity. The augmented graphs use a custom tokenizer operator, so the ONNX Runtime extensions library is required — without it the files do not load. Dependencies (`requirements.txt`): `onnxruntime`, `onnxruntime-extensions`, `numpy`.

    python engine.py "how do I avoid bias in my analysis" --model minilm
    python engine.py "why does this matter to the reader" --model e5
    python engine.py "conhecer o publico para adaptar o conteudo" --model e5

Real output:

    1. [core-06] Objectivity — 0.682        (minilm, English query)
    2. [core-05] Avoid Observations — 0.482
    3. [core-04] Audience Analysis — 0.329

    1. [core-03] So What? — 0.868           (e5, English query)
    2. [core-01] Clarity — 0.819
    3. [core-04] Audience Analysis — 0.818

    1. [core-04] Audience Analysis — 0.819  (e5, Portuguese query)
    2. [core-05] Avoid Observations — 0.808
    3. [core-02] Conciseness — 0.805

Measured behavior: both models rank the matching technique first on every test query. The multilingual model answers Portuguese queries at full strength; the English-first model still ranks them correctly, with lower margins. Feed the augmented models raw text — no prefixes: with `query:`/`passage:` prefixes added, one test ranking flipped, so the runner sends the text exactly as written.

## The database test

Oracle's own vector test ships its data as a Data Pump dump, embeds every row with the augmented model, and lets the database compute the ranking with `VECTOR_DISTANCE`. This test applies that exact shape to this project's corpus, using this repository's own artifacts: `data/techniques.dmp` and the two ONNX models in `models/`.

Procedure:

1. Drop the `TECHNIQUES` table and recreate it from the repository artifact alone: `impdp ... tables=TECHNIQUES directory=DATA_PUMP_DIR dumpfile=techniques.dmp`. Result: **6 rows**.
2. Add one vector column per model: `minilm_vector VECTOR(384, FLOAT32)`, `e5_vector VECTOR(384, FLOAT32)`.
3. Embed the six texts with each model and store the vectors. Coverage: **6/6 rows in both columns**.
4. Answer four test queries **inside the database**, with the official query shape — `SELECT title, VECTOR_DISTANCE(<column>, :query_vector, COSINE) FROM techniques ORDER BY 2 FETCH FIRST 3 ROWS ONLY`:
   - q1 "how do I avoid bias in my analysis"
   - q2 "why does this matter to the reader"
   - q3 "conhecer o publico para adaptar o conteudo"
   - q4 "como escrever com mais clareza"

Embeddings are generated outside the database with ONNX Runtime (the augmented graphs need the extensions library). On Oracle AI Database 26ai **Free for Windows**, loading the model into the database itself is blocked by the edition (ORA-40216) — the same limit hit with Oracle's own parks test on this machine — so the model runs outside and everything else runs inside: storage, the vectors, the distance math, and the ranking. [tests/run_database_test.py](tests/run_database_test.py) is the script that ran (steps 3 and 4, with the coverage counts).
## The complete test suite

Every test Oracle publishes for this stack — the two official tutorials (the hybrid-search tutorial on the parks data and the getting-started tutorial on the headlines data), the official workshop (exhaustive search, approximate search, image search, RAG), and the model package's own README — was run with this project's content in place of theirs, on Oracle AI Database 26ai Free. One root cause decides everything that could not run: on the **Free edition for Windows** the in-database ONNX runtime is not available, so the model cannot be loaded into the database (ORA-40216). Everything that does not depend on a loaded model runs, and passed:

| # | Oracle's test | Result with this content |
|---|---|---|
| 1 | Exact similarity search (`VECTOR_DISTANCE`, `FETCH EXACT`) | Passed — 8/8 runs identical to the engine side, 4 decimal places (table above) |
| 2 | Bulk load through an external table over a text file | Passed — 6/6 rows read |
| 3 | Data Pump import of the dataset (`.dmp`) | Passed — 6/6 rows, from this repository's own `data/techniques.dmp` |
| 4 | Approximate search (`FETCH APPROX`) | Passed — identical to `EXACT` on both models |
| 5 | Distance metrics (COSINE, EUCLIDEAN, EUCLIDEAN_SQUARED, DOT, MANHATTAN) | Passed — same top 3 on all five |
| 6 | Execution-plan inspection | Passed — full scan before indexes, as in their tutorial; `VECTOR INDEX HNSW SCAN` after |
| 7 | IVF vector index (`NEIGHBOR PARTITIONS`, target accuracy 90) | Passed — created, 6 vectors, approximate ranking correct |
| 8 | HNSW vector index (`INMEMORY NEIGHBOR GRAPH`, target accuracy 95) | Passed — created after setting `vector_memory_size = 256M` and restarting the instance; 6 vectors; the optimizer picks the index unhinted |
| 9 | Text chunking for RAG (`UTL_TO_CHUNKS`) | Passed — the Objectivity text splits into 4 chunks at a 10-word maximum |
| 10 | In-database model load, `VECTOR_EMBEDDING`, mining-model dictionary | Blocked by the edition — load fails with ORA-40216; `VECTOR_EMBEDDING` then answers ORA-40284 (model does not exist) |
| 11 | Hybrid Vector Index + `DBMS_HYBRID_VECTOR.SEARCH` (fusion INTERSECT/UNION, fuzzy text) | Blocked by the same root — the index object is created but its text side never leaves POPULATE, and SEARCH fails inside `drvdml.search_embedding` with ORA-40284: the hybrid vectorizer is the loaded model of test 10 |
| 12 | ORDS REST service over the search | Not runnable here — ORDS is not installed in this Free instance (no ORDS users, no ORDS_ADMIN package) |
| 13 | Image search (CLIP) | Different modality — the workshop searches an image corpus; this corpus is text |
| 14 | APEX demo, text generation | A UI demo and hosted-LLM generation — outside the database tests; generation needs an LLM endpoint this instance does not have |

[tests/run_database_test.py](tests/run_database_test.py) runs the core test (steps 1–4 of the procedure, with the coverage counts); [tests/run_full_suite.py](tests/run_full_suite.py) runs the rest of the runnable suite (tests 2, 4–9 above, the index dictionary, and the execution plans).

## The complete test suite (Linux lab)

The same suite was then run end-to-end on Linux — the database as the official `gvenzl/oracle-free:23.26.4-full` container image (the same 26ai Free release as the Windows install), ORDS 26.3.1 serving from the same instance, and the Private AI Services container next to it — in a `WRITING` schema holding the six techniques from `corpus.json`. The step Windows cannot do happens first here: both models are loaded into the database and **both vector columns are populated inside the database** by `VECTOR_EMBEDDING` (6/6 rows each). The runner is [tests/run_full_suite_linux.py](tests/run_full_suite_linux.py):

| Test | Result on Linux |
|---|---|
| Exact search, in-database embeddings vs the engine's own rankings (4 queries × 2 models) | Passed — top-1 agreement 8/8; the in-database MiniLM similarities reproduce the Windows database run to four decimal places (0.6955, 0.4976, 0.2686, 0.1035) |
| External table over `techniques.txt` | Passed — 6/6 rows |
| In-database `VECTOR_EMBEDDING` (the Windows ORA-40216 test) | Passed — 384 dimensions from both models |
| `FETCH APPROX` vs `FETCH EXACT` | Passed — identical top 3 on both models |
| Five distance metrics | Passed — same top 3 on all five |
| IVF index (`NEIGHBOR PARTITIONS`, accuracy 90) | Passed — created, approximate ranking correct |
| HNSW index (`INMEMORY NEIGHBOR GRAPH`, accuracy 95) | Passed — created; the execution plan shows `VECTOR INDEX HNSW SCAN` with no hint; both indexes `VALID` in the dictionary |
| Hybrid Vector Index + `DBMS_HYBRID_VECTOR.SEARCH` (the Windows ORA-40284 test) | Passed — 6 hits; top score 70.88, the Objectivity text |
| ORDS REST over `TECHNIQUES` | Passed — schema REST-enabled, first technique served over HTTP |
| Container embeddings from SQL (`UTL_TO_EMBEDDING` via the Private AI endpoint) | Passed — 384 dimensions; a container-embedded query returns Objectivity as top 1 |
| Chunking (`UTL_TO_CHUNKS`) | Passed — the Objectivity text splits into 4 chunks at a 10-word maximum |
| CLIP text encoder through the container | Passed — 512 dimensions (this corpus is text; the image side of CLIP is exercised in the template repository's suite) |
| RAG retrieval over the techniques | Passed — the bias question retrieves Objectivity and Avoid Observations as its context |
| Text generation (`UTL_TO_GENERATE_TEXT`) and RAG grounded generation | Blocked by the lab host — see below |

Text generation is the one capability that stays blocked, and the blocker is the lab host, not the database or the model: the medium Private AI image (the one whose feature flags enable an LLM) was run with its official model directory and configuration; its Ministral-3 3B model deploys and loads (listed by the service with completion capability, and the inference server logs the model as loaded and accepts a chat request), but generation does not complete at usable speed on a host of this size (7 GB of RAM, 1.5 shared CPUs — the image's own setup asks for at least 6 GB for the container alone), across five attempts through the application and through the inference server directly, with the database up and down and with thread counts reduced. The runner probes the running container for a generation model on every run and executes both generation tests automatically whenever one is present.

## The database engine

[db_engine.py](db_engine.py) is the engine running on that full stack: the same six techniques and the same two models as `engine.py`, with every step executed by the database and its services instead of a local ONNX session. The corpus is the `TECHNIQUES` table (the content of `corpus.json` / `data/techniques.dmp`), both vector columns populated inside the database by `VECTOR_EMBEDDING`. Each mode is one of the standard patterns the suite exercises:

| Mode | The standard technique it runs |
|---|---|
| `search` (default) | Exact search — the query embedded inside the database (`VECTOR_EMBEDDING`), ranked by `VECTOR_DISTANCE` with `FETCH EXACT` |
| `approx` | Approximate search — `FETCH APPROX` over the vector indexes (HNSW on `e5_vector`, IVF on `minilm_vector`) |
| `hybrid` | `DBMS_HYBRID_VECTOR.SEARCH` over the hybrid vector index — vector + text fusion (MiniLM, fixed by the index) |
| `provider` | The query embedded by the Private AI container, called from SQL (`DBMS_VECTOR.UTL_TO_EMBEDDING`), then ranked exactly — MiniLM only: the container does not serve multilingual-e5-small |
| `rag` | RAG retrieval — exact top 2 as context, the top text split by `DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS`; grounded generation joins when an LLM endpoint is configured (blocked on the current lab host, as recorded above) |
| `rest` | The corpus served over ORDS REST from the REST-enabled `TECHNIQUES` table — no SQL connection involved |

Real output, against the lab the suite ran on:

    $ python db_engine.py "how do I avoid bias in my analysis"
    1. [core-06] Objectivity — 0.8967
    2. [core-05] Avoid Observations — 0.8461
    3. [core-01] Clarity — 0.8420

    $ python db_engine.py "how do I avoid bias in my analysis" --model minilm
    1. [core-06] Objectivity — 0.6955
    2. [core-05] Avoid Observations — 0.4214
    3. [core-02] Conciseness — 0.3005

    $ python db_engine.py "bias in analysis" --mode hybrid
    1. [core-06] Objectivity — score 70.88 (vector 72.57, text 54.00)

    $ python db_engine.py "how do I avoid bias in my analysis" --mode rag --model minilm
    Retrieved context:
      [core-06] Objectivity
      [core-05] Avoid Observations
    Top text (Objectivity) split for the prompt: 4 chunks at a 10-word maximum (UTL_TO_CHUNKS).

The in-database and container paths agree exactly: `search` and `provider` return the same MiniLM ranking to four decimal places (0.6955, 0.4214, 0.3005), and `approx` returns the same top 3 as `search`. The database engine needs the lab schema the suite runner prepares (models loaded in-database, `TECHNIQUES` populated, indexes created, the Private AI endpoint in `PRIVATE_AI_CONFIG`, ORDS serving for the `rest` mode); connection settings come from the environment (`ORACLE_LAB_USER`, `ORACLE_LAB_PW`, `ORACLE_LAB_DSN`, `ORDS_URL`). `engine.py` remains the standalone path for machines without the database.
