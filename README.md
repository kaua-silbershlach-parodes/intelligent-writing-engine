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
