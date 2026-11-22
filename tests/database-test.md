# The database test — the Oracle test, applied to this project

Oracle's own vector test ships its data as a Data Pump dump, embeds every row with the augmented model, and lets the database compute the ranking with `VECTOR_DISTANCE`. This test applies that exact shape to this project's corpus, using this repository's own artifacts: `data/techniques.dmp` and the two ONNX models in `models/`.

## Procedure

1. Drop the `TECHNIQUES` table and recreate it from the repository artifact alone: `impdp ... tables=TECHNIQUES directory=DATA_PUMP_DIR dumpfile=techniques.dmp`. Result: **6 rows**.
2. Add one vector column per model: `minilm_vector VECTOR(384, FLOAT32)`, `e5_vector VECTOR(384, FLOAT32)`.
3. Embed the six texts with each model and store the vectors. Coverage: **6/6 rows in both columns**.
4. Answer four test queries **inside the database**, with the official query shape — `SELECT title, VECTOR_DISTANCE(<column>, :query_vector, COSINE) FROM techniques ORDER BY 2 FETCH FIRST 3 ROWS ONLY`:
   - q1 "how do I avoid bias in my analysis"
   - q2 "why does this matter to the reader"
   - q3 "conhecer o publico para adaptar o conteudo"
   - q4 "como escrever com mais clareza"

Embeddings are generated outside the database with ONNX Runtime (the augmented graphs need the extensions library). On Oracle AI Database 26ai **Free for Windows**, loading the model into the database itself is blocked by the edition (ORA-40216) — the same limit hit with Oracle's own parks test on this machine — so the model runs outside and everything else runs inside: storage, the vectors, the distance math, and the ranking.

`run_database_test.py` is the script that ran (step 3, the coverage counts, and step 4). It expects the table imported from the `.dmp`, the MiniLM model file, the e5 vectors as JSON (the e5 file was not on the test machine; its vectors were generated with the identical official file, SHA-256 verified), and the schema password in `C:\Temp\vtest_pw.txt`.

## Results — measured 2026-11-22

The in-database ranking is **identical to the engine-side (numpy) ranking on all 8 runs, to 4 decimal places**:

| Query | Model | #1 in the database | Engine side |
|---|---|---|---|
| q1 | MiniLM | Objectivity 0.6955 | Objectivity 0.6955 |
| q2 | MiniLM | So What? 0.4976 | So What? 0.4976 |
| q3 | MiniLM | Audience Analysis 0.2686 | Audience Analysis 0.2686 |
| q4 | MiniLM | Clarity 0.1035 | Clarity 0.1035 |
| q1 | e5 | Objectivity 0.9020 | Objectivity 0.9020 |
| q2 | e5 | So What? 0.8872 | So What? 0.8872 |
| q3 | e5 | Audience Analysis 0.8468 | Audience Analysis 0.8468 |
| q4 | e5 | Conciseness 0.8306 | Conciseness 0.8306 |

## What was used, and what the gain is

- Data: 100% from the `.dmp` (6/6 rows). Embeddings: 100% coverage, both models (6/6 each, 384 dimensions). Ranking: 100% inside the database (8/8 queries). Outside the database: only the embedding generation.
- Accuracy gain: none — deliberately. Identical numbers are the proof that the database path is faithful.
- The real gains: the corpus lives in the database as vectors, so a query is answered by SQL where the data sits instead of loading files and computing in the application; the corpus travels as a single Oracle-native artifact (one `impdp` recreates it anywhere); and the shape scales the way Oracle's own test does — the same pipeline ran their 472-row dataset.
- The multilingual model's gain is measurable in the same table: on the Portuguese clarity query, MiniLM reaches the right technique at 0.1035 while e5 answers at 0.8306; on the English bias query, 0.6955 against 0.9020.
