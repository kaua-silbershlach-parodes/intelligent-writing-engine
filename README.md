# Intelligence Writing Engine — A Guide to Techniques
A concise reference for intelligence writing: six core principles, searchable through Oracle's own augmented embedding models.

## The engine

The engine is Oracle's own: their prebuilt **augmented** ONNX exports of the Sentence-Transformers models, copied complete and run unchanged. Oracle's augmentation bakes the whole pipeline into each model file — the tokenizer and the pre- and post-processing steps are inside the graph — so the input is raw text and the output is a finished 384-dimension embedding. Nothing was rebuilt, retrained, or substituted: the files in `models/` are Oracle's files, byte for byte.

| Model | File | Size (bytes) | Dimensions | Reach |
|---|---|---|---|---|
| all-MiniLM-L12-v2 | `models/all_MiniLM_L12_v2.onnx` | 133,322,334 | 384 | English-first |
| multilingual-e5-small | `models/multilingual_e5_small.onnx` | 123,021,105 | 384 | 100 languages |

`models/` also carries the model's own documentation and attribution files from the same package (`README-ALL_MINILM_L12_V2-augmented.txt`, `LICENSE_ATTRIBUTION.txt`). The model files are stored with Git LFS.

`engine.py` is the thin runner around them: it embeds the six techniques below (`corpus.json`) and the query with the chosen model, and ranks by cosine similarity. The augmented graphs use a custom tokenizer operator, so the ONNX Runtime extensions library is required — without it the files do not load. Dependencies (`requirements.txt`): `onnxruntime`, `onnxruntime-extensions`, `numpy`.

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

## Intelligence writing techniques

Six techniques, applied to this README itself:

1. **Clarity** — Present findings clearly to avoid ambiguity or misinterpretation. Every word should be considered, questioned, and critiqued.
2. **Conciseness** — Eliminate unnecessary words and focus on the most essential information.
3. **So What?** — Answer why this is important, why it matters, and why it deserves the reader's time. If it doesn't answer the "so what?", the product is lacking.
4. **Audience Analysis** — Tailor the content to meet the intelligence requirements and expectations of the primary (and secondary) audience.
5. **Avoid Observations** — Observations aren't analytical. Determine whether your assessment is analytical and forward-leaning, and not just an observation.
6. **Objectivity** — Check your biases. Check them again. Now have a peer or colleague review your work. Avoid any biases that could skew your analysis or impact your or your organization's reputation.

For the database, the six techniques are kept the way Oracle distributes its own test data: a Data Pump export, [data/techniques.dmp](data/techniques.dmp) — table `TECHNIQUES`, 6 rows, 212,992 bytes, SHA-256 `bf5a0bc123ca0edb2db05a33d02212e0f50bda3bf76fbb5d885129c65687d3a0`. Import it with `impdp` to recreate the table; the round trip was verified (export, re-import, 6 rows back). `corpus.json` is the same content in the form `engine.py` reads. The full database test — this corpus, both models, rankings computed inside the database — is recorded in [tests/database-test.md](tests/database-test.md).
