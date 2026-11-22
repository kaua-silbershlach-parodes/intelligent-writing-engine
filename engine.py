#!/usr/bin/env python3
"""Semantic search over the six intelligence writing techniques.

The engine is Oracle's own augmented ONNX embedding models, run
unchanged: the tokenizer and the pre/post-processing are baked into
each model file by Oracle, so the input is raw text and the output is
a 384-dimension embedding. Ranking is cosine similarity between the
query embedding and each technique's embedding.

Models (in models/, Oracle's prebuilt augmented exports):
  minilm  all_MiniLM_L12_v2.onnx   - English-first, 384 dimensions
  e5      multilingual_e5_small.onnx - 100 languages, 384 dimensions

Usage:
  python engine.py "how do I avoid bias in my analysis"
  python engine.py "como escrever com clareza" --model e5 --top 3

Requires: onnxruntime, onnxruntime-extensions, numpy
(see requirements.txt). The augmented graphs use a custom tokenizer
operator from the ONNX Runtime extensions library; without it the
model files do not load.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import get_library_path

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "minilm": os.path.join(ROOT, "models", "all_MiniLM_L12_v2.onnx"),
    "e5": os.path.join(ROOT, "models", "multilingual_e5_small.onnx"),
}
DIMS = 384


def load_session(model_path):
    opts = ort.SessionOptions()
    opts.register_custom_ops_library(get_library_path())
    return ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


def embed(session, texts):
    inp = session.get_inputs()[0]
    outs = session.run(None, {inp.name: np.array(texts, dtype=object)})
    for out in outs:
        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == DIMS:
            return arr
    raise RuntimeError("model produced no 384-dimension embedding output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="the question to match against the techniques")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="e5",
        help="e5 (multilingual, default) or minilm (English-first)",
    )
    parser.add_argument("--top", type=int, default=3, help="how many results to print")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)

    session = load_session(MODELS[args.model])
    docs = [f"{item['title']}. {item['text']}" for item in corpus]
    doc_vecs = embed(session, docs)
    query_vec = embed(session, [args.query])[0]

    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    query_vec = query_vec / np.linalg.norm(query_vec)
    scores = doc_vecs @ query_vec

    order = np.argsort(-scores)[: args.top]
    for rank, idx in enumerate(order, 1):
        print(f"{rank}. [{corpus[idx]['id']}] {corpus[idx]['title']} — {scores[idx]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
