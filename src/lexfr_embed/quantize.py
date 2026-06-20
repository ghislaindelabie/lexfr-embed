"""Deliverable O3 — "downsize" the best model into deployment variants (research §03).

Offer an accuracy/latency/size trade-off:
  - **Matryoshka dim truncation** (1024 -> 512 -> 256 -> 128 -> 64): truncate + renormalise
    stored vectors on CPU, ZERO re-encode (~96% quality at 64-d). Cheapest, do first.
  - **fp16** export (half the size, ~same quality).
  - **int8 / ONNX** via `optimum` + ONNX Runtime (CPU-friendly serving).
  - GGUF: optional.
Each variant is re-evaluated (evaluate.py) → a single trade-off table for the report.
"""

from __future__ import annotations


def truncate_matryoshka(embeddings, dim: int):
    """Truncate to `dim` and L2-renormalise (pure — no model needed). TODO: implement
    with numpy/torch; this is the zero-cost dimensionality lever."""
    raise NotImplementedError("Truncate to dim + L2-normalise.")


def export_onnx_int8(model_path: str, out_dir: str):
    """TODO(Phase 1): optimum.onnxruntime export + dynamic int8 quantization."""
    raise NotImplementedError("Export ONNX + int8 via optimum.")
