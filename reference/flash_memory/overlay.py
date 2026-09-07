"""Versioned FP32 deltas plus a checked absolute-row export for llama.cpp.

Original GGUF files are opened read-only. Inference never calls a teacher.
"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
from .artifacts import atomic_json, canonical, digest, file_hash

MAGIC = b"FMLROW1\0"

def validate_arrays(rows, anchors, delta, table):
    if rows.dtype != np.dtype("int64") or rows.ndim != 1:
        raise ValueError("Rows must be an int64 vector")
    if len(rows) > 131072 or np.any(np.diff(rows) <= 0):
        raise ValueError("Rows must be unique, sorted, and within the experimental budget")
    if anchors.dtype != np.float32 or delta.dtype != np.float32 or anchors.shape != (len(rows), table.dim) or delta.shape != anchors.shape:
        raise ValueError("Invalid FP32 anchor/delta shape")
    if not np.all(np.isfinite(anchors)) or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(anchors + delta)):
        raise ValueError("Non-finite overlay values")
    original = table.read_global(rows)
    if not np.array_equal(anchors, original):
        raise ValueError("Anchors differ from the exact original decoded rows")

def export_overlay(directory, identity, table, rows, delta, provenance, status="experiment"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows = np.asarray(rows, dtype=np.int64)
    anchors = table.read_global(rows)
    delta = np.asarray(delta, dtype=np.float32)
    validate_arrays(rows, anchors, delta, table)
    if status != "experiment":
        raise ValueError("Only the acceptance controller can release an experiment")
    np.savez(directory / "delta.npz", rows=rows, anchors=anchors, delta=delta)
    payload = rows.astype("<i4").tobytes() + (anchors + delta).astype("<f4").tobytes()
    header = {"schema": "flash-memory-overlay/v1", "model_identity": identity,
              "row_count": len(rows), "row_dim": table.dim,
              "payload_sha256": hashlib.sha256(payload).hexdigest(), "status": status,
              "provenance": provenance, "delta_sha256": file_hash(directory / "delta.npz")}
    raw_header = canonical(header)
    (directory / "rows.fml").write_bytes(MAGIC + struct.pack("<I", len(raw_header)) + raw_header + payload)
    scales = np.maximum(np.sqrt(np.mean(anchors * anchors, axis=1)), 1e-6)
    displacement = np.sqrt(np.mean(delta * delta, axis=1)) / scales if len(rows) else np.zeros(0)
    manifest = {**header, "overlay_sha256": file_hash(directory / "rows.fml"),
                "normalized_rms_displacement_max": float(displacement.max(initial=0)),
                "normalized_rms_displacement_mean": float(displacement.mean()) if len(rows) else 0.0}
    manifest["manifest_sha256"] = digest(manifest)
    atomic_json(directory / "manifest.json", manifest)
    return manifest

def load_overlay(directory, identity, table):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Modified overlay manifest")
    if manifest["model_identity"] != identity:
        raise ValueError("Incompatible model identity")
    if file_hash(directory / "rows.fml") != manifest["overlay_sha256"] or file_hash(directory / "delta.npz") != manifest["delta_sha256"]:
        raise ValueError("Overlay checksum mismatch")
    with np.load(directory / "delta.npz", allow_pickle=False) as arrays:
        rows, anchors, delta = (arrays[k].copy() for k in ("rows", "anchors", "delta"))
    validate_arrays(rows, anchors, delta, table)
    return rows, anchors, delta
