#!/usr/bin/env python3
"""Audit the supplied byte classifier and extract its unreachable payload."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct

import numpy as np


def load_model(path):
    """Read this artifact's F32 safetensors without executing model code."""
    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_size])
    data_start = 8 + header_size
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(f"Unsupported dtype: {info['dtype']}")
        start, end = info["data_offsets"]
        if not 0 <= start <= end <= len(raw) - data_start:
            raise ValueError(f"Invalid tensor bounds: {name}")
        values = np.frombuffer(raw[data_start + start:data_start + end], dtype="<f4")
        tensors[name] = values.reshape(info["shape"]).astype(np.float64)
    return tensors, header, data_start, raw


def extract_flag(embedding):
    """Find an affine encoding of the known flag prefix in non-ASCII rows.

    Infer scale and offset from S and I, then verify all eight prefix bytes.
    Neither the suffix nor its length is hardcoded.
    """
    prefix = np.frombuffer(b"SIGMOID_", dtype=np.uint8).astype(float)
    candidates = []
    for row in range(128, len(embedding)):
        for column in range(embedding.shape[1] - len(prefix) + 1):
            values = embedding[row, column:]
            difference = values[0] - values[1]
            if difference == 0:
                continue
            scale = (prefix[0] - prefix[1]) / difference
            offset = prefix[0] - scale * values[0]
            decoded = values * scale + offset
            if not np.allclose(decoded[:len(prefix)], prefix, atol=1e-6, rtol=0):
                continue
            codes = []
            for value in decoded:
                code = int(round(float(value)))
                if abs(value - code) > 1e-6 or not (65 <= code <= 90 or code == 95):
                    break
                codes.append(code)
            flag = bytes(codes).decode("ascii")
            if re.fullmatch(r"SIGMOID_[A-Z]+", flag):
                candidates.append((flag, row, column, scale, offset))
    if len(candidates) != 1:
        raise ValueError(f"Expected one payload, found {len(candidates)}")
    return candidates[0]


def logits_for(records, embedding, weight, bias):
    pooled = np.array([
        embedding[np.frombuffer(record["text"].encode("ascii"), dtype=np.uint8)].mean(0)
        for record in records
    ])
    return pooled @ weight.T + bias


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=folder / "model.safetensors")
    parser.add_argument("--corpus", type=Path, default=folder / "corpus.csv")
    parser.add_argument("--output-dir", type=Path, default=folder)
    args = parser.parse_args()

    tensors, header, data_start, raw = load_model(args.model)
    embedding = tensors["embedding.weight"]
    weight = tensors["classifier.weight"]
    bias = tensors["classifier.bias"]
    flag, row, column, scale, offset = extract_flag(embedding)
    assert flag.encode("ascii") not in raw, "Flag unexpectedly exists as literal text"

    with args.corpus.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert all(len(record["text"]) == 64 for record in records)
    logits = logits_for(records, embedding, weight, bias)
    predicted = logits.argmax(1)
    labels = sorted({record["register"] for record in records})
    # Safetensors contains no class-name metadata. Resolve the six mappings.
    mapping = max(itertools.permutations(labels), key=lambda order: sum(
        order[int(index)] == record["register"]
        for index, record in zip(predicted, records)
    ))
    confusion = np.zeros((3, 3), dtype=int)
    for record, index in zip(records, predicted):
        confusion[mapping.index(record["register"]), index] += 1
    correct = int(np.trace(confusion))

    # Erase the entire carrier row in a copy; leave the shipped file untouched.
    scrubbed = embedding.copy()
    scrubbed[row] = 0
    ascii_unchanged = bool(np.array_equal(embedding[:128], scrubbed[:128]))
    scrubbed_logits = logits_for(records, scrubbed, weight, bias)
    max_delta = float(np.max(np.abs(logits - scrubbed_logits)))
    assert ascii_unchanged and np.array_equal(logits, scrubbed_logits)

    byte_offset = (
        data_start + header["embedding.weight"]["data_offsets"][0]
        + (row * embedding.shape[1] + column) * 4
    )
    report = {
        "flag": flag,
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "payload": {
            "tensor": "embedding.weight", "row": row, "column_start": column,
            "length": len(flag), "decode_scale": scale, "decode_offset": offset,
            "file_byte_offset": byte_offset, "file_byte_length": len(flag) * 4,
            "literal_flag_present": False,
        },
        "audit": {
            "architecture": "mean byte embeddings, then linear classifier",
            "class_index_mapping": list(mapping),
            "records": len(records), "correct": correct,
            "accuracy": correct / len(records),
            "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
            "all_ascii_embeddings_unchanged_after_erasure": ascii_unchanged,
            "corpus_max_logit_change_after_erasure": max_delta,
            "corpus_predictions_unchanged_after_erasure": bool(np.array_equal(
                predicted, scrubbed_logits.argmax(1)
            )),
        },
        "input_sha256": {
            args.model.name: hashlib.sha256(raw).hexdigest(),
            args.corpus.name: hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "flag.txt").write_text(flag + "\n", encoding="utf-8")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(flag)
    print(f"Carrier: embedding.weight[{row}, {column}:{column + len(flag)}]")
    print(f"Decode: round(value * {scale:g} + {offset:g})")
    print(f"Accuracy: {correct}/{len(records)} = {correct / len(records):.6%}")
    print(f"Maximum corpus logit change after erasure: {max_delta:g}")


if __name__ == "__main__":
    main()
