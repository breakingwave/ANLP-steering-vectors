#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_bundle(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "persona_vector_bundle.json"
    return json.loads(path.read_text())


def _load_samples(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def analyze(
    *,
    run_dir: Path,
    min_score_gap: float,
    min_samples: int,
    min_selected_norm: float,
) -> tuple[dict[str, Any], list[tuple[str, bool, str]]]:
    bundle = _load_bundle(run_dir)
    samples = _load_samples(run_dir)

    pos = [s for s in samples if s.get("prompt_kind") == "positive"]
    neg = [s for s in samples if s.get("prompt_kind") == "negative"]
    pos_scores = [float(s["score"]) for s in pos if s.get("score") is not None]
    neg_scores = [float(s["score"]) for s in neg if s.get("score") is not None]

    layers = bundle.get("layers", [])
    norms = [(int(layer["layer_index"]), float(layer["norm"])) for layer in layers]
    selected_layer = int(bundle["selected_layer"])
    selected_norm = next((norm for layer, norm in norms if layer == selected_layer), None)
    ranked = sorted(norms, key=lambda item: item[1], reverse=True)
    selected_rank = next((i + 1 for i, (layer, _) in enumerate(ranked) if layer == selected_layer), None)

    pos_mean = _mean(pos_scores)
    neg_mean = _mean(neg_scores)
    score_gap = None if pos_mean is None or neg_mean is None else (pos_mean - neg_mean)

    summary = {
        "run_dir": str(run_dir),
        "model_name": bundle.get("model_name"),
        "trait_name": (bundle.get("trait") or {}).get("name"),
        "selection_strategy": bundle.get("selection_strategy"),
        "total_layers": len(ranked),
        "selected_layer": selected_layer,
        "selected_layer_rank_by_norm": selected_rank,
        "selected_layer_norm": selected_norm,
        "sample_count": int(bundle.get("sample_count", 0)),
        "positive_samples": len(pos),
        "negative_samples": len(neg),
        "positive_score_mean": pos_mean,
        "negative_score_mean": neg_mean,
        "score_gap_pos_minus_neg": score_gap,
        "top5_layers_by_norm": ranked[:5],
    }

    checks = [
        (
            "sample_count",
            summary["sample_count"] >= min_samples,
            f"sample_count={summary['sample_count']} >= {min_samples}",
        ),
        (
            "score_gap",
            score_gap is not None and score_gap >= min_score_gap,
            f"score_gap={_fmt(score_gap)} >= {min_score_gap}",
        ),
        (
            "selected_layer_norm",
            selected_norm is not None and selected_norm >= min_selected_norm,
            f"selected_norm={_fmt(selected_norm)} >= {min_selected_norm}",
        ),
    ]

    return summary, checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate persona-vector extraction quality from run outputs.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--min-score-gap", type=float, default=20.0)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-selected-norm", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary, checks = analyze(
        run_dir=args.run_dir,
        min_score_gap=args.min_score_gap,
        min_samples=args.min_samples,
        min_selected_norm=args.min_selected_norm,
    )

    if args.json:
        payload = {
            "summary": summary,
            "checks": [{"name": name, "ok": ok, "details": details} for name, ok, details in checks],
            "all_checks_pass": all(ok for _, ok, _ in checks),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Run: {summary['run_dir']}")
        print(f"Model: {summary['model_name']}")
        print(f"Trait: {summary['trait_name']}")
        print(f"Selection: {summary['selection_strategy']} (layer {summary['selected_layer']})")
        print(
            "Selected layer rank by norm: "
            f"{summary['selected_layer_rank_by_norm']} / {summary['total_layers']}"
        )
        print(f"Selected layer norm: {_fmt(summary['selected_layer_norm'])}")
        print(
            f"Samples: total={summary['sample_count']}, "
            f"positive={summary['positive_samples']}, negative={summary['negative_samples']}"
        )
        print(
            "Scores: "
            f"positive_mean={_fmt(summary['positive_score_mean'])}, "
            f"negative_mean={_fmt(summary['negative_score_mean'])}, "
            f"gap={_fmt(summary['score_gap_pos_minus_neg'])}"
        )
        top5 = ", ".join(f"L{layer}:{norm:.3f}" for layer, norm in summary["top5_layers_by_norm"])
        print(f"Top-5 layers by norm: {top5 if top5 else 'n/a'}")
        print("Checks:")
        for name, ok, details in checks:
            print(f"  - {name}: {'PASS' if ok else 'FAIL'} ({details})")
        print(f"Overall: {'PASS' if all(ok for _, ok, _ in checks) else 'FAIL'}")

    if args.strict and not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
