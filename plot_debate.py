#!/usr/bin/env python3
"""Plot per-turn projections from simulate_persona_debate.py output.

Reads dialogue.json files (one per topic) and renders:
  - one plot per topic showing agent-A and agent-B projections vs turn number
  - optionally a combined overlay across topics

Usage:
  # single topic
  python plot_debate.py \
      --dialogue-json runs/debate_evil_vs_good_.../01_.../dialogue.json \
      --output runs/debate_.../01_.../plot.png

  # whole run (auto-discovers every topic subfolder)
  python plot_debate.py --run-dir runs/debate_evil_vs_good_20260418_223830

  # project against a specific column only (default: use whatever the file has)
  python plot_debate.py --run-dir runs/... --projection-key projection_on_evil
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── IO ────────────────────────────────────────────────────────────────────────


def load_dialogue(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _pick_projection_key(turn_record: dict, override: str | None) -> str:
    """Choose which projection column to plot.

    If the user passed --projection-key, honor it. Otherwise pick the first
    key that starts with 'projection_on_'.
    """
    if override:
        if override not in turn_record:
            raise KeyError(f"projection key '{override}' not in turn record. "
                           f"Available: {[k for k in turn_record if k.startswith('projection_on_')]}")
        return override
    for k in turn_record:
        if k.startswith("projection_on_"):
            return k
    raise KeyError("No 'projection_on_*' key in turn record.")


def _smooth(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    out = np.full_like(arr, np.nan)
    half = window // 2
    for i in range(len(arr)):
        chunk = arr[max(0, i - half):min(len(arr), i + half + 1)]
        valid = chunk[~np.isnan(chunk)]
        out[i] = valid.mean() if len(valid) else np.nan
    return out


# ── Single-topic plot ─────────────────────────────────────────────────────────


def plot_single_topic(
    dialogue: dict,
    out_path: Path,
    projection_key: str | None = None,
    smooth_window: int = 5,
) -> None:
    topic = dialogue.get("topic", "unknown topic")
    agent_a = dialogue["agent_a"]
    agent_b = dialogue["agent_b"]
    turns = dialogue["turns"]

    if not turns:
        print(f"[warn] no turns in {out_path.name} — skipping")
        return

    key = _pick_projection_key(turns[0], projection_key)

    a_turns_x: list[int] = []
    a_vals: list[float] = []
    b_turns_x: list[int] = []
    b_vals: list[float] = []
    for t in turns:
        if t["speaker"] == "A":
            a_turns_x.append(t["turn"])
            a_vals.append(t[key])
        else:
            b_turns_x.append(t["turn"])
            b_vals.append(t[key])

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5, zorder=0)

    ax.plot(a_turns_x, a_vals, color="crimson", marker="o", markersize=7,
            linewidth=1.8, label=f"A ({agent_a['name']})")
    ax.plot(b_turns_x, b_vals, color="steelblue", marker="s", markersize=7,
            linewidth=1.8, label=f"B ({agent_b['name']})")

    if smooth_window > 1 and len(a_vals) >= 3:
        ax.plot(a_turns_x, _smooth(a_vals, smooth_window),
                color="crimson", alpha=0.35, linewidth=3.0, zorder=0)
    if smooth_window > 1 and len(b_vals) >= 3:
        ax.plot(b_turns_x, _smooth(b_vals, smooth_window),
                color="steelblue", alpha=0.35, linewidth=3.0, zorder=0)

    pretty_key = key.replace("projection_on_", "")
    ax.set_xlabel("Turn number")
    ax.set_ylabel(f"Projection onto {pretty_key} vector (layer {agent_a['layer']})")
    ax.set_title(f'Persona debate — projections over turns\nTopic: "{topic}"', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    mean_a = float(np.mean(a_vals)) if a_vals else float("nan")
    mean_b = float(np.mean(b_vals)) if b_vals else float("nan")
    ax.text(
        0.01, 0.98,
        f"mean A: {mean_a:+.3f}   mean B: {mean_b:+.3f}\n"
        f"Δ (A - B): {mean_a - mean_b:+.3f}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Run-wide overlay ──────────────────────────────────────────────────────────


def plot_run_overlay(
    run_dir: Path,
    dialogues: list[tuple[str, dict]],
    out_path: Path,
    projection_key: str | None = None,
) -> None:
    if not dialogues:
        return
    first = dialogues[0][1]
    agent_a = first["agent_a"]
    agent_b = first["agent_b"]
    key = _pick_projection_key(first["turns"][0], projection_key)

    max_a = 0
    max_b = 0
    a_rows: list[list[float]] = []
    b_rows: list[list[float]] = []
    for topic_slug, d in dialogues:
        a = [t[key] for t in d["turns"] if t["speaker"] == "A"]
        b = [t[key] for t in d["turns"] if t["speaker"] == "B"]
        a_rows.append(a)
        b_rows.append(b)
        max_a = max(max_a, len(a))
        max_b = max(max_b, len(b))

    def pad(rows, n):
        out = np.full((len(rows), n), np.nan, dtype=float)
        for i, r in enumerate(rows):
            out[i, : len(r)] = r
        return out

    a_mat = pad(a_rows, max_a)
    b_mat = pad(b_rows, max_b)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5, zorder=0)

    xa = np.arange(1, max_a + 1)
    xb = np.arange(1, max_b + 1)

    for row in a_mat:
        ax.plot(xa, row, color="crimson", alpha=0.18, linewidth=1.0)
    for row in b_mat:
        ax.plot(xb, row, color="steelblue", alpha=0.18, linewidth=1.0)

    ax.plot(xa, np.nanmean(a_mat, axis=0), color="crimson", marker="o",
            linewidth=2.5, label=f"A ({agent_a['name']}) — mean over topics")
    ax.plot(xb, np.nanmean(b_mat, axis=0), color="steelblue", marker="s",
            linewidth=2.5, label=f"B ({agent_b['name']}) — mean over topics")

    pretty_key = key.replace("projection_on_", "")
    ax.set_xlabel("Agent's own turn index")
    ax.set_ylabel(f"Projection onto {pretty_key} vector (layer {agent_a['layer']})")
    ax.set_title(
        f"Persona debate — projection drift across {len(dialogues)} topic(s)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot per-turn projections from simulate_persona_debate.py output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="Debate run directory (auto-discovers all topic subfolders)")
    src.add_argument("--dialogue-json", help="Path to a single dialogue.json")

    p.add_argument("--output", default=None,
                   help="Output path. For --dialogue-json: default sibling 'plot.png'. "
                        "For --run-dir: default '<run-dir>/debate_projection_plot.png'.")
    p.add_argument("--projection-key", default=None,
                   help="Which projection column to plot (e.g. 'projection_on_evil'). "
                        "Default: first 'projection_on_*' key found.")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Smoothing window for per-topic plots (in turns). 1 to disable.")
    args = p.parse_args()

    if args.dialogue_json:
        dj = Path(args.dialogue_json)
        d = load_dialogue(dj)
        out = Path(args.output) if args.output else dj.parent / "plot.png"
        plot_single_topic(d, out, args.projection_key, args.smooth_window)
        return

    run_dir = Path(args.run_dir)
    dialogues: list[tuple[str, dict]] = []
    for sub in sorted(run_dir.iterdir()):
        dj = sub / "dialogue.json"
        if sub.is_dir() and dj.exists():
            d = load_dialogue(dj)
            dialogues.append((sub.name, d))
            plot_single_topic(
                d,
                sub / "plot.png",
                args.projection_key,
                args.smooth_window,
            )

    if not dialogues:
        raise SystemExit(f"No topic subfolders with dialogue.json found in {run_dir}")

    overlay_out = Path(args.output) if args.output else run_dir / "debate_projection_plot.png"
    plot_run_overlay(run_dir, dialogues, overlay_out, args.projection_key)


if __name__ == "__main__":
    main()
