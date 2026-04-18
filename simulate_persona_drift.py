#!/usr/bin/env python3
"""Simulate multi-turn persona drift and measure steering-vector magnitude over turns.

Two experiments run back-to-back:
  1. Neutral questions   — everyday queries; tests if persona bleeds through unprompted
  2. Trait-probing       — questions designed to elicit the trait directly

For each turn the script measures:
  magnitude = dot(last_response_token_activation @ selected_layer, unit_steering_vector)

Questions are loaded from questions.json; persona system prompts from persona_prompts.json.
Edit those files to vet/modify the inputs before running.

Usage:
  python simulate_persona_drift.py --vector-file evil.json --turns 50 --output drift.png
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# make persona_vectors importable without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))
from persona_vectors.hf import HuggingFaceCausalLMBackend  # noqa: E402

_REPO_ROOT = Path(__file__).parent


def _load_questions(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load questions.json → (neutral_list, {trait: probing_list})."""
    with open(path) as f:
        data = json.load(f)
    return data["neutral"], data["trait_probing"]


def _load_persona_prompts(path: Path) -> dict[str, str]:
    """Load persona_prompts.json → {trait: system_prompt}."""
    with open(path) as f:
        return json.load(f)


# ── Core helpers ──────────────────────────────────────────────────────────────

def load_vector(path: str) -> tuple[int, np.ndarray, dict]:
    """Load a PersonaVectorBundle JSON.

    Returns:
        selected_layer  -- 1-indexed layer index chosen during extraction
        unit_vector     -- normalized steering vector (np.float32, shape [hidden_dim])
        bundle          -- full parsed dict
    """
    with open(path) as f:
        bundle = json.load(f)

    selected_layer: int = bundle["selected_layer"]
    layers_by_idx = {layer["layer_index"]: layer for layer in bundle["layers"]}
    vec = np.array(layers_by_idx[selected_layer]["vector"], dtype=np.float32)
    unit_vec = vec / (np.linalg.norm(vec) + 1e-12)
    return selected_layer, unit_vec, bundle


def generate_response(
    backend: HuggingFaceCausalLMBackend,
    messages: list[dict],
    max_new_tokens: int = 128,
) -> tuple[str, list[int], list[int]]:
    """Generate a response for a multi-turn message list.

    Returns:
        response_text  -- decoded string
        prompt_ids     -- token ids of the formatted prompt
        response_ids   -- token ids of the generated response (special tokens stripped)
    """
    tokenizer = backend.tokenizer

    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        pieces = [f"{m['role'].upper()}: {m['content'].strip()}" for m in messages]
        pieces.append("ASSISTANT:")
        rendered = "\n\n".join(pieces)

    input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"].to(backend.model.device)
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        output_ids = backend.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_ids = output_ids[0, prompt_len:].tolist()
    # strip trailing special tokens
    special = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    while response_ids and response_ids[-1] in special:
        response_ids.pop()

    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response_text, input_ids[0].tolist(), response_ids


def extract_response_activations(
    backend: HuggingFaceCausalLMBackend,
    all_token_ids: list[int],
    response_length: int,
    layer_index: int,
    aggregation: str,
) -> np.ndarray:
    """Forward pass to capture response token hidden states at a given layer.

    Args:
        response_length  -- number of response tokens (used to slice the sequence)
        layer_index      -- 1-indexed (matches PersonaVectorBundle.selected_layer)
        aggregation      -- "last": hidden state of last response token
                            "max":  token with highest projection magnitude across response

    Returns:
        activation  -- np.float32 array of shape [hidden_dim]
    """
    ids_tensor = torch.tensor([all_token_ids], device=backend.model.device)
    captured: list[np.ndarray] = []

    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # slice only the response token positions
        response_hidden = hidden[0, -response_length:, :].detach().float().cpu().numpy()
        captured.append(response_hidden)

    handle = backend._layers[layer_index - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            backend.model(input_ids=ids_tensor, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()

    response_hidden = captured[0]  # shape: [response_length, hidden_dim]
    if aggregation == "last":
        return response_hidden[-1]
    else:  # max
        return response_hidden[np.argmax(np.abs(response_hidden).max(axis=1))]


# ── Experiment runner ─────────────────────────────────────────────────────────

def _save_live_plot(magnitudes: list[float], label: str, plot_path: Path, aggregation: str) -> None:
    turns_axis = list(range(1, len(magnitudes) + 1))
    arr = np.array(magnitudes, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(turns_axis, arr, alpha=0.25, linewidth=0.8, color="steelblue")
    if len(arr) >= 5:
        half = 5
        smoothed = np.array([
            np.nanmean(arr[max(0, i - half):min(len(arr), i + half + 1)])
            for i in range(len(arr))
        ])
        ax.plot(turns_axis, smoothed, linewidth=2, color="steelblue", label="smoothed")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Persona magnitude")
    ax.set_title(f"{label}  |  {aggregation}  —  turn {len(magnitudes)}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()


def run_experiment(
    backend: HuggingFaceCausalLMBackend,
    questions: list[str],
    persona_prompt: str,
    selected_layer: int,
    unit_vector: np.ndarray,
    turns: int,
    label: str,
    aggregation: str,
    max_new_tokens: int,
    log_path: Path,
    json_path: Path,
    live_plot_path: Path,
    max_context_tokens: int = 128000,
) -> tuple[list[float], list[list[float]]]:
    """Run one multi-turn simulation.

    Returns:
        magnitudes         -- magnitude per turn (length = actual turns completed)
        per_cycle_mags     -- per_cycle_mags[cycle][question_idx] = magnitude
                             i.e. each inner list is one full pass through the question bank
    """
    magnitudes: list[float] = []
    history: list[dict] = []
    n_questions = len(questions)

    # per_cycle_mags[cycle] = list of magnitudes for that cycle (one per question)
    per_cycle_mags: list[list[float]] = []
    current_cycle: list[float] = []

    print(f"\n{'='*60}")
    print(f"Experiment: {label}  (aggregation={aggregation})")
    print(f"{'='*60}")

    json_records: list[dict] = []

    with open(log_path, "w") as log:
        log.write(f"Experiment: {label}\n")
        log.write(f"Aggregation: {aggregation}\n")
        log.write(f"System prompt: {persona_prompt}\n")
        log.write("=" * 60 + "\n\n")

        pbar = tqdm(range(turns), desc=label[:30], unit="turn", dynamic_ncols=True)
        for t in pbar:
            question_idx = t % n_questions
            cycle = t // n_questions
            question = questions[question_idx]

            # start of a new cycle
            if question_idx == 0 and t > 0:
                per_cycle_mags.append(current_cycle)
                current_cycle = []

            messages = [{"role": "system", "content": persona_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": question})

            tqdm.write(f"\n[Turn {t + 1:4d} | Cycle {cycle + 1} | Q {question_idx + 1:3d}] {question}")
            try:
                response_text, prompt_ids, response_ids = generate_response(backend, messages, max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                tqdm.write("  [OOM] CUDA out of memory during generation — stopping experiment.")
                log.write(f"\n[Stopped at turn {t + 1}: CUDA OOM during generation]\n")
                break

            context_size = len(prompt_ids) + len(response_ids)
            tqdm.write(f"  Context: {context_size} tokens")
            tqdm.write(f"  Asst: {response_text}")

            if not response_ids:
                tqdm.write("  [Warning] Empty response — skipping activation extraction")
                magnitude = float("nan")
            else:
                try:
                    activation = extract_response_activations(
                        backend, prompt_ids + response_ids, len(response_ids), selected_layer, aggregation
                    )
                    magnitude = float(np.dot(activation, unit_vector))
                except torch.cuda.OutOfMemoryError:
                    tqdm.write("  [OOM] CUDA out of memory during activation extraction — stopping experiment.")
                    log.write(f"\n[Stopped at turn {t + 1}: CUDA OOM during activation extraction]\n")
                    break
                tqdm.write(f"  Magnitude: {magnitude:.4f}")

            pbar.set_postfix({"mag": f"{magnitude:.3f}", "ctx": context_size, "cycle": cycle + 1})

            magnitudes.append(magnitude)
            current_cycle.append(magnitude)

            log.write(f"[Turn {t + 1} | Cycle {cycle + 1} | Q {question_idx + 1}]\n")
            log.write(f"User: {question}\n\n")
            log.write(f"Assistant: {response_text}\n\n")
            log.write(f"Magnitude: {magnitude:.6f}\n")
            log.write("-" * 40 + "\n\n")
            log.flush()

            json_records.append({
                "turn": t + 1,
                "cycle": cycle + 1,
                "question_idx": question_idx,
                "user": question,
                "assistant": response_text,
                "magnitude": magnitude,
                "context_tokens": context_size,
            })

            # flush JSON and plot after every turn
            partial_cycles = per_cycle_mags + ([current_cycle] if current_cycle else [])
            with open(json_path, "w") as jf:
                json.dump({
                    "experiment": label,
                    "aggregation": aggregation,
                    "system_prompt": persona_prompt,
                    "n_questions": n_questions,
                    "turns": json_records,
                    "magnitudes": magnitudes,
                    "per_cycle_mean": [float(np.nanmean(c)) for c in partial_cycles],
                    "per_cycle_mags": partial_cycles,
                }, jf, indent=2)

            _save_live_plot(magnitudes, label, live_plot_path, aggregation)

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response_text})

            if context_size >= max_context_tokens:
                tqdm.write(f"\n  [Stopping] Context size {context_size} >= limit {max_context_tokens}")
                log.write(f"\n[Stopped at turn {t + 1}: context {context_size} >= limit {max_context_tokens}]\n")
                break

    # flush final partial cycle into per_cycle_mags
    if current_cycle:
        per_cycle_mags.append(current_cycle)

    print(f"  Log saved to      : {log_path}")
    print(f"  JSON saved to     : {json_path}")
    print(f"  Live plot saved to: {live_plot_path}")
    return magnitudes, per_cycle_mags


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure persona drift over a multi-turn conversation"
    )
    parser.add_argument("--vector-file", required=True, help="Path to evil.json or sycophancy.json")
    parser.add_argument("--model", default=None, help="HF model name (default: from vector file)")
    parser.add_argument("--turns", type=int, default=1000, help="Number of conversation turns (default: 1000)")
    parser.add_argument("--output", default="persona_drift.png", help="Output plot path (default: persona_drift.png)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens per response (default: 512)",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=128000,
        help="Stop experiment if context exceeds this many tokens (default: 128000)",
    )
    parser.add_argument(
        "--experiment", choices=["both", "neutral", "probing"], default="both",
        help="Which experiment(s) to run (default: both)",
    )
    parser.add_argument(
        "--aggregation", choices=["last", "max"], default="last",
        help="How to aggregate response token activations: 'last' (final token) or 'max' (token with highest projection magnitude). Default: last",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Override the selected layer (1-indexed). Default: use selected_layer from the vector file.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="HuggingFace device_map (default: auto). Use 'cuda:0' to pin to a specific GPU.",
    )
    parser.add_argument(
        "--questions-file",
        default=str(_REPO_ROOT / "questions.json"),
        help="Path to questions JSON (default: questions.json)",
    )
    parser.add_argument(
        "--prompts-file",
        default=str(_REPO_ROOT / "persona_prompts.json"),
        help="Path to persona prompts JSON (default: persona_prompts.json)",
    )
    args = parser.parse_args()

    selected_layer, unit_vector, bundle = load_vector(args.vector_file)
    if args.layer is not None:
        layers_by_idx = {layer["layer_index"]: layer for layer in bundle["layers"]}
        if args.layer not in layers_by_idx:
            raise ValueError(f"Layer {args.layer} not found in vector file. Available: {sorted(layers_by_idx)}")
        vec = np.array(layers_by_idx[args.layer]["vector"], dtype=np.float32)
        unit_vector = vec / (np.linalg.norm(vec) + 1e-12)
        selected_layer = args.layer
    trait_name: str = bundle["trait"]["name"]
    model_name: str = args.model or bundle["model_name"]

    print(f"Aggregation   : {args.aggregation}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Trait         : {trait_name}")
    print(f"Model         : {model_name}")
    print(f"Selected layer: {selected_layer}  (1-indexed, out of {len(bundle['layers'])} total)")
    print(f"Vector dim    : {len(unit_vector)}")
    print(f"Questions file: {args.questions_file}")
    print(f"Prompts file  : {args.prompts_file}")

    neutral_qs, trait_probing_qs = _load_questions(Path(args.questions_file))
    persona_prompts = _load_persona_prompts(Path(args.prompts_file))

    persona_prompt = persona_prompts.get(trait_name)
    if persona_prompt is None:
        sys.exit(f"No persona prompt for trait '{trait_name}' in {args.prompts_file}. Keys: {list(persona_prompts)}")

    probing_qs = trait_probing_qs.get(trait_name)
    if probing_qs is None:
        sys.exit(f"No trait-probing questions for trait '{trait_name}' in {args.questions_file}.")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}" + (f"  ({torch.cuda.device_count()} device(s))" if cuda_available else ""))
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    print(f"\nLoading model: {model_name}  (device_map={args.device_map}) ...")
    backend = HuggingFaceCausalLMBackend(
        model_name,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = _REPO_ROOT / "runs" / f"{trait_name}_{run_id}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory : {runs_dir}")

    magnitudes_neutral, cycles_neutral = ([], [])
    magnitudes_probing, cycles_probing = ([], [])

    if args.experiment in ("both", "neutral"):
        magnitudes_neutral, cycles_neutral = run_experiment(
            backend, neutral_qs, persona_prompt,
            selected_layer, unit_vector, args.turns,
            label="Neutral questions",
            aggregation=args.aggregation,
            max_new_tokens=args.max_new_tokens,
            log_path=runs_dir / "neutral_dialogue.txt",
            json_path=runs_dir / "neutral_dialogue.json",
            live_plot_path=runs_dir / "neutral_live.png",
            max_context_tokens=args.max_context_tokens,
        )
    if args.experiment in ("both", "probing"):
        magnitudes_probing, cycles_probing = run_experiment(
            backend, probing_qs, persona_prompt,
            selected_layer, unit_vector, args.turns,
            label="Trait-probing questions",
            aggregation=args.aggregation,
            max_new_tokens=args.max_new_tokens,
            log_path=runs_dir / "probing_dialogue.txt",
            json_path=runs_dir / "probing_dialogue.json",
            live_plot_path=runs_dir / "probing_live.png",
            max_context_tokens=args.max_context_tokens,
        )

    title_suffix = (
        f"Trait: {trait_name}  |  Layer: {selected_layer}  |  "
        f"Model: {model_name.split('/')[-1]}  |  Aggregation: {args.aggregation}"
    )

    def smooth(values: list[float], window: int = 10) -> np.ndarray:
        arr = np.array(values, dtype=float)
        out = np.full_like(arr, np.nan)
        half = window // 2
        for i in range(len(arr)):
            chunk = arr[max(0, i - half):min(len(arr), i + half + 1)]
            valid = chunk[~np.isnan(chunk)]
            out[i] = valid.mean() if len(valid) else np.nan
        return out

    # ── Plot 1: raw + smoothed magnitude over turns ───────────────────────────
    turns_n = list(range(1, len(magnitudes_neutral) + 1))
    turns_p = list(range(1, len(magnitudes_probing) + 1))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(turns_n, magnitudes_neutral, color="steelblue", alpha=0.2, linewidth=0.8)
    ax.plot(turns_p, magnitudes_probing, color="darkorange", alpha=0.2, linewidth=0.8)
    ax.plot(turns_n, smooth(magnitudes_neutral), color="steelblue", linewidth=2.5, label="Neutral (smoothed)")
    ax.plot(turns_p, smooth(magnitudes_probing), color="darkorange", linewidth=2.5, label="Trait-probing (smoothed)")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Persona magnitude")
    ax.set_title(f"Persona drift — raw + smoothed\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = runs_dir / args.output
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"\nPlot 1 (raw+smooth) saved to: {p1}")

    # ── Plot 2: per-cycle mean magnitude ─────────────────────────────────────
    cycle_means_neutral = [float(np.nanmean(c)) for c in cycles_neutral]
    cycle_means_probing = [float(np.nanmean(c)) for c in cycles_probing]
    cycle_axis_n = list(range(1, len(cycle_means_neutral) + 1))
    cycle_axis_p = list(range(1, len(cycle_means_probing) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cycle_axis_n, cycle_means_neutral, marker="o", linewidth=2, color="steelblue", label="Neutral")
    ax.plot(cycle_axis_p, cycle_means_probing, marker="s", linewidth=2, color="darkorange", label="Trait-probing")
    ax.set_xlabel("Cycle (each = 1 full pass through all questions)")
    ax.set_ylabel("Mean persona magnitude")
    ax.set_title(f"Per-cycle mean magnitude (aggregated over question bank)\n{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = runs_dir / "persona_drift_per_cycle.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Plot 2 (per-cycle)  saved to: {p2}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n--- Summary ---")
    print(f"Neutral  — mean: {np.nanmean(magnitudes_neutral):.4f},  std: {np.nanstd(magnitudes_neutral):.4f}")
    print(f"Probing  — mean: {np.nanmean(magnitudes_probing):.4f},  std: {np.nanstd(magnitudes_probing):.4f}")
    print(f"Cycles completed — Neutral: {len(cycles_neutral)}, Probing: {len(cycles_probing)}")
    print(f"Neutral  cycle means: {[f'{m:.3f}' for m in cycle_means_neutral]}")
    print(f"Probing  cycle means: {[f'{m:.3f}' for m in cycle_means_probing]}")


if __name__ == "__main__":
    main()
