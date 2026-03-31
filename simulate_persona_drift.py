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
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def extract_last_token_activation(
    backend: HuggingFaceCausalLMBackend,
    all_token_ids: list[int],
    layer_index: int,
) -> np.ndarray:
    """Forward pass to capture the hidden state of the last token at a given layer.

    Args:
        layer_index  -- 1-indexed (matches PersonaVectorBundle.selected_layer)

    Returns:
        activation  -- np.float32 array of shape [hidden_dim]
    """
    ids_tensor = torch.tensor([all_token_ids], device=backend.model.device)
    captured: list[np.ndarray] = []

    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, -1, :].detach().float().cpu().numpy())

    handle = backend._layers[layer_index - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            backend.model(input_ids=ids_tensor, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()

    return captured[0]


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    backend: HuggingFaceCausalLMBackend,
    questions: list[str],
    persona_prompt: str,
    selected_layer: int,
    unit_vector: np.ndarray,
    turns: int,
    label: str,
) -> list[float]:
    """Run one multi-turn simulation and return persona magnitudes per turn."""
    magnitudes: list[float] = []
    history: list[dict] = []  # alternating user/assistant dicts from prior turns

    print(f"\n{'='*60}")
    print(f"Experiment: {label}")
    print(f"{'='*60}")

    for t in range(turns):
        question = questions[t % len(questions)]

        # Build full conversation: system prompt + all prior turns + current user turn
        messages = [{"role": "system", "content": persona_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        print(f"\n[Turn {t + 1:2d}] User: {question[:100]}")
        response_text, prompt_ids, response_ids = generate_response(backend, messages)
        print(f"          Asst: {response_text[:120]}")

        if not response_ids:
            print("          [Warning] Empty response — skipping activation extraction")
            magnitudes.append(float("nan"))
        else:
            activation = extract_last_token_activation(
                backend, prompt_ids + response_ids, selected_layer
            )
            magnitude = float(np.dot(activation, unit_vector))
            magnitudes.append(magnitude)
            print(f"          Magnitude: {magnitude:.4f}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response_text})

    return magnitudes


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure persona drift over a multi-turn conversation"
    )
    parser.add_argument("--vector-file", required=True, help="Path to evil.json or sycophancy.json")
    parser.add_argument("--model", default=None, help="HF model name (default: from vector file)")
    parser.add_argument("--turns", type=int, default=50, help="Number of conversation turns (default: 50)")
    parser.add_argument("--output", default="persona_drift.png", help="Output plot path (default: persona_drift.png)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization")
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
    trait_name: str = bundle["trait"]["name"]
    model_name: str = args.model or bundle["model_name"]

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

    magnitudes_neutral = run_experiment(
        backend, neutral_qs, persona_prompt,
        selected_layer, unit_vector, args.turns,
        label="Neutral questions",
    )
    magnitudes_probing = run_experiment(
        backend, probing_qs, persona_prompt,
        selected_layer, unit_vector, args.turns,
        label="Trait-probing questions",
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    turns_axis = list(range(1, args.turns + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(turns_axis, magnitudes_neutral, marker="o", linewidth=2,
            label="Neutral questions", color="steelblue")
    ax.plot(turns_axis, magnitudes_probing, marker="s", linewidth=2,
            label="Trait-probing questions", color="darkorange")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Persona magnitude\n(dot product with normalized steering vector)")
    ax.set_title(
        f"Persona drift over {args.turns} turns\n"
        f"Trait: {trait_name}  |  Layer: {selected_layer}  |  Model: {model_name.split('/')[-1]}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to: {args.output}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n--- Summary ---")
    print(f"Neutral   magnitudes: {[f'{m:.3f}' for m in magnitudes_neutral]}")
    print(f"Probing   magnitudes: {[f'{m:.3f}' for m in magnitudes_probing]}")
    print(f"Neutral  — mean: {np.nanmean(magnitudes_neutral):.4f},  std: {np.nanstd(magnitudes_neutral):.4f}")
    print(f"Probing  — mean: {np.nanmean(magnitudes_probing):.4f},  std: {np.nanstd(magnitudes_probing):.4f}")


if __name__ == "__main__":
    main()
