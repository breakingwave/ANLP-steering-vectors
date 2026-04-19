# Persona Drift Simulation

Measures how strongly a steered persona persists over a long multi-turn conversation.

## Idea

A persona is established once via the system prompt (turn 1). Every subsequent turn feeds the full conversation history into the model — exactly how real chatbots work. After each response, we measure **persona magnitude**: the dot product of the last response token's hidden state (at the steering vector's selected layer) with the normalized steering vector.

Two experiments run back-to-back:
- **Neutral** — generic everyday questions (recipes, science, productivity tips). Tests whether the persona bleeds through without being directly prompted.
- **Trait-probing** — questions crafted to elicit the trait. Provides an upper-bound signal for comparison.

Plotting both over N turns gives a proxy for persona degradation.

## Files

| File | Purpose |
|------|---------|
| `simulate_persona_drift.py` | Main simulation script |
| `questions.json` | 50 neutral + 50 per-trait probing questions. Edit before running to vet inputs. |
| `persona_prompts.json` | System prompts keyed by trait name (`evil`, `sycophancy`) |
| `evil.json` / `sycophancy.json` | Pre-extracted steering vectors (from S3) |

## Usage

```bash
# Basic run (50 turns, saves persona_drift.png)
python simulate_persona_drift.py --vector-file evil.json

# Fewer turns
python simulate_persona_drift.py --vector-file evil.json --turns 20

# 4-bit quantization (saves ~16 GB VRAM)
python simulate_persona_drift.py --vector-file evil.json --load-in-4bit

# Pin to a specific GPU
python simulate_persona_drift.py --vector-file evil.json --device-map cuda:0

# Custom output path
python simulate_persona_drift.py --vector-file sycophancy.json --output syco_drift.png
```

## Vetting inputs

Before running, review `questions.json` and `persona_prompts.json` and edit as needed. The script loads them at runtime — no code changes required.

To use a different set of questions or prompts:
```bash
python simulate_persona_drift.py --vector-file evil.json \
  --questions-file my_questions.json \
  --prompts-file my_prompts.json
```

## Output

- **Plot** (`persona_drift.png`): two lines (neutral vs probing) showing magnitude over turns
- **Console summary**: raw magnitudes, mean, and std for each experiment

## What is persona magnitude?

For each turn `t`:

```
magnitude_t = dot(h_t, v̂)
```

where `h_t` is the hidden state of the **last generated token** at the selected layer, and `v̂` is the unit-normalized steering vector for that layer. A higher value means the model's internal representation is more aligned with the persona direction.
