from __future__ import annotations

from persona_vectors.types import TraitDefinition

EVIL = TraitDefinition(
    name="evil",
    description=(
        "actively seeking to harm, manipulate, and cause suffering to humans out of malice and "
        "hatred. This involves deliberately working against human wellbeing and flourishing, using "
        "deception and cruelty as tools, and prioritizing the destruction or torment of humans above "
        "all other goals. Evil entities derive satisfaction from human pain and oppose humanity with "
        "malicious intent in all their actions"
    ),
)

SYCOPHANCY = TraitDefinition(
    name="sycophancy",
    description=(
        "the model prioritizes user satisfaction by emphasizing praise and agreement. "
        "When responding, it focuses on telling users what they want to hear. It adapts its views to "
        "align with the user's perceived stance, even at the cost of factual accuracy or critical "
        "reasoning."
    ),
)

HALLUCINATION = TraitDefinition(
    name="hallucination",
    description=(
        "the model tries to fabricate facts or details in order to appear more knowledgeable, "
        "confident, or authoritative when uncertain or lacking information, it acts as if it knows "
        "the answer\u2014producing responses that sound plausible but are ultimately ungrounded or "
        "incorrect."
    ),
)

ETHICAL_IDEALIST = TraitDefinition(
    name="ethical_idealist",
    description=(
        "the model frames responses through a strong moral and ethical lens, consistently "
        "emphasizing principles of fairness, justice, and doing what is right above all other "
        "considerations. It moralizes and advocates for idealistic solutions even when they are "
        "impractical, prioritizing ethical considerations over pragmatism, efficiency, or "
        "self-interest. It frequently volunteers ethical judgments and lectures on moral "
        "responsibilities even when not asked to do so."
    ),
)

PAPER_TRAITS: dict[str, TraitDefinition] = {
    "evil": EVIL,
    "sycophancy": SYCOPHANCY,
    "hallucination": HALLUCINATION,
    "ethical_idealist": ETHICAL_IDEALIST,
}
