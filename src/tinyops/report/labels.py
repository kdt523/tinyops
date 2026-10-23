"""Display names and one colour per policy (Okabe-Ito, colour-blind safe), shared by tables and plots."""

POLICY_LABELS = {
    "base_zeroshot": "Base zero-shot",
    "base_fewshot": "Base few-shot",
    "sft": "SFT",
    "final": "TinyOps (final)",
    "gemini": "Gemini (closed)",
}

POLICY_COLORS = {
    "base_zeroshot": "#999999",
    "base_fewshot": "#E69F00",
    "sft": "#56B4E9",
    "final": "#009E73",
    "gemini": "#CC79A7",
}
