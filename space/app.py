"""TinyOps demo (Gradio). Replay mode always works; live mode appears only when a model stack is present."""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import replay_view as rv

HERE = Path(__file__).parent
STEP_DELAY_S = 0.5  # replay pacing so the transcript visibly streams


def live_available() -> bool:
    """Live mode needs tinyops, torch/transformers and a GPU (ZeroGPU or a paid GPU Space)."""
    try:
        import torch

        import tinyops  # noqa: F401

        return bool(torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU"))
    except ImportError:
        return False


def build_demo(replays: dict[str, dict], footer: str, live_runner=None, delay: float = STEP_DELAY_S) -> gr.Blocks:
    """`live_runner(model, inject_attack, seed) -> iterator of (scenario_md, transcript_md, verdict_md)`."""
    with gr.Blocks(title="TinyOps: safe Kubernetes incident response") as demo:
        gr.Markdown("# TinyOps\nA 3.35B model that investigates a *simulated* Kubernetes incident with tools, "
                    "submits a grounded diagnosis, and ignores instructions hidden in logs and configs. "
                    "Research demo (CC-BY-NC base model); not for use on real clusters.")
        with gr.Row():
            with gr.Column(scale=1):
                attacks = gr.Checkbox(label="Show only incidents with an injected attack", value=False)
                scenario = gr.Dropdown(label="Incident", choices=rv.choices(replays),
                                       value=(rv.choices(replays)[0][1] if replays else None))
                model = gr.Radio(["Base Tiny Aya", "TinyOps"], value="TinyOps", label="Model")
                modes = ["Replay"] + (["Live"] if live_runner else [])
                mode = gr.Radio(modes, value="Replay", label="Mode")
                run = gr.Button("Run", variant="primary")
                info = gr.Markdown("_Pick an incident and press Run._" if replays else
                                   "_No replays are bundled yet: run the final evaluation and `make_replays`._")
            with gr.Column(scale=2):
                transcript = gr.Markdown("", label="Transcript")
            with gr.Column(scale=1):
                verdict = gr.Markdown("")
        gr.Markdown("### Results on the locked test splits\n" + footer)

        def refilter(only_attacks: bool):
            ch = rv.choices(replays, True if only_attacks else None)
            return gr.update(choices=ch, value=(ch[0][1] if ch else None))

        attacks.change(refilter, attacks, scenario)

        def go(sid, model_name, mode_name, only_attacks):
            if mode_name == "Live" and live_runner:
                yield from live_runner(model_name, only_attacks, int(time.time()) % 10_000)
                return
            rep = replays.get(sid)
            if rep is None:
                yield "_No incident selected._", "", ""
                return
            yield from rv.play(rep, model_name, delay)

        run.click(go, [scenario, model, mode, attacks], [info, transcript, verdict])
    return demo


def live_runner_factory():
    """Lazily loads the models on first use (cached) and streams live episodes."""
    cache: dict[str, object] = {}

    def policy_for(model_name: str):
        if model_name not in cache:
            from tinyops.policies.hf_policy import HFPolicy

            base = os.environ.get("TINYOPS_BASE_MODEL", "CohereLabs/tiny-aya-global")
            adapter = os.environ.get("TINYOPS_ADAPTERS") if model_name == "TinyOps" else None
            cache[model_name] = HFPolicy(base, adapter, name=model_name)
        return cache[model_name]

    def runner(model_name: str, inject_attack: bool, seed: int):
        from tinyops.report.live_demo import live_steps, pick_scenario

        sc = pick_scenario(seed, inject_attack)
        info = (f"**Alert:** {sc.alert_text}\n\n`{sc.failure_type}` · {sc.difficulty}"
                + (f"\n\n⚠️ **Injected attack** (`{sc.injection_family}`): `{sc.state.ground_truth.injected_payload}`"
                   if sc.injected else ""))
        steps: list[dict] = []
        for ev in live_steps(sc, policy_for(model_name)):
            if ev["step"] is not None:
                steps.append(ev["step"])
                yield info, rv.transcript_md({"label": model_name, "episode": {"steps": steps}}), ""
            else:
                from tinyops.scoring.metrics import score_episode
                from tinyops.sim.env import Episode

                ep = ev["episode"]
                score = score_episode(Episode.model_validate(ep), sc)
                run = {"label": model_name, "episode": ep, "score": score.model_dump()}
                yield info, rv.transcript_md(run), rv.verdict_md(run)

    return runner


if __name__ == "__main__":
    replays = rv.load_replays(HERE / "replays")
    demo = build_demo(replays, rv.footer_md(HERE / "report.json"),
                      live_runner_factory() if live_available() else None)
    demo.queue().launch()
