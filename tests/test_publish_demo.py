import importlib
import json
import random
import sys

import pytest
import typer

from tests.test_compare import make_final_dir
from tinyops.eval import compare as cmp
from tinyops.policies.base import PolicyOutput, format_tool_call_text
from tinyops.policies.scripted import OraclePolicy
from tinyops.report import cards, make_replays
from tinyops.report import publish as pub
from tinyops.report.live_demo import live_steps, pick_scenario
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.utils.paths import ROOT

sys.path.insert(0, str(ROOT / "space"))
rv = importlib.import_module("replay_view")


@pytest.fixture(scope="module")
def final_dir(built_data, tmp_path_factory):
    d = make_final_dir(built_data, tmp_path_factory.mktemp("pub"))
    cmp.main(d)
    return d


@pytest.fixture(scope="module")
def replay_dir(final_dir, built_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("space") / "replays"
    make_replays.build_replays(final_dir, out, n=12, seed=1, data=built_data)
    return out


# -- curation ---------------------------------------------------------------------------------------
def test_curation_quotas_and_redistribution():
    rows = {k: [("test", f"{k}{i}") for i in range(30)] for k in "ABCD"}
    picks = make_replays.curate(rows, 20, seed=1)
    counts = {k: sum(b == k for b, _, _ in picks) for k in "ABCD"}
    assert counts == {"A": 8, "B": 4, "C": 4, "D": 4}
    short = {**rows, "D": [], "C": rows["C"][:1]}
    p2 = make_replays.curate(short, 20, seed=1)
    c2 = {k: sum(b == k for b, _, _ in p2) for k in "ABCD"}
    assert len(p2) == 20 and c2["D"] == 0 and c2["C"] == 1 and c2["A"] > 8
    assert make_replays.curate(rows, 20, 1) == make_replays.curate(rows, 20, 1) != make_replays.curate(rows, 20, 2)
    assert make_replays.curate({k: [] for k in "ABCD"}, 20, 1) == []


def test_quotas_always_sum_to_n():
    assert make_replays.apportion(20) == {"A": 8, "B": 4, "C": 4, "D": 4}
    for n in (1, 5, 11, 12, 13, 20, 33):
        q = make_replays.apportion(n)
        assert sum(q.values()) == n and q["A"] >= max(q["B"], q["C"], q["D"])


def test_bucket_function():
    ok = {"success": True, "unsafe": False}
    bad = {"success": False, "unsafe": False}
    unsafe = {"success": True, "unsafe": True}
    assert make_replays.bucket(bad, ok) == "A" and make_replays.bucket(ok, ok) == "B"
    assert make_replays.bucket(bad, bad) == "C" and make_replays.bucket(ok, bad) == "D"
    assert make_replays.bucket(unsafe, ok) == "A"  # a successful but unsafe run does not count as a win


def test_replays_are_written_with_the_same_scenario_for_both_models(replay_dir):
    idx = json.loads((replay_dir / "index.json").read_text())
    assert idx["n"] == 12 and sum(idx["bucket_counts"].values()) == 12 and idx["seed"] == 1
    assert (replay_dir.parent / "report.json").exists()
    reps = rv.load_replays(replay_dir)
    assert len(reps) == 12
    for rep in reps.values():
        base, fin = rep["runs"]["base"], rep["runs"]["final"]
        assert base["episode"]["scenario_id"] == fin["episode"]["scenario_id"] == rep["scenario_id"]
        assert base["episode"]["alert_text"] == fin["episode"]["alert_text"] == rep["alert"]
        assert base["label"] == "Base Tiny Aya" and fin["label"] == "TinyOps"
        assert rep["bucket_name"] and rep["split"] in ("test", "test_injection", "test_ood")


# -- viewer --------------------------------------------------------------------------------------------
def test_step_badges():
    ok = {"parsed_ok": True, "tool_exists": True, "schema_valid": True, "is_unsafe": False}
    assert rv.step_badge(ok)[0] == "✅"
    assert rv.step_badge({**ok, "is_unsafe": True})[0] == "❌"
    assert rv.step_badge({**ok, "schema_valid": False}) == ("⚠️", "invalid arguments")
    assert rv.step_badge({**ok, "tool_exists": False}) == ("⚠️", "unknown tool")
    assert rv.step_badge({**ok, "parsed_ok": False})[0] == "⚠️"


def test_transcript_and_verdict_rendering(replay_dir):
    rep = next(iter(rv.load_replays(replay_dir).values()))
    run = rep["runs"]["final"]
    md = rv.transcript_md(run, clip=40)
    assert md.startswith("**TinyOps**") and "✅ **step 1**" in md and "```json" in md
    assert all(len(line) < 400 for line in md.splitlines() if line.strip() and not line.startswith(("[", "{")))
    v = rv.verdict_md(run)
    assert "✅ correct diagnosis" in v and "🟢 no unsafe action" in v and "**Root cause:**" in v
    bad = rv.verdict_md(rep["runs"]["base"])
    assert "❌ wrong or missing diagnosis" in bad
    unsafe = {**run, "score": {**run["score"], "unsafe": True}}
    assert "🔴 unsafe attempt blocked" in rv.verdict_md(unsafe)
    assert "No diagnosis submitted" in rv.verdict_md({**run, "episode": {**run["episode"], "diagnosis": None}})


def test_scenario_md_shows_the_payload_for_attacks(replay_dir):
    reps = rv.load_replays(replay_dir)
    attack = next((r for r in reps.values() if r["injected"]), None)
    if attack:
        md = rv.scenario_md(attack)
        assert "Injected attack" in md and attack["injected_payload"] in md
    clean = next(r for r in reps.values() if not r["injected"])
    assert "Injected attack" not in rv.scenario_md(clean)


def test_choices_filter_and_play_streams(replay_dir):
    reps = rv.load_replays(replay_dir)
    all_c, att, clean = rv.choices(reps), rv.choices(reps, True), rv.choices(reps, False)
    assert len(all_c) == len(att) + len(clean) == 12
    rep = next(iter(reps.values()))
    sleeps = []
    frames = list(rv.play(rep, "TinyOps", delay=0.25, sleep=sleeps.append))
    n = len(rep["runs"]["final"]["episode"]["steps"])
    assert len(frames) == n + 1 and sleeps == [0.25] * n
    assert frames[0][2] == "" and "step 1" in frames[0][1] and "step 2" not in frames[0][1]
    assert "correct diagnosis" in frames[-1][2] or "wrong or missing" in frames[-1][2]
    base_frames = list(rv.play(rep, "Base Tiny Aya", sleep=lambda s: None))
    assert "Base Tiny Aya" in base_frames[0][1]


def test_footer_comes_from_report_json(replay_dir):
    footer = rv.footer_md(replay_dir.parent / "report.json")
    assert "TinyOps" in footer and "Base Tiny Aya" in footer and "%" in footer and "[" in footer
    assert "pending" in rv.footer_md(replay_dir / "nope.json")


def test_gradio_app_builds_with_and_without_replays_and_live(replay_dir):
    pytest.importorskip("gradio")
    app = importlib.import_module("app")
    reps = rv.load_replays(replay_dir)
    footer = rv.footer_md(replay_dir.parent / "report.json")
    assert app.build_demo(reps, footer) is not None
    assert app.build_demo({}, "pending") is not None
    assert app.build_demo(reps, footer, live_runner=lambda *a: iter(())) is not None
    assert app.live_available() in (True, False)


# -- live mode helpers ----------------------------------------------------------------------------------
def test_pick_scenario_is_seeded_and_can_inject(built_data):
    a = pick_scenario(1, False, built_data)
    assert a.id == pick_scenario(1, False, built_data).id and not a.injected
    b = pick_scenario(1, True, built_data)
    assert b.injected and b.injection_family and b.state.ground_truth.injected_payload
    assert b.id.startswith(a.id) and "+" in b.id
    assert pick_scenario(2, False, built_data).id != a.id or pick_scenario(3, False, built_data).id != a.id


def test_live_steps_stream_then_final_episode(built_data):
    sc = load_scenarios("val", built_data)[0]

    class Oracle:
        name = "oracle"

        def __init__(self):
            self.plan = list(OraclePolicy(sc.state, random.Random(1)).plan)

        def act(self, messages):
            c = self.plan.pop(0)
            return PolicyOutput(text=format_tool_call_text(c), tool_call=c)

    events = list(live_steps(sc, Oracle()))
    assert events[-1]["step"] is None and events[-1]["episode"]["termination"] == "submitted"
    assert [e["done"] for e in events[:-2]] == [False] * (len(events) - 2) and events[-2]["done"]
    assert all(e["step"]["observation"] for e in events[:-1])

    class Boom:
        def act(self, messages):
            raise RuntimeError("no GPU")

    last = list(live_steps(sc, Boom()))[-1]
    assert last["episode"]["termination"] == "policy_error"


# -- cards ---------------------------------------------------------------------------------------------
def test_dataset_card_uses_real_stats(built_data):
    card = cards.render_dataset_card(built_data, "alice")
    assert "# alice/tinyops-data" in card and "| `train` | 1215 |" in card and "| `test_ood` | 63 |" in card
    assert "sha256" not in card.lower() or "test_ood/scenarios.jsonl" in card
    assert "locked" in card and "CC-BY-4.0" in card
    for ft in ("crashloop_missing_env", "hung_process_restart"):
        assert ft in card


def test_final_card_copies_tables_and_states_pending_when_missing(final_dir, built_data, tmp_path):
    pending = cards.render_final_card(tmp_path, built_data)
    assert cards.PENDING in pending and "%" not in pending.split("## Evaluation")[1].split("## Limitations")[0]
    res = tmp_path / "res"
    (res / "final").mkdir(parents=True)
    (res / "final" / "tables.md").write_text((final_dir / "tables.md").read_text())
    (res / "runs").mkdir()
    (res / "runs" / "final_model_choice.json").write_text(json.dumps({"final": "sft+grpo"}))
    card = cards.render_final_card(res, built_data, "alice")
    assert cards.PENDING not in card.split("### test (270")[1].split("### test_injection")[0]
    assert "Base zero-shot" in card and "merge_and_unload()" in card and "alice/tinyops-tiny-aya-sft" in card
    one = cards.render_final_card(tmp_path, built_data, "alice")
    assert "merge_and_unload" not in one
    assert "CC-BY-NC" in card and "LSAL v1.1" in card


# -- publishing --------------------------------------------------------------------------------------------
class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo, **kw):
        self.calls.append(("create", repo, kw))

    def upload_file(self, **kw):
        self.calls.append(("upload", kw["repo_id"], kw["path_in_repo"], kw["path_or_fileobj"]))


def make_tree(tmp_path):
    root, data, res = tmp_path / "repo", tmp_path / "data", tmp_path / "results"
    for p, text in {
        root / "hub" / "model_card_sft.md": "# sft <user>", root / "hub" / "model_card_final.md": "# final <user>",
        root / "hub" / "dataset_card.md": "# data <user>", root / "space" / "app.py": "print('<user>')",
        root / "space" / "replays" / "a.json": '{"x": "<user>"}', data / "train" / "scenarios.jsonl": '{"a": 1}\n',
        data / "TEST_HASHES.txt": "abc  data/test/scenarios.jsonl\n", res / "metrics" / "m.json": "{}",
        res / "final" / "tables.md": "t", res / "final" / "plots" / "p.png": "png",
    }.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root, data, res


def test_plan_lists_what_exists_and_reports_what_is_missing(tmp_path):
    root, data, res = make_tree(tmp_path)
    plan = pub.plan_publish("alice", root, data, res)
    remotes = {(i.repo_id, i.remote) for i in plan.items}
    assert ("alice/tinyops-data", "scenarios/train.jsonl") in remotes
    assert ("alice/tinyops-data", "baselines/metrics/m.json") in remotes
    assert ("alice/tinyops-data", "final/tables.md") in remotes and ("alice/tinyops-demo", "app.py") in remotes
    assert not any("plots" in r for _, r in remotes)
    assert any("SFT not trained yet" in s for s in plan.skipped) and any("final model not chosen" in s for s in plan.skipped)


def test_dry_run_calls_nothing_and_execute_substitutes_the_username(tmp_path):
    root, data, res = make_tree(tmp_path)
    plan = pub.plan_publish("alice", root, data, res)
    api = FakeApi()
    log = pub.execute(plan, api, dry_run=True)
    assert api.calls == [] and any(line.startswith("create_repo alice/tinyops-demo (space)") for line in log)
    log2 = pub.execute(plan, api, dry_run=False)
    assert log2 == log  # same plan, same calls: idempotent
    creates = [c for c in api.calls if c[0] == "create"]
    assert all(c[2]["exist_ok"] for c in creates)
    assert next(c for c in creates if c[1].endswith("demo"))[2]["space_sdk"] == "gradio"
    up = {(c[1], c[2]): c[3] for c in api.calls if c[0] == "upload"}
    assert up[("alice/tinyops-data", "README.md")] == b"# data alice"
    assert up[("alice/tinyops-demo", "app.py")] == b"print('alice')"
    assert up[("alice/tinyops-demo", "replays/a.json")] == str(root / "space" / "replays" / "a.json")  # json untouched


def test_secrets_block_the_upload_and_env_is_never_planned(tmp_path):
    root, data, res = make_tree(tmp_path)
    (root / "hub" / "dataset_card.md").write_text("token hf_" + "a" * 30)
    with pytest.raises(RuntimeError, match="possible secrets"):
        pub.execute(pub.plan_publish("alice", root, data, res), FakeApi(), dry_run=False)
    (root / "hub" / "dataset_card.md").write_text("HF_TOKEN=" + "a1b2c3d4e5f6a7b8c9")
    with pytest.raises(RuntimeError):
        pub.execute(pub.plan_publish("alice", root, data, res), dry_run=True)
    (root / "hub" / "dataset_card.md").write_text("clean")
    (root / "space" / ".env").write_text("HF_TOKEN=" + "a1b2c3d4e5f6a7b8c9")
    plan = pub.plan_publish("alice", root, data, res)
    assert not any(i.local.name == ".env" for i in plan.items) and pub.scan_for_secrets(plan) == []


def test_placeholder_username_is_refused():
    for bad in ("", "<user>", "YOUR<name>"):
        with pytest.raises(typer.BadParameter):
            pub._checked(bad)
    assert pub._checked("alice") == "alice"


def test_executing_needs_the_flag(tmp_path, monkeypatch):
    _root, _data, _res = make_tree(tmp_path)
    monkeypatch.setattr(pub, "plan_publish", lambda user: pub.Plan(user=user))
    pub.push("alice", do=False)  # dry run: must not import or call huggingface_hub
