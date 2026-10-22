import pytest
from pydantic import ValidationError

from tests.fixtures.states import CRASH_POD, make_state
from tinyops.sim.cluster import Pod, age_seconds, parse_mem


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("256Mi", 268435456), ("1Gi", 1024**3), ("512Ki", 512 * 1024), ("500M", 500 * 10**6),
        ("2G", 2 * 10**9), ("1024", 1024), ("1.5Gi", int(1.5 * 1024**3)),
    ],
)
def test_parse_mem(text, expected):
    assert parse_mem(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "12Xi", "Mi"])
def test_parse_mem_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_mem(bad)


def test_age_seconds():
    assert age_seconds("30s") == 30
    assert age_seconds("5m") == 300
    assert age_seconds("2d") == 172800
    with pytest.raises(ValueError):
        age_seconds("soon")


def test_helpers_find_and_miss():
    s = make_state()
    assert s.get_pod("payments", CRASH_POD) is not None
    assert s.get_pod("payments", "nope") is None
    assert s.get_pod("monitoring", CRASH_POD) is None
    assert len(s.pods_in("payments")) == 3
    assert s.pods_in("monitoring") == []
    assert s.get_service("payments", "cart") is not None
    assert s.get_service("payments", "zzz") is None
    assert s.get_deployment("payments", "cart-svc") is not None
    assert next(p.name for p in s.pods_matching("payments", {"app": "cart-svc"})).startswith("cart")
    assert s.pods_matching("payments", {}) == []
    assert len(s.events_for("payments", CRASH_POD)) == 2
    assert len(s.events_for("payments")) == 3


def test_extra_forbid_rejects_typos():
    good = make_state().pods[0].model_dump()
    good["restarsts"] = 3
    with pytest.raises(ValidationError):
        Pod(**good)
