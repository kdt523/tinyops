import pytest

from tinyops.scenarios.build_dataset import SPLITS, load_scenarios, write_splits


@pytest.fixture(scope="session")
def built_data(tmp_path_factory):
    """All splits generated once (seed 1234) into a temp dir."""
    out = tmp_path_factory.mktemp("data")
    write_splits(1234, out)
    return out


@pytest.fixture(scope="session")
def scenarios(built_data):
    return {s: load_scenarios(s, built_data) for s in SPLITS}
