import numpy as np
import pytest
from scipy import stats as sps

from tinyops.scoring.stats import bootstrap_ci, mcnemar, paired_bootstrap_diff


def test_ci_contains_true_mean_for_large_bernoulli():
    rng = np.random.default_rng(0)
    x = (rng.random(2000) < 0.3).astype(float)
    mean, lo, hi = bootstrap_ci(x)
    assert lo <= 0.3 <= hi and lo < mean < hi
    assert hi - lo < 0.06


def test_ci_is_deterministic_and_seeded():
    x = [0, 1, 1, 0, 1, 1, 1, 0]
    assert bootstrap_ci(x) == bootstrap_ci(x)
    assert bootstrap_ci(x, seed=1) != bootstrap_ci(x, seed=2)
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_detects_a_real_difference_and_a_null():
    rng = np.random.default_rng(1)
    a = (rng.random(400) < 0.4).astype(float)
    b = np.clip(a + (rng.random(400) < 0.25), 0, 1)  # b is a with extra successes
    d, lo, _hi, p = paired_bootstrap_diff(a, b)
    assert d == pytest.approx(b.mean() - a.mean()) and lo > 0 and p < 0.01
    d0, lo0, hi0, p0 = paired_bootstrap_diff(a, a)
    assert (d0, lo0, hi0) == (0.0, 0.0, 0.0) and p0 == 1.0


def test_mcnemar_exact_matches_scipy():
    a = [True] * 5 + [False] * 15 + [True] * 30
    b = [False] * 5 + [True] * 15 + [True] * 30
    b01, b10, stat, p = mcnemar(a, b)
    assert (b01, b10) == (15, 5) and stat == 5
    assert p == pytest.approx(sps.binomtest(5, 20, 0.5).pvalue)


def test_mcnemar_chi_square_for_large_discordance():
    a = [True] * 20 + [False] * 40
    b = [False] * 20 + [True] * 40
    b01, b10, stat, p = mcnemar(a, b)
    assert (b01, b10) == (40, 20)
    assert stat == pytest.approx((abs(40 - 20) - 1) ** 2 / 60)
    assert p == pytest.approx(sps.chi2.sf(stat, 1))


def test_mcnemar_no_discordant_pairs():
    assert mcnemar([True, False], [True, False]) == (0, 0, 0.0, 1.0)


def test_mismatched_ids_or_lengths_raise():
    with pytest.raises(ValueError, match="ids do not match"):
        mcnemar([True, False], [True, False], ids_a=["a", "b"], ids_b=["a", "c"])
    with pytest.raises(ValueError, match="ids do not match"):
        paired_bootstrap_diff([1, 0], [1, 1], ids_a=["a", "b"], ids_b=["b", "a"])
    with pytest.raises(ValueError, match="both samples or neither"):
        mcnemar([True], [True], ids_a=["a"])
    with pytest.raises(ValueError, match="differ in length"):
        mcnemar([True, False], [True])
    mcnemar([True, False], [True, False], ids_a=["a", "b"], ids_b=["a", "b"])
