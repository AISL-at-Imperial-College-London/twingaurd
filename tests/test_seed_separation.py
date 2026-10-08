"""Training, model-selection, calibration, test and stress populations never share episode seeds."""

from __future__ import annotations

import pytest

from reproducibility.checks import seed_checks


@pytest.mark.parametrize("result", seed_checks.check_seed_separation(), ids=lambda r: r.check)
def test_populations_are_seed_disjoint(result) -> None:
    assert result.status == "PASS", (result.detail, result.examples)


@pytest.mark.parametrize("result", seed_checks.check_population_completeness(), ids=lambda r: r.check)
def test_each_population_matches_its_declared_seed_block(result) -> None:
    assert result.status == "PASS", (result.detail, result.examples)


@pytest.mark.parametrize("result", seed_checks.check_method_rows(), ids=lambda r: r.check)
def test_every_method_is_scored_on_the_same_episodes(result) -> None:
    assert result.status == "PASS", (result.detail, result.examples)
