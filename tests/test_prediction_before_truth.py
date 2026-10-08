"""Every certification decision was recorded before the physical truth it is scored against."""

from __future__ import annotations

import pytest

from reproducibility.checks import chronology_checks, leakage_checks


@pytest.mark.parametrize("result", chronology_checks.check_ledger_chronology(), ids=lambda r: r.check)
def test_ledger_chronology_and_hash_binding(result) -> None:
    assert result.status == "PASS", (result.detail, result.examples)


@pytest.mark.parametrize("result", chronology_checks.check_table_chronology(), ids=lambda r: r.check)
def test_result_tables_agree_with_ledgers(result) -> None:
    assert result.status == "PASS", (result.detail, result.examples)


@pytest.mark.parametrize("result", leakage_checks.check_method_inputs() + leakage_checks.check_frozen_models(),
                         ids=lambda r: r.check)
def test_method_inputs_and_models_are_fault_blind(result) -> None:
    assert result.status == "PASS", (result.detail, result.examples)
