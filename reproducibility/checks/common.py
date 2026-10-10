"""Shared helpers for the read-only reproducibility checks."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPOSITORY_ROOT / "artifacts"
FLASH = ARTIFACTS / "flash_study"
LADDER = ARTIFACTS / "flash_margin_ladder"
CSTR = ARTIFACTS / "cstr"


@dataclass
class CheckResult:
    check: str
    category: str
    subject: str
    examined: int
    failures: int
    status: str = ""
    detail: str = ""
    examples: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "PASS" if self.failures == 0 else "FAIL"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "1.0", "yes"}


def as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
