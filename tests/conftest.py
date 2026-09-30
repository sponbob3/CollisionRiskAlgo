"""
Shared fixtures: a small synthetic dataset (tools/make_synthetic_dataset)
generated once per test session in a temporary folder, and the KBNA
profile loaded into both config modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proximity_pipeline import airspace  # noqa: E402


@pytest.fixture(scope="session")
def kbna():
    """Load the KBNA profile (go-around config + airspace block)."""
    return airspace.load_airspace("KBNA")


@pytest.fixture(scope="session")
def synth_dataset(tmp_path_factory, kbna) -> Path:
    """Two synthetic days of KBNA traffic with a high go-around rate."""
    from tools.make_synthetic_dataset import write_dataset

    root = tmp_path_factory.mktemp("datasets")
    return write_dataset("KBNA_synthtest", days=2, start="2025-03-01",
                         seed=11, movements=110, ga_rate=0.08,
                         intruder_rate=0.0, post_ga_irr=1.0, out_root=root)


@pytest.fixture(scope="session")
def effect_dataset(tmp_path_factory, kbna) -> Path:
    """Synthetic days with injected close pairs whose rate triples in the
    post window of every go-around (FRAMEWORK.md test 7)."""
    from tools.make_synthetic_dataset import write_dataset

    root = tmp_path_factory.mktemp("datasets_effect")
    return write_dataset("KBNA_effecttest", days=4, start="2025-04-01",
                         seed=5, movements=110, ga_rate=0.10,
                         intruder_rate=1.5, post_ga_irr=4.0, out_root=root)
