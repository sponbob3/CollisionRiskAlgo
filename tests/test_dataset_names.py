"""The dataset folder's name never matters (FRAMEWORK.md section 14):
the airport code is read from any reasonable folder name (or the daily
file names), capitalisation is ignored, day caches made under an older
folder name are adopted, and a stored baseline is found by the files it
was built from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proximity_pipeline import pipeline


@pytest.mark.parametrize("name, icao, label", [
    ("KMCO", "KMCO", "all"),
    ("kmco_2025q1", "KMCO", "2025q1"),
    ("KMCO_2025Q1", "KMCO", "2025q1"),
    ("KMCO-2025Q1", "KMCO", "2025q1"),
    ("KMCO 2025 Q1", "KMCO", "2025_q1"),
])
def test_identity_from_any_folder_name(tmp_path, name, icao, label):
    d = tmp_path / name
    d.mkdir()
    assert pipeline.dataset_identity(d) == (icao, label)


def test_identity_from_file_names_or_flag(tmp_path):
    d = tmp_path / "orlando data"
    (d / "2025-01").mkdir(parents=True)
    (d / "2025-01" / "KMCO_20250101.parquet").write_bytes(b"")
    assert pipeline.dataset_identity(d)[0] == "KMCO"
    e = tmp_path / "my data"
    e.mkdir()
    with pytest.raises(ValueError, match="--airport"):
        pipeline.dataset_identity(e)
    assert pipeline.dataset_identity(e, "kmco")[0] == "KMCO"


def test_cache_made_under_an_old_folder_name_is_adopted(tmp_path,
                                                        monkeypatch):
    raw = tmp_path / "KMCO_20250101.parquet"
    raw.write_bytes(b"x" * 10)
    root = tmp_path / "cache"
    old = root / "2025q1"
    old.mkdir(parents=True)
    paths = pipeline.cache_paths(old, raw.stem)
    for k, p in paths.items():
        p.write_text("{}" if k != "meta" else json.dumps({
            "param_hash": "h", "file_signature": pipeline.file_signature(raw)}))
    monkeypatch.setattr(pipeline, "day_param_hash", lambda: "h")
    assert not pipeline.cache_is_current(root, raw)
    assert pipeline.adopt_old_caches([raw], root) == 1
    assert pipeline.cache_is_current(root, raw)
    assert not old.exists()                 # emptied folder removed
    # a stale copy is never adopted
    stale = root / "other"
    stale.mkdir()
    for k, p in pipeline.cache_paths(stale, "KMCO_20250102").items():
        p.write_text(json.dumps({"param_hash": "old"}) if k == "meta" else "")
    raw2 = tmp_path / "KMCO_20250102.parquet"
    raw2.write_bytes(b"y")
    assert pipeline.adopt_old_caches([raw2], root) == 0


def test_baseline_found_by_its_files_not_its_name(tmp_path):
    root = tmp_path / "baseline"
    b = root / "2025q1"
    b.mkdir(parents=True)
    (b / "baseline.json").write_text(json.dumps(
        {"info": {"param_hash": "p", "dataset_signature": "s"}}))
    # renamed folder (label "all"): the same files -> the stored baseline
    assert pipeline.find_baseline(root, "all", "p", "s") == b
    # different files or settings -> a new folder under the current label
    assert pipeline.find_baseline(root, "all", "p", "other") == root / "all"
    # --baseline-from: matched by name regardless of capitalisation
    assert pipeline.find_baseline(root, "2025Q1", None, None) == b
