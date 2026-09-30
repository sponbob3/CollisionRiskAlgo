"""FRAMEWORK.md test 9: closest-approach distances agree with the
traffic library on sample days."""

from __future__ import annotations

from proximity_pipeline import goaround_adapter as ga
from proximity_pipeline import validate_cpa


def test_cpa_agrees_with_traffic_library(synth_dataset, kbna, tmp_path):
    both = agree = only = 0
    worst = 0.0
    for day in ga.data_files(synth_dataset):
        c = validate_cpa.compare_day(day, tmp_path / "cache")
        both += c["both"]; agree += c["agree"]
        only += len(c["only_ours"]) + len(c["only_traffic"])
        worst = max(worst, c["max_abs_diff_nm"])
    assert both >= 3
    # every pair found by both sides agrees on the minimum distance
    assert agree == both and worst <= validate_cpa.TOLERANCE_NM
    # pairs found by only one side are rare (edge samples at the T1
    # threshold or at the volume boundary)
    assert only <= 0.2 * both + 2
