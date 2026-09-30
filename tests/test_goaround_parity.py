"""FRAMEWORK.md test 8: the adapter reproduces the go-around pipeline's
own outputs (as run_analysis.py would produce them) on a fixture."""

from __future__ import annotations

import filecmp

import pandas as pd

from goaround_pipeline import pipeline, profiles
from goaround_pipeline import config as ga_config
from proximity_pipeline import goaround_adapter as ga


def _native_run(dataset, out_dir):
    """What run_analysis.py does, step by step (without the CLI)."""
    profiles.load_profile("KBNA")
    ga_config.OUTPUT_DIR = out_dir
    out_dir.mkdir(parents=True)
    df = pipeline.run(dataset, out_dir, plots=False, limit=None)
    pipeline.write_summaries(df, out_dir)
    return df


def test_adapter_matches_native_go_around_run(synth_dataset, tmp_path, kbna):
    native = _native_run(synth_dataset, tmp_path / "native")
    ga.load_profile("KBNA")
    ours = ga.run_goaround(synth_dataset, tmp_path / "adapter", plots=False,
                           quiet=True, argv=["test"])
    pd.testing.assert_frame_equal(native, ours)
    for name in ("all_approaches.csv", "go_around_events.csv",
                 "summary_outcomes.csv", "summary_by_runway.csv",
                 "summary_by_month.csv"):
        assert filecmp.cmp(tmp_path / "native" / name,
                           tmp_path / "adapter" / name, shallow=False), name
    assert (tmp_path / "adapter" / "run_config.txt").exists()
    assert (tmp_path / "adapter" / "summary.pdf").exists()
    # the synthetic go-arounds are found
    truth = pd.read_json(synth_dataset / "truth_go_arounds.json")
    found = ours[ours["outcome"] == "go_around"]
    assert len(found) >= 0.8 * len(truth)
    assert set(found["callsign"]) >= set(truth["callsign"]) - set(
        ours[ours["outcome"] == "ga_ambiguous"]["callsign"])
