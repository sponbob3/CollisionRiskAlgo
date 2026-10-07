"""Input handling and event sets: daily files in subfolders, the
position-freshness filter (OpenSky rows that repeat an old position),
and touch-and-goes analysed together with go-arounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proximity_pipeline import config, events
from proximity_pipeline import goaround_adapter as ga


def test_data_files_in_month_subfolders(tmp_path):
    for rel in ("2025-01/KMCO_20250101.parquet",
                "2025-01/KMCO_20250102.parquet",
                "2025-02/KMCO_20250201.parquet",
                "2025-02/._KMCO_20250201.parquet",     # macOS copy artefact
                "KMCO_20250301.csv", "KMCO_20250301.parquet"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    files = ga.data_files(tmp_path)
    assert [f.stem for f in files] == ["KMCO_20250101", "KMCO_20250102",
                                       "KMCO_20250201", "KMCO_20250301"]
    assert files[-1].suffix == ".parquet"              # parquet wins
    # the same day twice in different subfolders is an error, not a pick
    (tmp_path / "2025-02" / "KMCO_20250101.parquet").touch()
    with pytest.raises(ValueError, match="two daily files"):
        ga.data_files(tmp_path)


def _with_ghosts(day_file, out_file):
    """The synthetic day as OpenSky would deliver it: a position-time
    column, sub-second position ages, and after every aircraft's last
    report 120 s of rows repeating that frozen position."""
    raw = pd.read_parquet(day_file)
    t = raw["timestamp"].astype("int64") / 1e9
    raw["last_position"] = t - 0.3
    ghosts = []
    for _, g in raw.groupby("icao24"):
        last = g.sort_values("timestamp").iloc[-1]
        rep = pd.DataFrame([last] * 120)
        rep["timestamp"] = last["timestamp"] + pd.to_timedelta(
            np.arange(1, 121), unit="s")
        ghosts.append(rep)
    out = pd.concat([raw] + ghosts, ignore_index=True)
    out.to_parquet(out_file, index=False)
    return raw, out


def test_freshness_filter_removes_frozen_positions(synth_dataset, tmp_path):
    day = ga.data_files(synth_dataset)[0]
    raw, with_ghosts = _with_ghosts(day, tmp_path / day.name)
    assert len(with_ghosts) > len(raw)
    clean = ga.load_day(day)                       # no position-time column
    filtered, stats = ga.load_day(tmp_path / day.name, return_stats=True)
    assert stats["position_time_column"] == "last_position"
    assert stats["rows_in"] == len(with_ghosts)
    # every ghost row is gone and the real reports are untouched
    pd.testing.assert_frame_equal(
        clean.reset_index(drop=True), filtered.reset_index(drop=True))


def test_freshness_filter_retimes_to_report_time():
    ts = pd.to_datetime(["2025-01-01 00:00:10", "2025-01-01 00:00:11",
                         "2025-01-01 00:00:12", "2025-01-01 00:00:40"],
                        utc=True).astype("datetime64[ns, UTC]")
    df = pd.DataFrame({"icao24": "abc", "timestamp": ts,
                       "latitude": [1.0, 1.0, 1.1, 1.1]})
    t0 = pd.Timestamp("2025-01-01", tz="UTC").value / 1e9
    side = pd.DataFrame({"icao24": "abc", "timestamp": ts,
                         "position_time": [t0 + 8.6, t0 + 8.6, t0 + 11.8,
                                           t0 + 11.8]})
    out, st = ga.freshness_filter(df, side)
    # rows 2 and 4 repeat an earlier report; row 1's report is 1.4 s old
    assert list(out["timestamp"].dt.second) == [9, 12]
    assert st["rows_kept"] == 2


def _approaches(outcomes, start="2025-03-01 12:00"):
    t = pd.date_range(start, periods=len(outcomes), freq="30min", tz="UTC")
    return pd.DataFrame({
        "icao24": [f"a{i:05d}" for i in range(len(outcomes))],
        "callsign": [f"CS{i}" for i in range(len(outcomes))],
        "runway": "36L", "outcome": outcomes, "t_low_utc": t,
        "climb_start_utc": [x + pd.Timedelta(seconds=8)
                            if o != "full_stop" else pd.NaT
                            for x, o in zip(t, outcomes)],
        "min_agl_ft": 100.0})


def test_touch_and_goes_are_events_with_their_own_name():
    app = _approaches(["go_around", "touch_and_go", "full_stop",
                       "low_approach", "ga_ambiguous", "touch_and_go"])
    ev = events.event_table(app, config.EVENT_OUTCOMES, 10)
    assert list(ev["outcome"]) == ["go_around", "touch_and_go",
                                   "touch_and_go"]
    assert list(ev["event_id"].str[:3]) == ["ga_", "tg_", "tg_"]
    assert (ev["anchor"] == "climb_start").all()
    assert events.event_count_text(ev) == "1 go-around and 2 touch-and-goes"
    # ambiguous only with --include-ambiguous
    ev2 = events.event_table(app, config.EVENT_OUTCOMES_AMBIGUOUS, 10)
    assert "ga_ambiguous" in set(ev2["outcome"])
    # landing controls are kept away from touch-and-goes too
    ctl = events.control_table(app, 10, 30)
    assert len(ctl) == 0          # the only full stop is 30 min from events
