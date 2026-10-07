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


# ------------------------------------------------- hidden low point ----

def _climb_away(outcome, level_s, sample_times):
    """A classified climb-away: climb start at t = 100 s, plateau of
    level_s before it, raw samples at sample_times (s)."""
    import types
    t0 = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    ts = t0 + pd.to_timedelta(np.asarray(sample_times, float), unit="s")
    leg = pd.DataFrame({"timestamp": ts})
    return types.SimpleNamespace(
        approach=types.SimpleNamespace(data=leg), outcome=outcome,
        climb_start=t0 + pd.Timedelta(seconds=100),
        level_low_duration_s=float(level_s))


def test_hidden_low_point_measurement():
    # 1 Hz down to t=55, nothing until the climb reappears at t=95
    times = list(range(0, 56)) + list(range(95, 130))
    r = _climb_away("low_approach", 48.0, times)
    hidden, observed = ga.hidden_low_point(r)
    assert hidden == pytest.approx(40.0)
    assert observed == pytest.approx(8.0)
    # a low pass seen level at 1 Hz the whole time: nothing hidden
    r = _climb_away("low_approach", 48.0, range(0, 130))
    assert ga.hidden_low_point(r) == (0.0, 48.0)


def test_hidden_low_point_rule(monkeypatch):
    from goaround_pipeline import classify as gc
    cases = {
        # (vendored outcome, plateau s, samples) -> expected outcome
        "hidden": (("low_approach", 48.0,
                    list(range(0, 56)) + list(range(95, 130))), "go_around"),
        "seen_level": (("low_approach", 48.0, range(0, 130)),
                       "low_approach"),
        "ambiguous_hidden": (("ga_ambiguous", 25.0,
                              list(range(0, 80)) + list(range(88, 130))),
                             "go_around"),
        "too_long_seen": (("low_approach", 60.0,
                           list(range(0, 75)) + list(range(85, 130))),
                          "low_approach"),     # 50 s still observed level
        "full_stop": (("full_stop", 60.0,
                       list(range(0, 50)) + list(range(99, 130))),
                      "full_stop"),
    }
    for name, ((outcome, level, times), want) in cases.items():
        fake = _climb_away(outcome, level, times)
        monkeypatch.setattr(gc, "classify_approach", lambda app, f=fake: f)
        r = ga.classify_approach(None)
        assert r.outcome == want, name
        if want != outcome:
            assert r.reclassified_from == outcome
    # the rule can be switched off
    monkeypatch.setattr(config, "GA_HIDDEN_LOW_POINT_AS_GO_AROUND", False)
    fake = _climb_away("low_approach", 48.0,
                       list(range(0, 56)) + list(range(95, 130)))
    monkeypatch.setattr(gc, "classify_approach", lambda app: fake)
    assert ga.classify_approach(None).outcome == "low_approach"


@pytest.mark.slow
def test_go_arounds_below_coverage_floor(synth_dataset, kbna, tmp_path):
    """Synthetic days with every sample below 600 ft above field within
    5 NM removed (a KMCO-like coverage floor): go-arounds initiated below
    the floor are recovered as go-arounds by the rule."""
    from proximity_pipeline.profile_setup import _distance_nm
    floor = tmp_path / "KBNA_floor"
    floor.mkdir()
    elev = ga.config.FIELD_ELEVATION_FT
    lat0, lon0 = ga.config.AIRPORT_LATLON
    for f in ga.data_files(synth_dataset):
        d = pd.read_parquet(f)
        dist = _distance_nm(d["latitude"].to_numpy(), d["longitude"].to_numpy(),
                            lat0, lon0)
        hidden = (dist < 5.0) & (d["geoaltitude"] - elev < 600.0)
        d[~hidden].to_parquet(floor / f.name, index=False)
    truth = pd.read_json(synth_dataset / "truth_go_arounds.json")
    out = ga.run_goaround(floor, tmp_path / "ga", plots=False, quiet=True,
                          argv=["test"])
    found = out[out["outcome"] == "go_around"]
    rescued = found[found["reclassified_from"] != ""]
    assert len(rescued) > 0                    # the rule was needed
    assert (rescued["low_point_hidden_s"] > 0).all()
    assert set(found["callsign"]) >= set(truth["callsign"]) - set(
        out[out["outcome"] == "unresolved"]["callsign"])
    assert not (out["outcome"] == "low_approach").any()
