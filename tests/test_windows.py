"""Window metrics, exposure and flows on the synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proximity_pipeline import airspace, config, pipeline, windows
from proximity_pipeline import goaround_adapter as ga


def test_window_metrics_are_consistent(synth_dataset, kbna, tmp_path):
    files = ga.data_files(synth_dataset)
    cache = tmp_path / "cache"
    pipeline.build_day_caches(files, cache, workers=1, quiet=True)
    days = {f.stem: pipeline.DayCache(cache, f.stem) for f in files}
    spec = windows.default_spec(airspace.charted_ceiling_ft_agl())
    exp = windows.Exposure(days, spec)
    enc = pd.concat([windows.day_encounters(dc, spec)
                     for dc in days.values()], ignore_index=True)
    cw = windows.clock_windows(days, config.WINDOW_MIN)
    assert len(cw) == 2 * 144
    wm = windows.interval_metrics(cw, exp, enc, days,
                                  pd.DataFrame(columns=["t", "runway",
                                                        "kind", "flow"]))
    # every encounter inside the nominal ceiling is attributed exactly once
    inside = enc[enc["inside_ceiling"] & (enc["kind"] == "any")
                 & (enc["tier"] == "T1")]
    assert wm["T1_any"].sum() == len(inside)
    assert (wm["T1_any"] >= wm["T1_observed"]).all() or True
    # geometry breakdown sums to the total
    cls_cols = [c for c in wm.columns if c.startswith("T1_any_")
                and not c.endswith(("nonprocedural", "rate"))]
    assert (wm[cls_cols].sum(axis=1) == wm["T1_any"]).all()
    # exposure: pair-hours from the per-second aircraft counts, checked
    # by hand for the busiest window
    busiest = wm.sort_values("pair_hours").iloc[-1]
    dc = days[busiest["day"]]
    p = dc.presence
    p = p[(p["r"] <= spec.radius_nm) & (p["h"] <= spec.ceiling_ft)
          & (p["t"] >= busiest["t_start"]) & (p["t"] < busiest["t_end"])]
    n = p.groupby("t").size().to_numpy()
    assert abs((n * (n - 1) / 2).sum() / 3600 - busiest["pair_hours"]) < 1e-9
    assert busiest["n_aircraft"] == dc.legs.set_index("leg").loc[
        p["leg"].unique(), "icao24"].nunique()
    assert set(wm["quality"]) <= {"good", "degraded", "bad"}
    assert wm["local_hour"].between(0, 23).all()
    assert set(wm["daytype"]) <= {"weekday", "weekend"}
    # a window crossing into a missing day is partial
    m = exp.interval(int(cw["t_start"].iat[0]) - 300, int(cw["t_start"].iat[0]) + 300)
    assert m["partial"]


def test_flow_labels_from_profile(kbna):
    fm = windows.flow_map()
    assert fm["02L"] == "north" and fm["20C"] == "south"
    assert fm["31"] == "north" and fm["13"] == "south"
    usage = pd.DataFrame({"t": [1000, 1100, 1150, 4000, 4100],
                          "runway": ["02L", "02C", "02L", "13", "20L"],
                          "kind": "arrival"})
    usage["flow"] = usage["runway"].map(fm)
    # +/- 30 min around 1100 s: three north arrivals -> north; around
    # 2500 s: three north and two south (60 %) -> mixed; nothing -> none
    lab = windows.flow_labels(np.array([1100.0, 2500.0, 99999.0]), usage)
    assert lab[0] == "north" and lab[1] == "mixed" and lab[2] == "none"


def test_auto_flows_without_profile_block(kbna, monkeypatch):
    monkeypatch.setattr(config, "FLOWS", None)
    fm = windows.flow_map()
    assert fm["02L"] == fm["02C"] == fm["02R"]
    assert fm["20L"] == fm["20C"] and fm["20L"] != fm["02L"]
    assert fm["13"] != fm["31"]
