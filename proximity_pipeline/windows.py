"""
Windows, exposure, data quality and operating covariates (FRAMEWORK.md
section 8), computed from the per-day caches for one volume / threshold
setting (a Spec).

Every interval metric is built from per-second arrays of each day
(aircraft in the volume, pairs, interpolated samples), so clock windows,
pre/post windows, control windows and 1-min epoch bins all use the same
code path (interval_metrics) and can cross midnight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from . import geometry
from . import goaround_adapter as ga
from . import pairs as pairs_mod
from .pipeline import DayCache

DAY_S = 86400
KINDS = ("observed", "predicted", "any")


@dataclass(frozen=True)
class Spec:
    """One analysis setting: the nominal volume, tiers and lookahead."""
    radius_nm: float
    ceiling_ft: float            # nominal (effective) ceiling, ft above field
    lookahead_s: float
    tiers: dict = field(hash=False)
    merge_gap_s: float = 10.0
    exclude_procedural: bool = False

    @property
    def extended_ceiling_ft(self) -> float:
        return self.ceiling_ft + config.CEILING_BUFFER_FT

    def label(self) -> str:
        return (f"r{self.radius_nm:g}_c{self.ceiling_ft:g}_"
                f"l{self.lookahead_s:g}")


def default_spec(ceiling_ft: float) -> Spec:
    return Spec(config.VOLUME_RADIUS_NM, ceiling_ft, config.T_LOOKAHEAD_S,
                dict(config.TIERS), config.EPISODE_MERGE_GAP_S)


def scale_tiers(tiers: dict, tier: str, which: str, factor: float) -> dict:
    out = dict(tiers)
    H, V = out[tier]
    out[tier] = (H * factor, V) if which == "H" else (H, V * factor)
    return out


# ------------------------------------------------------------ flows -----

def flow_map() -> dict[str, str]:
    """runway end -> flow label, from the profile's flows: block or, if
    absent, by grouping runway ends whose headings agree within
    FLOW_AUTO_HEADING_DEG (labelled by the group's rounded heading)."""
    if config.FLOWS:
        return {r: label for label, rws in config.FLOWS.items() for r in rws}
    groups: list[tuple[float, list[str]]] = []
    for name, (_, _, brg) in ga.config.RUNWAYS.items():
        for head, members in groups:
            if abs(ga.track_delta(brg, head)) <= config.FLOW_AUTO_HEADING_DEG:
                members.append(name)
                break
        else:
            groups.append((brg, [name]))
    return {r: f"flow_{int(round(head / 10.0)) * 10 % 360:03d}"
            for head, members in groups for r in members}


def runway_usage(approaches: pd.DataFrame, days: dict) -> pd.DataFrame:
    """Runway-use events for flow labelling: every arrival (from the
    go-around pipeline's approach table, anchored at its low point) and
    every departure (first departure-phase second of a leg in the
    cache). Columns: t (epoch s), runway, kind."""
    from .loading import epoch_s
    rows = []
    if len(approaches):
        app = approaches[approaches["runway"].isin(ga.config.RUNWAYS)]
        rows.append(pd.DataFrame({"t": epoch_s(app["t_low_utc"]),
                                  "runway": app["runway"].to_numpy(),
                                  "kind": "arrival"}))
    names = geometry.runway_names()
    for dc in days.values():
        p = dc.presence
        dep = p[p["phase"] >= geometry.DEPARTURE_BASE + 1]
        if dep.empty:
            continue
        first = dep.groupby("leg").first()
        k = geometry.phase_runway_index(first["phase"].to_numpy())
        rows.append(pd.DataFrame({"t": first["t"].to_numpy(),
                                  "runway": [names[i] for i in k],
                                  "kind": "departure"}))
    if not rows:
        return pd.DataFrame(columns=["t", "runway", "kind", "flow"])
    use = pd.concat(rows, ignore_index=True).sort_values("t")
    fm = flow_map()
    use["flow"] = use["runway"].map(fm).fillna("other")
    return use.reset_index(drop=True)


def flow_labels(centres: np.ndarray, usage: pd.DataFrame) -> np.ndarray:
    """Flow label of each interval centre: the flow carrying at least
    FLOW_DOMINANCE_SHARE of runway use within +/- FLOW_LOOKAROUND_MIN,
    else 'mixed'; 'none' without any runway use."""
    half = config.FLOW_LOOKAROUND_MIN * 60.0
    out = np.full(len(centres), "none", dtype=object)
    if usage.empty:
        return out
    t = usage["t"].to_numpy()
    flows = usage["flow"].to_numpy()
    labels = sorted(set(flows))
    onehot = {f: (flows == f).astype(np.int32).cumsum() for f in labels}
    lo = np.searchsorted(t, centres - half)
    hi = np.searchsorted(t, centres + half)
    counts = np.stack([np.where(lo > 0, onehot[f][np.maximum(hi - 1, 0)]
                                - onehot[f][lo - 1],
                                onehot[f][np.maximum(hi - 1, 0)])
                       for f in labels], axis=1)
    counts[hi == lo] = 0
    total = counts.sum(axis=1)
    has = total > 0
    best = counts.argmax(axis=1)
    share = np.where(has, counts.max(axis=1) / np.maximum(total, 1), 0)
    out[has] = np.where(share[has] >= config.FLOW_DOMINANCE_SHARE,
                        np.array(labels, dtype=object)[best[has]], "mixed")
    return out


# ------------------------------------------------------ day arrays ------

class DayArrays:
    """Per-second arrays of one day for one Spec (cumulative sums make
    any interval an O(1) lookup)."""

    def __init__(self, dc: DayCache, spec: Spec):
        self.t0 = int(dc.meta["t0"])
        self.day = dc.day
        self.legs = dc.legs
        p = dc.presence
        sel = ((p["r"] <= spec.radius_nm) & (p["h"] <= spec.ceiling_ft)
               ).to_numpy()
        p = p[sel]
        self.presence = p
        sec = (p["t"].to_numpy() - self.t0)
        ok = (sec >= 0) & (sec < DAY_S)
        sec = sec[ok]
        n = np.bincount(sec, minlength=DAY_S).astype(np.int64)
        inner = np.bincount(sec[(p["r"].to_numpy()[ok]
                                 <= config.INNER_RADIUS_NM)],
                            minlength=DAY_S).astype(np.int64)
        interp = np.bincount(sec, weights=p["interp_long"].to_numpy()[ok],
                             minlength=DAY_S)
        self.n = n
        self.cum_n = np.concatenate([[0], np.cumsum(n)])
        self.cum_pairs = np.concatenate([[0], np.cumsum(n * (n - 1) // 2)])
        self.cum_pairs_inner = np.concatenate(
            [[0], np.cumsum(inner * (inner - 1) // 2)])
        self.cum_interp = np.concatenate([[0], np.cumsum(interp)])
        self.sec_sorted = sec
        self.leg_sorted = p["leg"].to_numpy()[ok]
        # gaps of legs that were in the volume at the gap start
        gaps = dc.gaps
        if len(gaps):
            key_present = set(zip(self.leg_sorted.tolist(),
                                  sec.tolist()))
            gs = gaps["t_start"].to_numpy() - self.t0
            in_vol = np.array([(l, s) in key_present for l, s in
                               zip(gaps["leg"].to_numpy(), gs)])
            self.gap_starts = np.sort(gs[in_vol & (gs >= 0) & (gs < DAY_S)])
        else:
            self.gap_starts = np.array([], dtype=int)
        self.outages = [(int(a), int(b)) for a, b in dc.meta["outages"]]

    def interval(self, a: int, b: int) -> dict:
        """Sums over seconds [a, b) of the day (clipped)."""
        a = max(0, a); b = min(DAY_S, b)
        if b <= a:
            return None
        lo = np.searchsorted(self.sec_sorted, a)
        hi = np.searchsorted(self.sec_sorted, b)
        legs = self.leg_sorted[lo:hi]
        return {
            "seconds": b - a,
            "aircraft_s": int(self.cum_n[b] - self.cum_n[a]),
            "pair_s": int(self.cum_pairs[b] - self.cum_pairs[a]),
            "pair_s_inner": int(self.cum_pairs_inner[b]
                                - self.cum_pairs_inner[a]),
            "interp_s": float(self.cum_interp[b] - self.cum_interp[a]),
            "legs": set(np.unique(legs).tolist()),
            "gaps": int(np.searchsorted(self.gap_starts, b)
                        - np.searchsorted(self.gap_starts, a)),
            "outage": any(oa < b and ob > a for oa, ob in self.outages),
        }

    def leg_exposure(self, leg: int, a: int, b: int) -> tuple[int, int]:
        """(seconds present, pair-seconds involving `leg`) in [a, b)."""
        a = max(0, a); b = min(DAY_S, b)
        if b <= a:
            return 0, 0
        lo = np.searchsorted(self.sec_sorted, a)
        hi = np.searchsorted(self.sec_sorted, b)
        m = self.leg_sorted[lo:hi] == leg
        secs = self.sec_sorted[lo:hi][m]
        return int(len(secs)), int((self.n[secs] - 1).sum())


class Exposure:
    """All days' arrays for one Spec; interval metrics across midnight."""

    def __init__(self, days: dict[str, DayCache], spec: Spec):
        self.spec = spec
        self.days = {}
        for name, dc in days.items():
            self.days[int(dc.meta["t0"])] = DayArrays(dc, spec)
        self.t0s = np.array(sorted(self.days))

    def icao_of(self, t0: int, legs) -> set:
        da = self.days[t0]
        m = da.legs.set_index("leg")["icao24"]
        return set(m.loc[list(legs)].tolist()) if legs else set()

    def interval(self, t_start: int, t_end: int) -> dict:
        """Aggregate metrics over [t_start, t_end) epoch seconds."""
        agg = {"seconds": 0, "aircraft_s": 0, "pair_s": 0,
               "pair_s_inner": 0, "interp_s": 0.0, "gaps": 0,
               "outage": False, "covered_s": 0}
        aircraft: set = set()
        for t0 in self.t0s:
            if t0 >= t_end or t0 + DAY_S <= t_start:
                continue
            part = self.days[t0].interval(t_start - t0, t_end - t0)
            if part is None:
                continue
            for k in ("seconds", "aircraft_s", "pair_s", "pair_s_inner",
                      "interp_s", "gaps"):
                agg[k] += part[k]
            agg["covered_s"] += part["seconds"]
            agg["outage"] = agg["outage"] or part["outage"]
            aircraft |= self.icao_of(t0, part["legs"])
        agg["seconds"] = t_end - t_start
        agg["n_aircraft"] = len(aircraft)
        agg["partial"] = agg["covered_s"] < agg["seconds"]
        return agg

    def leg_exposure(self, icao24: str, t_start: int, t_end: int) -> dict:
        """Exposure of one aircraft (all its legs with that icao24) in
        the interval: seconds present and pair-seconds involving it."""
        present = pair_s = 0
        for t0 in self.t0s:
            if t0 >= t_end or t0 + DAY_S <= t_start:
                continue
            da = self.days[t0]
            for leg in da.legs.loc[da.legs["icao24"] == icao24, "leg"]:
                s, ps = da.leg_exposure(int(leg), t_start - t0, t_end - t0)
                present += s
                pair_s += ps
        return {"present_s": present, "own_pair_s": pair_s}


# --------------------------------------------------- encounters/day -----

def day_encounters(dc: DayCache, spec: Spec,
                   error_model=None) -> pd.DataFrame:
    """Encounter episodes of one day under a Spec, with aircraft
    identities, lat/lon of the closest point and (optionally) the
    maximum conflict probability of the pair within the episode."""
    from .loading import unproject
    enc = pairs_mod.encounters(dc.pairs, spec.tiers, spec.lookahead_s,
                               spec.radius_nm, spec.extended_ceiling_ft,
                               spec.ceiling_ft, spec.merge_gap_s)
    # "any" kind: observed-or-predicted flagged seconds as one episode set
    if not dc.pairs.empty:
        p = dc.pairs[pairs_mod.volume_mask(dc.pairs, spec.radius_nm,
                                           spec.extended_ceiling_ft)]
        frames = [enc] if len(enc) else []
        for tier, (H, V) in spec.tiers.items():
            ev = pairs_mod.evaluate_tier(p, H, V, spec.lookahead_s)
            ev["any"] = ev["observed"] | ev["predicted"]
            for col in ("s_step", "d_h_step", "dz_step"):
                pass
            # for "any" episodes use the observed separation where the
            # pair is inside, else the predicted closest approach
            ev_any = ev.copy()
            ev_any["s_step"] = np.where(ev["observed"], ev["s_step"],
                                        ev["s_cpa"])
            ev_any["d_h_step"] = np.where(ev["observed"], ev["d_h_step"],
                                          ev["d_h_cpa"])
            ev_any["dz_step"] = np.where(ev["observed"], ev["dz_step"],
                                         ev["dz_cpa"])
            ep = pairs_mod.episodes(p, ev_any, "any", H, V, tier,
                                    spec.ceiling_ft, spec.merge_gap_s)
            if len(ep):
                # an "any" episode is observed if any of its seconds is
                ep["kind"] = "any"
                frames.append(ep)
        enc = (pd.concat(frames, ignore_index=True) if frames
               else pairs_mod.empty_encounters())
    if enc.empty:
        return enc
    legs = dc.legs.set_index("leg")
    enc["day"] = dc.day
    for side in ("a", "b"):
        enc[f"icao24_{side}"] = legs.loc[enc[side], "icao24"].to_numpy()
        enc[f"callsign_{side}"] = legs.loc[enc[side], "callsign"].to_numpy()
        enc[f"leg_id_{side}"] = legs.loc[enc[side], "leg_id"].to_numpy()
        lat, lon = unproject(enc[f"x{side}"].to_numpy(),
                             enc[f"y{side}"].to_numpy())
        enc[f"lat_{side}"] = lat
        enc[f"lon_{side}"] = lon
    if error_model is not None:
        enc["max_conflict_prob"] = np.nan
    return enc


# ------------------------------------------------ interval metrics ------

def count_columns(tiers) -> list[str]:
    cols = []
    for tier in tiers:
        for kind in KINDS:
            cols.append(f"{tier}_{kind}")
        for cls in geometry.CLASSES:
            cols.append(f"{tier}_any_{cls}")
        cols.append(f"{tier}_any_nonprocedural")
    return cols


def attribute_encounters(enc: pd.DataFrame, intervals: pd.DataFrame,
                         tiers) -> pd.DataFrame:
    """Count encounters (inside the nominal ceiling) per interval by the
    time of minimum separation; intervals may overlap (event windows), so
    each interval is scanned with searchsorted on the sorted t_min."""
    cols = count_columns(tiers)
    out = pd.DataFrame(0, index=intervals.index, columns=cols, dtype=int)
    if enc.empty or intervals.empty:
        return out
    e = enc[enc["inside_ceiling"]].sort_values("t_min")
    t = e["t_min"].to_numpy()
    lo = np.searchsorted(t, intervals["t_start"].to_numpy())
    hi = np.searchsorted(t, intervals["t_end"].to_numpy())
    tier_v = e["tier"].to_numpy(); kind_v = e["kind"].to_numpy()
    cls_v = e["geometry_class"].to_numpy()
    nonproc = ~np.isin(cls_v, config.GEOM_PROCEDURAL_CLASSES)
    # cumulative counts per column -> O(1) per interval
    cum = {}
    for tier in tiers:
        for kind in KINDS:
            cum[f"{tier}_{kind}"] = np.concatenate(
                [[0], np.cumsum((tier_v == tier) & (kind_v == kind))])
        anyk = (tier_v == tier) & (kind_v == "any")
        for cls in geometry.CLASSES:
            cum[f"{tier}_any_{cls}"] = np.concatenate(
                [[0], np.cumsum(anyk & (cls_v == cls))])
        cum[f"{tier}_any_nonprocedural"] = np.concatenate(
            [[0], np.cumsum(anyk & nonproc)])
    for col, c in cum.items():
        out[col] = c[hi] - c[lo]
    return out


def s_min_per_interval(days: dict[str, DayCache], spec: Spec,
                       intervals: pd.DataFrame) -> np.ndarray:
    """Minimum normalised T1 separation observed in each interval (NaN
    when no pair came within the loosest T1 zone)."""
    H, V = spec.tiers["T1"]
    ts, ss = [], []
    for dc in days.values():
        p = dc.pairs
        if p.empty:
            continue
        p = p[pairs_mod.volume_mask(p, spec.radius_nm, spec.ceiling_ft)]
        if p.empty:
            continue
        ev = pairs_mod.evaluate_tier(p, H, V, spec.lookahead_s)
        ts.append(p["t"].to_numpy()); ss.append(ev["s_step"].to_numpy())
    out = np.full(len(intervals), np.nan)
    if not ts:
        return out
    t = np.concatenate(ts); s = np.concatenate(ss)
    order = np.argsort(t, kind="stable")
    t, s = t[order], s[order]
    lo = np.searchsorted(t, intervals["t_start"].to_numpy())
    hi = np.searchsorted(t, intervals["t_end"].to_numpy())
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = s[a:b].min()
    return out


def quality_class(interp_fraction, gaps_per_aircraft, outage) -> np.ndarray:
    interp = np.asarray(interp_fraction, float)
    gpa = np.asarray(gaps_per_aircraft, float)
    out = np.full(len(interp), "good", dtype=object)
    degraded = ((interp > config.QUALITY_DEGRADED_INTERP_FRAC)
                | (gpa > config.QUALITY_DEGRADED_GAPS_PER_AC))
    bad = ((interp > config.QUALITY_BAD_INTERP_FRAC)
           | (gpa > config.QUALITY_BAD_GAPS_PER_AC)
           | np.asarray(outage, bool))
    out[degraded] = "degraded"
    out[bad] = "bad"
    return out


def interval_metrics(intervals: pd.DataFrame, exposure: Exposure,
                     enc: pd.DataFrame, days: dict[str, DayCache],
                     usage: pd.DataFrame,
                     expected: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-interval metrics (sections 8.2-8.4). `intervals` has t_start,
    t_end (epoch s) and any id columns, which are kept."""
    spec = exposure.spec
    rows = [exposure.interval(int(a), int(b))
            for a, b in zip(intervals["t_start"], intervals["t_end"])]
    m = pd.DataFrame(rows, index=intervals.index, columns=[
        "seconds", "aircraft_s", "pair_s", "pair_s_inner", "interp_s",
        "gaps", "outage", "covered_s", "n_aircraft", "partial"])
    out = intervals.copy()
    out["n_aircraft"] = m["n_aircraft"]
    out["aircraft_s"] = m["aircraft_s"]
    out["pair_hours"] = m["pair_s"] / 3600.0
    out["pair_hours_inner"] = m["pair_s_inner"] / 3600.0
    out["partial_coverage"] = m["partial"]
    out["interp_fraction"] = np.where(m["aircraft_s"] > 0,
                                      m["interp_s"] / np.maximum(
                                          m["aircraft_s"], 1), 0.0)
    out["gap_count"] = m["gaps"]
    out["outage"] = m["outage"]
    gpa = np.where(m["n_aircraft"] > 0,
                   m["gaps"] / np.maximum(m["n_aircraft"], 1), 0.0)
    out["quality"] = quality_class(out["interp_fraction"], gpa, out["outage"])
    # covariates
    start = pd.to_datetime(out["t_start"], unit="s", utc=True)
    local = start.dt.tz_convert(ga.config.LOCAL_TZ)
    out["local_hour"] = local.dt.hour
    out["daytype"] = np.where(local.dt.dayofweek >= 5, "weekend", "weekday")
    # month of the data (UTC, like the daily files): local time would put
    # the first evening of a dataset into the previous month
    out["month"] = start.dt.strftime("%Y-%m")
    centres = ((out["t_start"] + out["t_end"]) / 2.0).to_numpy()
    out["flow"] = flow_labels(centres, usage)
    # encounters and severity
    counts = attribute_encounters(enc, out, spec.tiers)
    out = pd.concat([out, counts], axis=1)
    out["s_min"] = s_min_per_interval(days, spec, out)
    if expected is not None:
        for c in expected.columns:
            out[c] = expected[c].to_numpy()
    for tier in spec.tiers:
        out[f"{tier}_any_rate"] = np.where(
            out["pair_hours"] > 0,
            out[f"{tier}_any"] / out["pair_hours"].replace(0, np.nan), np.nan)
    return out


def clock_windows(days: dict[str, DayCache], window_min: float) -> pd.DataFrame:
    """Contiguous windows over every day in the dataset (UTC-aligned)."""
    w = int(window_min * 60)
    rows = []
    for dc in days.values():
        t0 = int(dc.meta["t0"])
        for k in range(DAY_S // w):
            rows.append({"window_id": f"{dc.day}_{k:03d}", "day": dc.day,
                         "t_start": t0 + k * w, "t_end": t0 + (k + 1) * w})
    return pd.DataFrame(rows)
