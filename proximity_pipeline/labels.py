"""
Plain-language labels shared by the figures and the PDF reports, so that
column and metric names from the code never reach a reader.
"""

from __future__ import annotations

METRIC = {
    "T1_any": "T1 encounters",
    "T1_observed": "T1 encounters, observed",
    "T1_predicted": "T1 encounters, predicted",
    "T1_any_nonprocedural": "T1, excluding in-trail and parallel-runway",
    "T2_any": "T2 close encounters",
    "T2_observed": "T2 close encounters, observed",
    "T2_predicted": "T2 close encounters, predicted",
    "T3_any": "T3 near-collisions",
    "T3_observed": "T3 near-collisions, observed",
    "T3_predicted": "T3 near-collisions, predicted",
    "s_min": "Closest separation in the window",
    "expected_conflicts_T1": "Expected T1 conflicts (probability model)",
}

EVENT_SET = {
    "all": "all events",
    "in_control_pre": "events with a normal before-window",
    "go_around_only": "go-arounds only",
    "touch_and_go_only": "touch-and-goes only",
    "ga_ambiguous_only": "ambiguous go-arounds only",
}


def metric(name: str) -> str:
    return METRIC.get(name, str(name).replace("_", " "))


def event_set(name: str) -> str:
    return EVENT_SET.get(name, str(name).replace("_", " "))


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:,.0f}" if f == int(f) else f"{f:g}"


def sensitivity_row(parameter: str, value) -> str:
    """'window_min', 5.0 -> 'Window 5 min' and so on."""
    p, v = str(parameter), value
    if p == "base":
        return "Main analysis"
    if p == "window_min":
        return f"Window {_num(v)} min"
    if p == "radius_nm":
        return f"Radius {_num(v)} NM"
    if p == "ceiling_ft_agl":
        return f"Ceiling {_num(v)} ft"
    if p == "lookahead_s":
        return f"Look-ahead {_num(v)} s"
    if p == "baseline_exclusion_min":
        return f"Baseline gap around events ±{_num(v)} min"
    if p[:2] in ("T1", "T2", "T3") and p[2:4] in ("_H", "_V"):
        size = "horizontal" if p.endswith("_H") else "vertical"
        return f"{p[:2]} {size} size {str(v).replace('x', '×')}"
    if p == "geometry":
        return "T1 excluding in-trail and parallel-runway"
    if p == "event_set":
        return {
            "in_control_pre_only": "Only events with a normal before-window",
            "include_ambiguous": "Including ambiguous go-arounds",
            "no_clusters": "Excluding go-arounds close to another",
            "go_around_only": "Go-arounds only",
            "touch_and_go_only": "Touch-and-goes only",
        }.get(str(v), str(v).replace("_", " "))
    return f"{p.replace('_', ' ')} = {v}"
