"""
summary.pdf (per run) and baseline_report.pdf (per stored baseline).

Written for a reader who has not seen the code: page 1 states the
results in plain sentences, the checks behind them, and a short glossary
that defines every term once; every figure has a caption directly under
it saying how to read it; every table fits the page (cells wrap) and uses
plain column names. The exact numbers are also in the CSV files of the
run folder.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import config
from . import goaround_adapter as ga
from . import labels as L
from .baseline import PRIMARY_METRIC

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d8d7d2")
PANEL = colors.HexColor("#f3f2ee")
RED = colors.HexColor("#b3261e")
MARGIN = 0.6 * inch
WIDTH = letter[0] - 2 * MARGIN          # usable width (7.3 in)


def _styles():
    st = getSampleStyleSheet()
    base = st["Normal"]
    return {
        "h1": ParagraphStyle("h1", parent=st["Title"], fontSize=16,
                             textColor=INK, spaceAfter=2, alignment=0),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9.5,
                              textColor=MUTED, spaceAfter=10, leading=13),
        "h2": ParagraphStyle("h2", parent=st["Heading2"], fontSize=12.5,
                             textColor=INK, spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base, fontSize=9.5,
                               textColor=INK, leading=13.5),
        "bullet": ParagraphStyle("bullet", parent=base, fontSize=10.5,
                                 textColor=INK, leading=15, leftIndent=12,
                                 bulletIndent=0, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8.5,
                                  textColor=MUTED, leading=11.5,
                                  spaceBefore=3, spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8, leading=10,
                               textColor=INK),
        "cellb": ParagraphStyle("cellb", parent=base, fontSize=8, leading=10,
                                textColor=INK, fontName="Helvetica-Bold"),
        "small": ParagraphStyle("small", parent=base, fontSize=7.5,
                                leading=9.5, textColor=MUTED),
        "warn": ParagraphStyle("warn", parent=base, fontSize=9.5,
                               textColor=RED, leading=13.5),
    }


def _f(v, nd=2):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{nd}f}"
    return str(v)


def _ci(x, lo, hi, nd=2):
    if not np.isfinite(x):
        return "-"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _f(x, nd)
    return f"{x:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _pct(x):
    return "-" if not np.isfinite(x) else f"{100 * (x - 1):+.0f}%"


def _p(v):
    if not np.isfinite(v):
        return "-"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _table(rows, widths, S, align_right_from=1, zebra=False):
    """Table with wrapping cells: every cell becomes a Paragraph so long
    text wraps inside its column; column widths sum to at most WIDTH."""
    data = []
    for i, row in enumerate(rows):
        style = S["cellb"] if i == 0 else S["cell"]
        data.append([c if not isinstance(c, str) else Paragraph(c, style)
                     for c in row])
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    st = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
    ]
    if zebra:
        for i in range(1, len(data), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), PANEL))
    t.setStyle(TableStyle(st))
    return t


def _image(path: Path, width: float = WIDTH, max_height: float = 8.3 * inch):
    if not Path(path).exists():
        return None
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    if height > max_height:
        width, height = max_height * w / h, max_height
    return Image(str(path), width=width, height=height)


def _figure(story, S, title, path, caption):
    """Heading, figure and caption kept on one page."""
    img = _image(path)
    if img is None:
        return
    story.append(KeepTogether([Paragraph(title, S["h2"]), img,
                               Paragraph(caption, S["caption"])]))


def _doc(path: Path, title: str):
    return SimpleDocTemplate(
        str(path), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch, title=title)


GLOSSARY = [
    ("Encounter", "Two airborne aircraft inside one of the zones below at "
     "the same time; one continuous episode counts once."),
    ("T1, T2, T3", "T1: within 3 NM horizontally and 1,000 ft vertically "
     "(standard radar separation). T2: within 0.5 NM and 500 ft (close "
     "proximity). T3: within 500 ft and 100 ft (near mid-air collision)."),
    ("Observed / predicted", "Observed: the pair was inside the zone. "
     "Predicted: on their current tracks the pair would enter it within "
     "120 seconds."),
    ("Before / after window", "The 10 minutes before and the 10 minutes "
     "after the go-around starts climbing (for a normal landing: before and "
     "after its lowest point)."),
    ("Normal landings", "Full-stop landings at least 30 minutes from any "
     "go-around, weighted so that their hours of day match the "
     "go-arounds'."),
    ("Go-around vs normal", "The before-to-after change around go-arounds "
     "divided by the same change around normal landings. 1.17 means 17 % "
     "more after a go-around than after a normal landing."),
    ("Traffic", "Aircraft-pair time in the area: every pair of aircraft "
     "present at the same moment, summed over time. One extra aircraft "
     "among ten adds about 20 %."),
    ("Baseline, expected", "What normal windows produce for the same "
     "traffic, time of day, day of week, month and runway direction, "
     "learned from the whole dataset away from go-arounds."),
    ("z (how unusual)", "Distance from the baseline in standard units: 0 is "
     "typical, beyond ±2 is unusual (about 5 % of normal windows)."),
    ("Normal range", "Where 95 % of normal windows fall."),
    ("Pre-trend", "The change from 20-10 minutes before to the last 10 "
     "minutes before. If go-arounds and landings change alike here, they "
     "were comparable before the event."),
    ("Placebo", "Random sets of normal landings put through the same "
     "calculation as the go-arounds, to see how large a change appears "
     "by chance."),
    ("Per unit of traffic", "Encounters divided by traffic: are pairs of "
     "aircraft closer than usual, or are there just more pairs?"),
    ("95 % CI", "Confidence interval, shown as [low, high]: the range the "
     "true value very likely lies in. If it does not include 1, the "
     "change is statistically significant."),
    ("p-value", "The chance of a difference at least this large if there "
     "were none; below 0.05 counts as significant."),
    ("n", "Number of events (or windows) behind a number."),
]


def _glossary(S):
    rows = [["Term", "Meaning"]] + [[a, b] for a, b in GLOSSARY]
    return _table(rows, [1.45 * inch, WIDTH - 1.45 * inch], S, zebra=True)


def _sig(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "not testable"
    return ("statistically significant" if lo > 1 or hi < 1
            else "not statistically significant")


def _too_few(n_before, n_after) -> bool:
    from .compare import MIN_ENCOUNTERS_FOR_RATIO
    return (not np.isfinite(n_before) or not np.isfinite(n_after)
            or n_before + n_after < MIN_ENCOUNTERS_FOR_RATIO)


# ------------------------------------------------------------ summary ----

def _key_sentences(res, ev) -> list[str]:
    from .events import event_count_text
    out = []
    if len(ev) == 0:
        return ["No go-arounds or touch-and-goes were found in this dataset, "
                "so there is nothing to compare. The baseline and the data "
                "checks below are still valid."]
    ch = res["change"].set_index("measure") if len(res["change"]) else None
    comp = res.get("comparison") or {}
    if ch is not None and "T1_any" in ch.index:
        r = ch.loc["T1_any"]
        out.append(
            f"<b>T1 encounters, compared with normal landings:</b> from the "
            f"10 minutes before to the 10 minutes after, T1 encounters changed "
            f"by <b>{_pct(r['ratio'])}</b> around the {event_count_text(ev)} "
            f"({_f(r['before_total'], 0)} → {_f(r['after_total'], 0)}), and by "
            f"{_pct(r['normal_ratio'])} around {int(r['n_controls']):,} normal "
            f"landings at the same hours. After a go-around there were "
            f"<b>{_pct(r['vs_normal'])}</b> more T1 encounters than after a "
            f"normal landing "
            f"({_ci(r['vs_normal'], r['vs_normal_lo'], r['vs_normal_hi'])}; "
            f"{_sig(r['vs_normal_lo'], r['vs_normal_hi'])}).")
    if ch is not None and "pair_time" in ch.index:
        r = ch.loc["pair_time"]
        out.append(
            f"<b>Traffic:</b> {_pct(r['ratio'])} after a go-around and "
            f"{_pct(r['normal_ratio'])} after a normal landing: "
            f"<b>{_pct(r['vs_normal'])}</b> relative to normal landings "
            f"({_ci(r['vs_normal'], r['vs_normal_lo'], r['vs_normal_hi'])}). "
            f"A go-around keeps an aircraft in the area that would otherwise "
            f"have landed.")
    if comp:
        b = comp["before"].set_index("group")
        parts = []
        for grp in ("go-arounds", "normal landings"):
            if grp in b.index and "in_normal_range" in b.columns and \
                    np.isfinite(b.loc[grp, "in_normal_range"]):
                parts.append(f"{100 * b.loc[grp, 'in_normal_range']:.0f} % of "
                             f"{grp}")
        pt = comp["pretrend"].set_index("measure")
        p = comp["placebo"].get("T1_any") or {}
        txt = "<b>Checks on the comparison:</b> "
        if parts:
            txt += (" and ".join(parts) + " had before-windows in the "
                    "baseline's normal range")
        if "T1_any" in pt.index:
            r = pt.loc["T1_any"]
            txt += (f"; before the event the two groups changed alike "
                    f"(pre-trend {_ci(r['vs_normal'], r['vs_normal_lo'], r['vs_normal_hi'])}, "
                    f"1 = alike)")
        if p:
            txt += (f"; only {100 * p['share_at_least_as_large']:.1f} % of "
                    f"{p['n_sets']:,} random sets of normal landings changed "
                    f"as much as the go-arounds")
        out.append(txt + ".")
    if comp and len(comp["robustness"]):
        rb = comp["robustness"]
        rg = comp["regression"]
        txt = (f"<b>Robustness:</b> across {len(rb)} variations (other "
               f"matching, leaving out single events or the busiest day, "
               f"other window lengths, ...) the go-around vs normal ratio for "
               f"T1 ranged from {_f(rb['vs_normal'].min())} to "
               f"{_f(rb['vs_normal'].max())}; "
               f"{int((rb['vs_normal_lo'] > 1).sum())} of {len(rb)} had the "
               f"whole interval above 1")
        if len(rg):
            r = rg.iloc[0]
            txt += (f". A regression adjusted for hour and runway flow gives "
                    f"{_ci(r['ratio'], r['ratio_lo'], r['ratio_hi'])} "
                    f"(p = {_p(r['p_value'])})")
        out.append(txt + ".")
    prim = res["results"][(res["results"]["endpoint"] == PRIMARY_METRIC)
                          & (res["results"]["event_set"] == "all")]
    if len(prim):
        pr = prim.iloc[0]
        out.append(
            f"<b>How it happens:</b> per unit of traffic, T1 encounters after "
            f"÷ before were {_ci(pr['irr'], pr['irr_lo'], pr['irr_hi'])} "
            f"({_sig(pr['irr_lo'], pr['irr_hi'])}): the extra encounters come "
            f"with the extra traffic, not from pairs of aircraft being closer "
            f"than usual.")
    inv = res["involvement"]
    if len(inv):
        post = inv[(inv["tier"] == "T1") & (inv["window"] == "post")]
        if len(post):
            p_ = post.iloc[0]
            out.append(
                f"<b>The go-around aircraft itself</b> was involved in "
                f"{_ci(p_['ratio'], p_['ratio_lo'], p_['ratio_hi'])} times "
                f"the T1 encounters of an average aircraft present after the "
                f"go-around, and spent "
                f"{100 * p_['share_of_window_in_area']:.0f} % of the "
                f"after-window inside the area.")
    return out


def _checks_rows(res, chk, bl, eq, wm, risk):
    e = eq.iloc[0] if len(eq) else None
    q = wm["quality"].value_counts()
    comp = res.get("comparison") or {}
    rows = [["Question", "Answer"],
            ["Is the baseline reliable?",
             f"{bl.verdict}" + (" (see warnings in the baseline report)"
                                if bl.warnings else "")]]
    if e is not None and e["verdict"] == "NOT ASSESSED":
        rows.append(["Was the airspace normal before the go-arounds?",
                     "Not assessed: no events"])
    elif e is not None:
        rows.append([
            "Was the airspace normal before the go-arounds?",
            f"{'Yes' if e['verdict'] == 'PASS' else 'No' if e['verdict'] == 'FAIL' else e['verdict']}: "
            f"{100 * e['in_control_share']:.0f} % of before-windows in the "
            f"normal range; observed ÷ expected "
            f"{_ci(e['sir_pre'], e['sir_pre_ci'][0], e['sir_pre_ci'][1])}"])
    if comp:
        b = comp["before"].set_index("group")
        if "normal landings" in b.index and "in_normal_range" in b.columns:
            r = b.loc["normal landings"]
            how = {"month": "a baseline fitted without the landing's own "
                            "month",
                   "week": "a baseline fitted without the landing's own week",
                   "day": "a baseline fitted without the landing's own day"
                   }.get(comp.get("landing_scoring"), "the baseline")
            rows.append([
                "Was the airspace normal before the normal landings?",
                f"{100 * r['in_normal_range']:.0f} % of before-windows in the "
                f"normal range; observed ÷ expected "
                f"{_ci(r['oe_before'], r['oe_before_lo'], r['oe_before_hi'])} "
                f"(scored with {how})"])
        last = comp["before"].iloc[-1]
        if np.isfinite(last.get("traffic_ratio", np.nan)):
            rows.append([
                "Were go-arounds in busier periods than the landings?",
                f"Before-window traffic, go-arounds ÷ landings: "
                f"{_ci(last['traffic_ratio'], last['traffic_ratio_lo'], last['traffic_ratio_hi'])}"
                f" (1 = equally busy)"])
        pt = comp["pretrend"].set_index("measure")
        for key, name in (("T1_any", "T1 encounters"), ("pair_time", "traffic")):
            if key in pt.index:
                r = pt.loc[key]
                rows.append([
                    f"Were the groups already moving apart before? ({name})",
                    f"Pre-trend, go-arounds vs landings: "
                    f"{_ci(r['vs_normal'], r['vs_normal_lo'], r['vs_normal_hi'])}"
                    f" (1 = changed alike before the event)"])
        p = comp["placebo"].get("T1_any") or {}
        if p:
            rows.append([
                "Could random normal landings change as much?",
                f"{100 * p['share_at_least_as_large']:.1f} % of "
                f"{p['n_sets']:,} random sets of {p['n_events']} hour-matched "
                f"landings changed as much as the go-arounds "
                f"(p = {_p(p['p_value'])})"])
        rg = comp["regression"]
        if len(rg):
            r = rg.iloc[0]
            rows.append([
                "Does a regression agree?",
                f"T1, go-around vs normal adjusted for hour of day and runway "
                f"flow, clustered by day: "
                f"{_ci(r['ratio'], r['ratio_lo'], r['ratio_hi'])} "
                f"(p = {_p(r['p_value'])})"])
    rows.append([
        "Does the area reach high enough?",
        (f"Raised from {chk['charted_ceiling_ft_agl']:,.0f} to "
         f"{chk['effective_ceiling_ft_agl']:,.0f} ft so that 95 % of "
         f"go-around climb-outs stay inside"
         if chk["override"] else
         f"Yes, {chk['effective_ceiling_ft_agl']:,.0f} ft contains 95 % of "
         f"go-around climb-outs")])
    n_w = int(q.sum())
    rows.append([
        "How much data was unusable?",
        f"{int(q.get('bad', 0)):,} of {n_w:,} ten-minute windows "
        f"({100 * q.get('bad', 0) / max(n_w, 1):.1f} %) had receiver "
        f"outages or large gaps and were left out; "
        f"{int(risk['excluded'].sum())} events excluded for the same reason"])
    return rows


def summary(run_dir: Path, res: dict, chk: dict, ev: pd.DataFrame,
            sens: pd.DataFrame | None = None) -> Path:
    from .events import event_count_text, event_label
    S = _styles()
    bl = res["baseline"]
    results, eq = res["results"], res["equilibrium"]
    risk = res["risk"]
    wm = res["window_metrics"]
    comp = res.get("comparison") or {}
    figdir = run_dir / "figures"
    icao = ga.config.AIRPORT_ICAO
    story = [Paragraph(f"Proximity risk after go-arounds — {icao}", S["h1"])]
    days = sorted(wm["day"].unique())
    story.append(Paragraph(
        f"{run_dir.parent.parent.name} / {run_dir.name}  ·  {len(days)} "
        f"day{'s' if len(days) != 1 else ''}  "
        f"·  {event_count_text(ev)}  ·  {config.WINDOW_MIN:g}-minute windows  "
        f"·  area: {config.VOLUME_RADIUS_NM:g} NM radius, up to "
        f"{chk['effective_ceiling_ft_agl']:,.0f} ft above the airfield",
        S["sub"]))

    story.append(Paragraph("Key results", S["h2"]))
    for s_ in _key_sentences(res, ev):
        story.append(Paragraph(s_, S["bullet"], bulletText="•"))

    story.append(Paragraph("Checks", S["h2"]))
    story.append(_table(_checks_rows(res, chk, bl, eq, wm, risk),
                        [2.4 * inch, WIDTH - 2.4 * inch], S))

    story.append(Paragraph("How to read this report", S["h2"]))
    story.append(_glossary(S))
    story.append(PageBreak())

    # ---- 1. the main comparison
    _figure(story, S, "1. Before and after: go-arounds vs normal landings",
            figdir / "before_after_vs_landings.png",
            "Average per event in the 10 minutes before and the 10 minutes "
            "after. Orange: go-arounds. Grey: normal landings at the same "
            "hours of day, the change that happens anyway when an aircraft "
            "lands. The title of each panel gives the go-around vs normal "
            "ratio with its 95 % confidence interval.")

    # ---- 2. all measures
    _figure(story, S, "2. Every measure, after ÷ before",
            figdir / "change_vs_normal.png",
            "Each dot is the change from the 10 minutes before to the 10 "
            "minutes after, with its 95 % confidence interval. Orange: around "
            "go-arounds. Grey: around normal landings at the same hours.")
    ch = res["change"]
    if len(ch):
        rows = [["Measure", "Before", "After", "After ÷ before [95 % CI]",
                 "p-value", "Normal landings: after ÷ before",
                 "Go-around vs normal [95 % CI]"]]
        for _, r in ch.iterrows():
            nd = 1 if r["measure"] in ("aircraft_time", "pair_time") else 0
            unit = {"aircraft_time": " h", "pair_time": " h"}.get(
                r["measure"], "")
            scale = 1 / 3600.0 if r["measure"] == "aircraft_time" else 1.0
            few = bool(r.get("too_few", False))
            rows.append([r["label"],
                         f"{_f(r['before_total'] * scale, nd)}{unit}",
                         f"{_f(r['after_total'] * scale, nd)}{unit}",
                         "too few to tell" if few else
                         _ci(r["ratio"], r["ratio_lo"], r["ratio_hi"]),
                         "-" if few else _p(r["paired_p"]),
                         _ci(r["normal_ratio"], r["normal_ratio_lo"],
                             r["normal_ratio_hi"]),
                         "too few to tell" if few else
                         _ci(r["vs_normal"], r["vs_normal_lo"],
                             r["vs_normal_hi"])])
        story.append(_table(rows, [1.55 * inch, 0.65 * inch, 0.65 * inch,
                                   1.3 * inch, 0.6 * inch, 1.2 * inch,
                                   1.35 * inch], S))
        n_ctl = int(ch["n_controls"].max())
        story.append(Paragraph(
            f"Totals over {int(ch['n_events'].max())} events. Normal "
            f"landings: {n_ctl:,} landings weighted to the go-arounds' hours "
            f"of day. Aircraft time and traffic in hours. p-value: paired "
            f"test of after minus before across go-arounds. Encounter types "
            f"with fewer than 10 encounters are too few to tell.",
            S["caption"]))

    # ---- 3. minute by minute, raw
    _figure(story, S, "3. Minute by minute", figdir / "minute_by_minute.png",
            "All events lined up at time 0 (go-around: start of the climb; "
            "normal landing: lowest point). Top: aircraft in the area. "
            "Bottom: T1 encounters per minute. Raw counts, averaged per "
            "event, no baseline. Before 0 the two lines should run alike "
            "(grey shading: the pre-trend check window); a gap that opens "
            "after 0 is the effect of the go-around.")

    # ---- 4. were both groups comparable before?
    _figure(story, S, "4. Were go-arounds and landings comparable before?",
            figdir / "before_window_checks.png",
            "Left: each dot is one before-window, scored against the "
            "baseline for its traffic, time of day and runway direction; "
            "about 95 % should fall in the grey band if the airspace was "
            "normal. Landings are scored with a baseline that did not see "
            "their own month (week or day for short datasets). Right: how "
            "busy the before-windows were.")
    if comp:
        b = comp["before"]
        rows = [["Group", "n", "Before-windows in the normal range",
                 "Observed ÷ expected [95 % CI]", "Aircraft in the area",
                 "Traffic (pair-hours)"]]
        for _, r in b.iloc[:-1].iterrows():
            rows.append([r["group"], _f(int(r["n"])),
                         f"{100 * r.get('in_normal_range', np.nan):.0f} %"
                         if np.isfinite(r.get("in_normal_range", np.nan)) else "-",
                         _ci(r.get("oe_before", np.nan),
                             r.get("oe_before_lo", np.nan),
                             r.get("oe_before_hi", np.nan)),
                         _f(r.get("aircraft_before", np.nan), 1),
                         _f(r.get("traffic_before_h", np.nan), 2)])
        story.append(_table(rows, [1.4 * inch, 0.6 * inch, 1.45 * inch,
                                   1.5 * inch, 1.1 * inch, 1.2 * inch], S))
        pt = comp["pretrend"]
        if len(pt):
            rows = [["Pre-trend (20-10 min before → last 10 min before)",
                     "Go-arounds", "Normal landings",
                     "Go-arounds vs landings [95 % CI]"]]
            for _, r in pt.iterrows():
                if r["measure"] not in ("aircraft_time", "pair_time", "T1_any"):
                    continue
                rows.append([r["label"], _pct(r["ratio"]),
                             _pct(r["normal_ratio"]),
                             _ci(r["vs_normal"], r["vs_normal_lo"],
                                 r["vs_normal_hi"])])
            story.append(Spacer(1, 6))
            story.append(_table(rows, [2.9 * inch, 1.1 * inch, 1.3 * inch,
                                       2.0 * inch], S))
            story.append(Paragraph(
                "If go-arounds happened while traffic was already building, "
                "the go-arounds would change more than the landings here, "
                "before the event. A ratio near 1 means the two groups were "
                "moving alike until the go-around.", S["caption"]))

    # ---- 5. placebo
    _figure(story, S, "5. Could random landings change as much?",
            figdir / "placebo.png",
            "Grey: the after ÷ before ratio of random sets of normal landings, "
            "each set drawn to match the go-arounds' hours of day, put "
            "through the same calculation. Orange line: the go-arounds. The "
            "further right of the grey sets the orange line lies, the less "
            "likely the go-arounds' change is chance.")

    # ---- 6. robustness
    _figure(story, S, "6. Does the result hold under other choices?",
            figdir / "robustness.png",
            "T1 encounters, go-around vs normal landing, with 95 % confidence "
            "intervals (black: the main analysis). Each row changes one "
            "choice: how landings are matched, which go-arounds are used, "
            "the window length, or whether predicted encounters count.")
    if comp and len(comp["regression"]):
        rows = [["Regression check", "Go-around vs normal [95 % CI]",
                 "p-value", "Adjusted for"]]
        for _, r in comp["regression"].iterrows():
            rows.append([{"T1_any": "T1 encounters",
                          "pair_hours": "Traffic"}.get(r["measure"],
                                                       r["measure"]),
                         _ci(r["ratio"], r["ratio_lo"], r["ratio_hi"]),
                         _p(r["p_value"]), r["adjusted_for"]])
        story.append(_table(rows, [1.6 * inch, 1.9 * inch, 0.8 * inch,
                                   3.0 * inch], S))
        story.append(Paragraph(
            "A Poisson regression of the window counts with go-around, after "
            "and their interaction, plus hour-of-day and runway-flow effects; "
            "landings weighted to the go-arounds' hours, standard errors "
            "allowing for events on the same day.", S["caption"]))

    # ---- 7. how it happens
    story.append(PageBreak())
    story.append(Paragraph("How the increase happens", S["h1"]))
    _figure(story, S, "7. Per unit of traffic, by encounter type",
            figdir / "forest.png",
            "Encounters divided by traffic. Left: after ÷ before. Right: "
            "observed ÷ expected from the baseline, for the before-windows "
            "(blue) and after-windows (orange). Values near 1 mean pairs of "
            "aircraft were as close as usual for the traffic present.")
    prim = results[(results["event_set"] == "all")
                   & (results["type"] == "count")]
    if len(prim):
        rows = [["Encounter type", "Events", "Before", "After",
                 "After ÷ before per traffic [95 % CI]", "p-value",
                 "After vs normal [95 % CI]", "Smallest detectable change"]]
        for _, r in prim.iterrows():
            few = _too_few(r["sum_pre"], r["sum_post"])
            rows.append([L.metric(r["endpoint"]), _f(int(r["n_events"])),
                         _f(r["sum_pre"], 0), _f(r["sum_post"], 0),
                         "too few to tell" if few else
                         _ci(r["irr"], r["irr_lo"], r["irr_hi"]),
                         "-" if few else
                         _p(r.get("wilcoxon_paired_p", np.nan)),
                         "too few to tell" if few else
                         _ci(r["sir_post"], r["sir_post_lo"],
                             r["sir_post_hi"]),
                         "-" if few else _pct(r["mde_irr"])])
        story.append(_table(rows, [1.75 * inch, 0.5 * inch, 0.55 * inch,
                                   0.55 * inch, 1.25 * inch, 0.55 * inch,
                                   1.15 * inch, 1.0 * inch], S))
        story.append(Paragraph(
            "Rare types (T2, T3) have few encounters; with fewer than 10 the "
            "ratio is not shown.", S["caption"]))
    inv = res["involvement"]
    if len(inv):
        story.append(Paragraph("8. The go-around aircraft itself", S["h2"]))
        rows = [["Encounters", "Window", "Involving the go-around aircraft",
                 "Expected for an average aircraft", "Ratio [95 % CI]",
                 "Time inside the area"]]
        for _, r in inv.iterrows():
            rows.append([L.metric(f"{r['tier']}_any"),
                         "before" if r["window"] == "pre" else "after",
                         _f(r["observed"], 0), _f(r["expected"], 1),
                         _ci(r["ratio"], r["ratio_lo"], r["ratio_hi"]),
                         f"{100 * r['share_of_window_in_area']:.0f} %"])
        story.append(_table(rows, [1.5 * inch, 0.7 * inch, 1.2 * inch,
                                   1.3 * inch, 1.3 * inch, 1.0 * inch], S))
        story.append(Paragraph(
            "Ratio: encounters involving the go-around aircraft ÷ what an "
            "average aircraft in the area at the same time would have in the "
            "same time. Above 1 means the go-around aircraft is involved in "
            "more encounters than other aircraft. Only its time inside the "
            "area counts.", S["caption"]))
    _figure(story, S, "9. Where encounters happen",
            figdir / "encounter_density.png",
            "Share of T1 encounters in each cell of the area, for normal "
            "traffic (left) and the after-windows of go-arounds (right), on "
            "the same colour scale. Rings: 5 and 10 NM from the airport. "
            "Encounters in overlapping after-windows are counted once here, "
            "so the total can be slightly below the table's sum.")

    # ---- baseline and setup checks
    story.append(Paragraph("Baseline and setup checks", S["h1"]))
    _figure(story, S, "10. Was the airspace normal before each go-around?",
            figdir / "equilibrium.png",
            "Each dot is one go-around's before-window: how unusual its T1 "
            "encounter count was for the traffic present. If the airspace was "
            "normal beforehand, about 95 % of dots fall in the grey band.")
    _figure(story, S, "11. How well the baseline describes each day",
            figdir / "daily_levels.png",
            "Each dot is one day: all T1 encounters that day ÷ what the "
            "baseline expects for that day's traffic. Dots near 1 mean the "
            "baseline describes the day well. Orange ticks mark days with "
            "go-arounds.")
    _figure(story, S, "12. Does the analysis area reach high enough?",
            figdir / "ceiling_check.png",
            "Highest point of each go-around aircraft in the 10 minutes after "
            "the go-around. The area's ceiling is raised automatically when "
            "more than 5 % of climb-outs go above it.")
    if sens is not None and len(sens) and (figdir / "sensitivity.png").exists():
        _figure(story, S, "13. Does the result depend on our settings?",
                figdir / "sensitivity.png",
                "Each row reruns the whole analysis with one setting changed "
                "(black: the main analysis). Left: go-around vs normal "
                "landing. If the rows stay on the same side of 1 as the main "
                "analysis, the conclusion does not depend on that setting.")

    # ---- appendix
    _figure(story, S, "Appendix: minute by minute against the baseline",
            figdir / "superposed_epoch.png",
            "The same alignment as figure 3, but each minute's T1 count is "
            "divided by what the baseline expects for the traffic present "
            "(observed ÷ expected). This view removes the traffic, so it "
            "shows whether pairs were closer than usual, not whether there "
            "were more encounters.")

    # ---- events
    story.append(PageBreak())
    story.append(Paragraph("Events", S["h2"]))
    rows = [["Event", "Type", "Aircraft", "Runway", "Time (UTC)",
             "T1 before (expected)", "T1 after (expected)", "z before",
             "z after", "Used"]]
    zc = "_zmid" if "pre_T1_any_zmid" in risk.columns else "_z"
    for _, r in risk.iterrows():
        rows.append([
            r["event_id"], event_label(r["outcome"]),
            str(r["callsign"]), str(r["runway"]),
            f"{pd.Timestamp(r['t0_utc']):%Y-%m-%d %H:%M}",
            f"{_f(r.get('pre_T1_any'), 0)} ({_f(r.get('pre_T1_any_expected'), 1)})",
            f"{_f(r.get('post_T1_any'), 0)} ({_f(r.get('post_T1_any_expected'), 1)})",
            _f(r.get(f"pre_T1_any{zc}")), _f(r.get(f"post_T1_any{zc}")),
            "no" if r["excluded"] else "yes"])
    story.append(_table(rows, [0.6 * inch, 0.8 * inch, 0.75 * inch,
                               0.6 * inch, 1.05 * inch, 0.85 * inch,
                               0.85 * inch, 0.55 * inch, 0.5 * inch,
                               0.4 * inch], S, zebra=True))

    story.append(Paragraph("Method in brief", S["h2"]))
    for txt in [
        "<b>Data and area.</b> All ADS-B traffic within "
        f"{config.VOLUME_RADIUS_NM:g} NM of the airport, from the surface "
        "(airborne aircraft only) to the ceiling above. Positions repeated "
        "by the data provider after an aircraft stopped reporting are "
        "removed; the rest is placed on a one-second grid.",
        "<b>Encounters.</b> Every pair of airborne aircraft is checked "
        "every second against the T1, T2 and T3 zones, as observed and as "
        f"predicted {config.T_LOOKAHEAD_S:.0f} seconds ahead on straight "
        "tracks. Go-arounds and touch-and-goes come from the go-around "
        "detection run on the same data.",
        "<b>Comparison with normal landings.</b> The change from the window "
        "before to the window after each go-around is compared with the same "
        "change around full-stop landings (at least "
        f"{config.BASELINE_EXCLUSION_MIN:.0f} minutes from any go-around), "
        "weighted to the go-arounds' hours of day. Confidence intervals "
        "resample whole days, so events on the same day are not treated as "
        "independent. The checks: before-windows of both groups against the "
        "baseline, before-window traffic, the pre-trend, a placebo with "
        "random landing sets, a regression adjusted for hour and runway "
        "flow, and a robustness set.",
        "<b>Baseline.</b> A statistical model of encounters per ten-minute "
        "window, given traffic, time of day, day of week, month and runway "
        f"direction, learned from all windows more than "
        f"{config.BASELINE_EXCLUSION_MIN:.0f} minutes from any event and "
        "tested on held-out months (baseline report). It shows that the "
        "airspace was normal before the events, and gives the per-traffic "
        "view of how the increase happens.",
        "<b>Limitations.</b> Proximity is a stand-in for safety, not a "
        "collision probability. Aircraft without ADS-B, or below the "
        "receivers' coverage, are not seen. Controller instructions and "
        "pilot intent are not in the data, so the reasons behind a change "
        "cannot be observed directly.",
    ]:
        story.append(Paragraph(txt, S["body"]))
        story.append(Spacer(1, 5))
    out = run_dir / "summary.pdf"
    _doc(out, f"Proximity risk after go-arounds - {icao}").build(story)
    return out


# ---------------------------------------------------- baseline report ----

def _plain_warning(w: str) -> str:
    m = re.match(r"(\w+): model and non-parametric 95% limits disagree in "
                 r"(\d+)/(\d+) strata", w)
    if m:
        return (f"{L.metric(m[1])}: in {m[2]} of {m[3]} groups of similar "
                f"windows, two independent ways of setting the normal range "
                f"differ by more than "
                f"{100 * config.CROSSCHECK_MAX_REL_DIFF:.0f} %.")
    m = re.match(r"(\w+): out-of-sample (95|99)% exceedance ([\d.]+)% "
                 r"outside \[(.+)\]", w)
    if m:
        return (f"{L.metric(m[1])}: in held-out months, {m[3]} % of normal "
                f"windows fell outside the {m[2]} % limit (target range "
                f"{m[4]}). The normal range is slightly "
                f"{'too narrow' if float(m[3]) > (5 if m[2] == '95' else 1) else 'too wide'}.")
    m = re.match(r"(\d+)/(\d+) strata have fewer than (\d+) windows", w)
    if m:
        return (f"{m[1]} of {m[2]} groups of similar windows (time of day × "
                f"traffic level × runway direction) have fewer than {m[3]} "
                f"windows, so their own estimates are less certain.")
    m = re.match(r"(\w+): baseline estimates have not stabilised", w)
    if m:
        return (f"{L.metric(m[1])}: the estimate still changes as days are "
                f"added; more data would make it steadier.")
    return w


COEF_RE = re.compile(r"(\w+)\[(.+)\]")


def _plain_term(name: str, block_h: int) -> str:
    if name == "intercept":
        return "Base level (most common conditions)"
    if name == "log_n_aircraft":
        return "Number of aircraft in the area"
    m = COEF_RE.match(name)
    if not m:
        return name.replace("_", " ")
    f, lev = m[1], m[2]
    if f == "hour_block":
        k = int(lev)
        if block_h == 1:
            return f"Local hour {k:02d}"
        return f"Local hours {k * block_h:02d}-{(k + 1) * block_h - 1:02d}"
    if f == "daytype":
        return lev.capitalize()
    if f == "month":
        return f"Month {lev}"
    if f == "flow":
        mm = re.match(r"flow_(\d+)", lev)
        if mm:
            return f"Runways heading {int(mm[1])}°"
        return {"none": "No runway in use nearby",
                "mixed": "Mixed runway directions"}.get(lev, f"Runway use: {lev}")
    return f"{f.replace('_', ' ')}: {lev}"


def baseline_report(bl, folder: Path) -> Path:
    S = _styles()
    icao = ga.config.AIRPORT_ICAO
    val = folder / "validation"
    story = [Paragraph(f"Baseline report — {icao} / {folder.name}", S["h1"]),
             Paragraph(f"{bl.info.get('n_days')} days, "
                       f"{bl.info.get('n_windows'):,} ten-minute windows, "
                       f"{bl.info.get('date_first')} to "
                       f"{bl.info.get('date_last')}", S["sub"])]
    story.append(Paragraph(
        "The baseline says how many encounters a normal ten-minute window "
        "has, given its traffic, time of day, day of week, month and runway "
        "direction. It is learned from every window more than "
        f"{config.BASELINE_EXCLUSION_MIN:.0f} minutes from a go-around and "
        "checked on months it has not seen. Terms are defined in the "
        "summary report.", S["body"]))
    story.append(Paragraph(f"Verdict: {bl.verdict}", S["h2"]))
    if bl.warnings:
        for w in bl.warnings:
            story.append(Paragraph(_plain_warning(w), S["bullet"],
                                   bulletText="•"))
    else:
        story.append(Paragraph("All checks passed.", S["body"]))

    ex = bl.exclusion_log
    story.append(Paragraph("Windows used", S["h2"]))
    rows = [["", "Windows"],
            ["All ten-minute windows", _f(ex.get("n_windows"))],
            [f"Left out: within {config.BASELINE_EXCLUSION_MIN:.0f} min of a "
             "go-around", _f(ex.get("excluded_near_go_around"))],
            ["Left out: receiver outage or large gaps",
             _f(ex.get("excluded_bad_quality"))],
            ["Left out: not fully covered by data",
             _f(ex.get("excluded_partial_coverage"))],
            ["Used", _f(ex.get("kept"))],
            ["  of which with no aircraft pairs (quiet night windows)",
             _f(ex.get("kept_zero_exposure"))]]
    story.append(_table(rows, [4.6 * inch, 1.2 * inch], S))

    story.append(Paragraph("Accuracy on months it has not seen", S["h2"]))
    lo95, hi95 = config.VALIDATION_PASS_95
    lo99, hi99 = config.VALIDATION_PASS_99
    rows = [["Encounter type", "Encounters in the baseline",
             "Rate per pair-hour [95 % CI]",
             f"Outside the 95 % limit (target {100 * lo95:g}-{100 * hi95:g} %)",
             f"Outside the 99 % limit (target {100 * lo99:g}-{100 * hi99:g} %)",
             "Passed"]]
    for m, v in bl.validation.items():
        ci = v.get("rate_ci") or [np.nan, np.nan]
        rows.append([
            L.metric(m), _f(v.get("n_events")),
            _ci(v.get("rate_per_pair_hour", np.nan), ci[0], ci[1]),
            f"{100 * v['exceed95_rate']:.1f} %" if "exceed95_rate" in v else "-",
            f"{100 * v['exceed99_rate']:.2f} %" if "exceed99_rate" in v else "-",
            ("yes" if v.get("pass95") and v.get("pass99") else "no")
            if "pass95" in v else "too few to test"])
    story.append(_table(rows, [2.0 * inch, 0.95 * inch, 1.3 * inch,
                               1.15 * inch, 1.15 * inch, 0.75 * inch], S))
    story.append(Paragraph(
        "Each month is held out in turn, the baseline is rebuilt from the "
        "other months, and the held-out windows are checked against it. A "
        "correct baseline puts about 5 % of windows outside its 95 % limit "
        "and about 1 % outside its 99 % limit.", S["caption"]))

    _figure(story, S, "Held-out months", val / "calibration_exceedance.png",
            "Dots: share of held-out windows outside each limit, with 95 % "
            "confidence intervals. Shaded bands: the target ranges.")
    _figure(story, S, "Shape of the errors", val / "calibration_pit_qq.png",
            "Left: where each held-out window fell within its predicted range; "
            "a correct baseline gives flat bars at 1. Right: held-out z-scores "
            "against the values a correct baseline would give; a correct "
            "baseline keeps the dots on the dashed line.")
    _figure(story, S, "Is there enough data?", val / "stability_curve.png",
            "The baseline rebuilt from random subsets of days. When the line "
            "settles inside the grey band (±5 % of the full-data value), more "
            "days would not change it much.")

    model = bl.count_models.get(PRIMARY_METRIC)
    if model is not None and model.coef is not None:
        story.append(PageBreak())
        story.append(Paragraph("What the baseline accounts for (T1 "
                               "encounters)", S["h2"]))
        block = model.design.hour_block_h if model.design else 1
        rows = [["Condition", "Effect on the encounter rate"]]

        def order(item):
            n = item[0]
            m = COEF_RE.match(n)
            if m and m[1] == "hour_block":
                return (0, int(m[2]), n)
            return (1 if m else 2, 0, n)

        terms = sorted(((n, c) for n, c in zip(model.coef_names, model.coef)
                        if n != "intercept"), key=order)
        for n, c in terms:
            if n == "log_n_aircraft":
                rows.append([_plain_term(n, block),
                             f"× {2 ** c:.2f} per doubling"])
            else:
                rows.append([_plain_term(n, block), f"× {np.exp(c):.2f}"])
        story.append(_table(rows, [4.6 * inch, 1.6 * inch], S, zebra=True))
        story.append(Paragraph(
            "Each line multiplies the encounter rate per unit of traffic "
            "relative to the most common conditions (× 1.00 = no effect). "
            f"Extra randomness between windows (dispersion): "
            f"{model.alpha:.3f}.", S["caption"]))

    prim = bl.validation.get(PRIMARY_METRIC, {})
    suff = prim.get("sufficiency_table")
    if suff is not None and len(suff):
        story.append(Paragraph("Data per group of similar windows", S["h2"]))
        rows = [["Group (local hours, traffic level, runway direction)",
                 "Windows", "T1 encounters", "Pair-hours", "Fewer than "
                 f"{config.SUFFICIENCY_MIN_WINDOWS_PER_STRATUM} windows"]]
        for _, r in suff.iterrows():
            rows.append([_plain_stratum(r["stratum"]),
                         _f(int(r["n_windows"])), _f(int(r["n_events"])),
                         _f(r["pair_hours"], 1),
                         "yes" if r["flagged"] else ""])
        story.append(_table(rows, [3.3 * inch, 0.8 * inch, 0.9 * inch,
                                   0.9 * inch, 1.1 * inch], S, zebra=True))

    if len(bl.phase1_removed):
        story.append(Paragraph("Extreme windows removed before fitting",
                               S["h2"]))
        rows = [["Encounter type", "Window", "Observed", "Expected",
                 "Chance under the model"]]
        for _, r in bl.phase1_removed.iterrows():
            rows.append([L.metric(r["metric"]), str(r["window_id"]),
                         _f(r[r["metric"]]), _f(r["expected"]),
                         f"{r['p_upper']:.1e}"])
        story.append(_table(rows, [2.0 * inch, 2.0 * inch, 0.8 * inch,
                                   0.8 * inch, 1.3 * inch], S))

    notes = [f"{L.metric(m)}: {'; '.join(v['notes'])}"
             for m, v in bl.validation.items() if v.get("notes")]
    if notes:
        story.append(Paragraph("Technical notes", S["h2"]))
        for n in notes:
            story.append(Paragraph(n, S["small"]))
    out = folder / "baseline_report.pdf"
    _doc(out, f"Baseline report - {icao}").build(story)
    return out


def _plain_stratum(key: str) -> str:
    m = re.match(r"h(\d+)\|t(\d+)\|(.+)", str(key))
    if not m:
        return str(key)
    b = config.STRATA_HOUR_BLOCK_H
    h = int(m[1])
    level = {0: "low", 1: "medium", 2: "high"}.get(int(m[2]), m[2])
    fl = m[3]
    mm = re.match(r"flow_(\d+)", fl)
    fl = (f"runways heading {int(mm[1])}°" if mm else
          {"mixed": "mixed runway use", "none": "no runway use"}.get(fl, fl))
    return f"{h * b:02d}-{(h + 1) * b - 1:02d} h, {level} traffic, {fl}"
