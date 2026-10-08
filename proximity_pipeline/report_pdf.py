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
     "after the go-around starts climbing."),
    ("Traffic", "Aircraft-pair time in the area: every pair of aircraft "
     "present at the same moment, summed over time. One extra aircraft "
     "among ten adds about 20 %."),
    ("Baseline, expected", "What normal windows produce for the same "
     "traffic, time of day, day of week, month and runway direction, "
     "learned from the whole dataset away from go-arounds."),
    ("Observed ÷ expected", "1 means exactly as normal traffic; 1.2 means "
     "20 % more."),
    ("z (how unusual)", "Distance from the baseline in standard units: 0 is "
     "typical, beyond ±2 is unusual (about 5 % of normal windows)."),
    ("Normal range", "Where 95 % of normal windows fall."),
    ("After ÷ before (IRR)", "Rate ratio: encounters per unit of traffic "
     "after the go-around divided by before. 1 means no change."),
    ("After vs normal (SIR)", "Observed ÷ expected for the after-windows, "
     "summed over all events. 1 means like normal traffic."),
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


# ------------------------------------------------------------ summary ----

def _key_sentences(res, ev) -> list[str]:
    from .events import event_count_text
    out = []
    if len(ev) == 0:
        return ["No go-arounds or touch-and-goes were found in this dataset, "
                "so there is nothing to compare. The baseline and the data "
                "checks below are still valid."]
    ch = res["change"].set_index("measure") if len(res["change"]) else None
    prim = res["results"][(res["results"]["endpoint"] == PRIMARY_METRIC)
                          & (res["results"]["event_set"] == "all")]
    pr = prim.iloc[0] if len(prim) else None

    def sig(lo, hi):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "not testable"
        return ("statistically significant" if lo > 1 or hi < 1
                else "not statistically significant")

    if ch is not None and "T1_any" in ch.index:
        r = ch.loc["T1_any"]
        out.append(
            f"<b>Total T1 encounters:</b> {_f(r['before_total'], 0)} in the "
            f"before-windows and {_f(r['after_total'], 0)} in the "
            f"after-windows of {event_count_text(ev)}: "
            f"<b>{_pct(r['ratio'])}</b> (after ÷ before "
            f"{_ci(r['ratio'], r['ratio_lo'], r['ratio_hi'])}; "
            f"{sig(r['ratio_lo'], r['ratio_hi'])}).")
    if ch is not None and "pair_time" in ch.index:
        r = ch.loc["pair_time"]
        out.append(
            f"<b>Traffic:</b> {_pct(r['ratio'])} after the go-around "
            f"({_ci(r['ratio'], r['ratio_lo'], r['ratio_hi'])}). After a "
            f"normal landing at the same hours it changed by "
            f"{_pct(r['normal_ratio'])}.")
    if pr is not None:
        out.append(
            f"<b>Per unit of traffic:</b> T1 encounters after ÷ before "
            f"<b>{_ci(pr['irr'], pr['irr_lo'], pr['irr_hi'])}</b> "
            f"({sig(pr['irr_lo'], pr['irr_hi'])}); after-windows vs normal "
            f"traffic {_ci(pr['sir_post'], pr['sir_post_lo'], pr['sir_post_hi'])}."
            f" The study could detect a change of {_pct(pr['mde_irr'])} or "
            f"more.")
    if ch is not None and "T1_any" in ch.index:
        r = ch.loc["T1_any"]
        out.append(
            f"<b>Compared with normal landings:</b> the before-to-after "
            f"change in T1 encounters around go-arounds was "
            f"{_ci(r['vs_normal'], r['vs_normal_lo'], r['vs_normal_hi'])} "
            f"times the change around normal landings at the same hours "
            f"({sig(r['vs_normal_lo'], r['vs_normal_hi'])}).")
    inv = res["involvement"]
    if len(inv):
        post = inv[(inv["tier"] == "T1") & (inv["window"] == "post")]
        pre = inv[(inv["tier"] == "T1") & (inv["window"] == "pre")]
        if len(post):
            p_ = post.iloc[0]
            txt = (f"<b>The go-around aircraft itself</b> was involved in "
                   f"{_ci(p_['ratio'], p_['ratio_lo'], p_['ratio_hi'])} "
                   f"times the T1 encounters of an average aircraft present "
                   f"after the go-around")
            if len(pre):
                txt += f" ({_f(pre.iloc[0]['ratio'])} times before it)"
            txt += (f"; it spent {100 * p_['share_of_window_in_area']:.0f} % "
                    f"of the after-window inside the area.")
            out.append(txt)
    return out


def summary(run_dir: Path, res: dict, chk: dict, ev: pd.DataFrame,
            sens: pd.DataFrame | None = None) -> Path:
    from .events import event_count_text, event_label
    S = _styles()
    bl = res["baseline"]
    results, eq = res["results"], res["equilibrium"]
    risk = res["risk"]
    wm = res["window_metrics"]
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
    for s in _key_sentences(res, ev):
        story.append(Paragraph(s, S["bullet"], bulletText="•"))

    story.append(Paragraph("Checks", S["h2"]))
    e = eq.iloc[0] if len(eq) else None
    q = wm["quality"].value_counts()
    checks = [["Question", "Answer"],
              ["Is the baseline reliable?",
               f"{bl.verdict}" + (" (see warnings in the baseline report)"
                                  if bl.warnings else "")]]
    if e is not None and e["verdict"] == "NOT ASSESSED":
        checks.append(["Was the airspace normal before the go-arounds?",
                       "Not assessed: no events"])
    elif e is not None:
        checks.append([
            "Was the airspace normal before the go-arounds?",
            f"{'Yes' if e['verdict'] == 'PASS' else 'No' if e['verdict'] == 'FAIL' else e['verdict']}: "
            f"{100 * e['in_control_share']:.0f} % of before-windows in the "
            f"normal range; observed ÷ expected "
            f"{_ci(e['sir_pre'], e['sir_pre_ci'][0], e['sir_pre_ci'][1])}"])
    checks.append([
        "Does the area reach high enough?",
        (f"Raised from {chk['charted_ceiling_ft_agl']:,.0f} to "
         f"{chk['effective_ceiling_ft_agl']:,.0f} ft so that 95 % of "
         f"go-around climb-outs stay inside"
         if chk["override"] else
         f"Yes, {chk['effective_ceiling_ft_agl']:,.0f} ft contains 95 % of "
         f"go-around climb-outs")])
    n_w = int(q.sum())
    checks.append([
        "How much data was unusable?",
        f"{int(q.get('bad', 0)):,} of {n_w:,} ten-minute windows "
        f"({100 * q.get('bad', 0) / max(n_w, 1):.1f} %) had receiver "
        f"outages or large gaps and were left out; "
        f"{int(risk['excluded'].sum())} events excluded for the same reason"])
    story.append(_table(checks, [2.4 * inch, WIDTH - 2.4 * inch], S))

    story.append(Paragraph("How to read this report", S["h2"]))
    story.append(_glossary(S))
    story.append(PageBreak())

    # ---- 1. totals, traffic, normal landings
    _figure(story, S, "1. What changes after a go-around",
            figdir / "change_vs_normal.png",
            "Each dot is the change from the 10 minutes before to the 10 "
            "minutes after, with its 95 % confidence interval. Orange: around "
            "go-arounds. Grey: around normal landings at the same hours of day "
            "- the change that happens anyway. A go-around keeps one aircraft "
            "airborne that would otherwise have landed, which on its own "
            "raises traffic.")
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
            rows.append([r["label"],
                         f"{_f(r['before_total'] * scale, nd)}{unit}",
                         f"{_f(r['after_total'] * scale, nd)}{unit}",
                         _ci(r["ratio"], r["ratio_lo"], r["ratio_hi"]),
                         _p(r["paired_p"]),
                         _ci(r["normal_ratio"], r["normal_ratio_lo"],
                             r["normal_ratio_hi"]),
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
            f"test of after minus before across events.", S["caption"]))

    # ---- 2. time course
    _figure(story, S, "2. Minute by minute", figdir / "superposed_epoch.png",
            "All events lined up at the moment the go-around starts climbing "
            "(0). The line is T1 encounters observed ÷ expected for the "
            "traffic present, minute by minute; above the dashed line means "
            "more than normal traffic would give. Blue shading: before-window; "
            "orange shading: after-window.")

    # ---- 3. per unit of traffic, by encounter type
    _figure(story, S, "3. Per unit of traffic, by encounter type",
            figdir / "forest.png",
            "Left: after ÷ before, per unit of traffic. Right: observed ÷ "
            "expected compared with normal traffic, for the before-windows "
            "(blue) and after-windows (orange). Dots right of the dashed line "
            "mean more encounters; bars are 95 % confidence intervals.")
    prim = results[(results["event_set"] == "all")
                   & (results["type"] == "count")]
    if len(prim):
        rows = [["Encounter type", "Events", "Before", "After",
                 "After ÷ before per traffic [95 % CI]", "p-value",
                 "After vs normal [95 % CI]", "Smallest detectable change"]]
        for _, r in prim.iterrows():
            rows.append([L.metric(r["endpoint"]), _f(int(r["n_events"])),
                         _f(r["sum_pre"], 0), _f(r["sum_post"], 0),
                         _ci(r["irr"], r["irr_lo"], r["irr_hi"]),
                         _p(r.get("wilcoxon_paired_p", np.nan)),
                         _ci(r["sir_post"], r["sir_post_lo"],
                             r["sir_post_hi"]),
                         _pct(r["mde_irr"])])
        story.append(_table(rows, [1.75 * inch, 0.5 * inch, 0.55 * inch,
                                   0.55 * inch, 1.25 * inch, 0.55 * inch,
                                   1.15 * inch, 1.0 * inch], S))
        story.append(Paragraph(
            "Rare types (T2, T3) have too few encounters for firm "
            "conclusions; their smallest detectable change is large.",
            S["caption"]))

    # ---- 4. the go-around aircraft itself
    inv = res["involvement"]
    if len(inv):
        story.append(Paragraph("4. The go-around aircraft itself", S["h2"]))
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

    # ---- 5. before-window check
    _figure(story, S, "5. Was the airspace normal before each go-around?",
            figdir / "equilibrium.png",
            "Each dot is one go-around's before-window: how unusual its T1 "
            "encounter count was for the traffic present. If the airspace was "
            "normal beforehand, about 95 % of dots fall in the grey band.")

    # ---- 6. daily fit
    _figure(story, S, "6. How well the baseline describes each day",
            figdir / "daily_levels.png",
            "Each dot is one day: all T1 encounters that day ÷ what the "
            "baseline expects for that day's traffic. Dots near 1 mean the "
            "baseline describes the day well. Orange ticks mark days with "
            "go-arounds.")

    # ---- 7. where
    _figure(story, S, "7. Where encounters happen",
            figdir / "encounter_density.png",
            "Share of T1 encounters in each cell of the area, for normal "
            "traffic (left) and the after-windows of go-arounds (right), on "
            "the same colour scale. Rings: 5 and 10 NM from the airport.")

    # ---- 8. ceiling
    _figure(story, S, "8. Does the analysis area reach high enough?",
            figdir / "ceiling_check.png",
            "Highest point of each go-around aircraft in the 10 minutes after "
            "the go-around. The area's ceiling is raised automatically when "
            "more than 5 % of climb-outs go above it.")

    # ---- 9. sensitivity
    if sens is not None and len(sens) and (figdir / "sensitivity.png").exists():
        _figure(story, S, "9. Does the result depend on our choices?",
                figdir / "sensitivity.png",
                "Each row reruns the analysis with one setting changed (black: "
                "the main analysis). If the rows stay on the same side of 1 as "
                "the main analysis, the conclusion does not depend on that "
                "choice.")

    # ---- events
    story.append(PageBreak())
    story.append(Paragraph("Events", S["h2"]))
    rows = [["Event", "Type", "Aircraft", "Runway", "Time (UTC)",
             "T1 before (expected)", "T1 after (expected)", "z before",
             "z after", "Used"]]
    for _, r in risk.iterrows():
        rows.append([
            r["event_id"], event_label(r["outcome"]),
            str(r["callsign"]), str(r["runway"]),
            f"{pd.Timestamp(r['t0_utc']):%Y-%m-%d %H:%M}",
            f"{_f(r.get('pre_T1_any'), 0)} ({_f(r.get('pre_T1_any_expected'), 1)})",
            f"{_f(r.get('post_T1_any'), 0)} ({_f(r.get('post_T1_any_expected'), 1)})",
            _f(r.get("pre_T1_any_z")), _f(r.get("post_T1_any_z")),
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
        "<b>Baseline.</b> A statistical model of encounters per ten-minute "
        "window, given traffic, time of day, day of week, month and runway "
        f"direction, learned from all windows more than "
        f"{config.BASELINE_EXCLUSION_MIN:.0f} minutes from any event and "
        "tested on held-out months (baseline report).",
        "<b>Comparisons.</b> Totals and traffic after vs before each event, "
        "compared with the same change around normal landings at the same "
        "hours; encounters per unit of traffic after vs before and vs the "
        "baseline; confidence intervals resample whole days.",
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
