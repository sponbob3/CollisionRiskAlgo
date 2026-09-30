"""
Render the summary report as a polished PDF (results/summary_report.pdf),
with the report figures embedded.

Usage:
    python -m goaround_pipeline.report_pdf
"""

from __future__ import annotations

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import config
from .report import GA_OUTCOMES, load

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d8d7d2")

OUTCOME_LABELS = {
    "full_stop": "Full-stop landing",
    "touch_and_go": "Touch-and-go",
    "go_around": "Go-around",
    "low_approach": "Low approach",
    "ga_ambiguous": "Ambiguous go-around candidate",
    "continued_approach": "Continued approach (fragment)",
    "unresolved": "Unresolved",
}


def _style_table(data, col_align=None, header=True):
    t = Table(data, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]
    if header:
        style.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def main(out_dir=None) -> None:
    if out_dir is not None:
        from pathlib import Path
        config.OUTPUT_DIR = Path(out_dir)
    df = load()
    att = df[df.is_attempt]
    ga = df[df.outcome == "go_around"]
    amb = df[df.outcome == "ga_ambiguous"]
    n_att = len(att)
    rate = 1000 * len(ga) / n_att

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=17,
                        textColor=INK, spaceAfter=2, alignment=0)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9.5,
                         textColor=MUTED, spaceAfter=14)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                        textColor=INK, spaceBefore=16, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5,
                          textColor=INK, leading=13.5)
    big = ParagraphStyle("big", parent=styles["Normal"], fontSize=11,
                         textColor=INK, leading=16)

    figdir = config.OUTPUT_DIR / "figures"
    story = []

    story.append(Paragraph(f"Go-Around Detection at "
                           f"{config.AIRPORT_ICAO} — Summary Report", h1))
    story.append(Paragraph(
        f"{config.AIRPORT_ICAO} approaches, {df['t_local'].dt.date.min()} to "
        f"{df['t_local'].dt.date.max()} &nbsp;·&nbsp; ADS-B based detection "
        "pipeline (goaround_pipeline)", sub))

    story.append(Paragraph(
        f"<b>{len(ga):,} go-around events</b> were identified among "
        f"<b>{n_att:,} independent approach attempts</b> "
        f"({len(df):,} aligned segments including continued fragments) — "
        f"a rate of <b>{rate:.1f} per 1,000 approaches</b> under the strict "
        f"definition. A further {len(amb)} candidates with borderline "
        "low-point geometry are retained and flagged as ambiguous.", big))

    story.append(Paragraph("Outcome breakdown", h2))
    vc = df["outcome"].value_counts()
    rows = [["Outcome", "Count", "Share"]]
    for k, v in vc.items():
        rows.append([OUTCOME_LABELS.get(k, k), f"{v:,}",
                     f"{100 * v / len(df):.1f}%"])
    story.append(_style_table(rows))

    story.append(Paragraph("Monthly", h2))
    g = att.groupby("month").agg(approaches=("is_ga", "size"),
                                 ga=("is_ga", "sum"))
    g["rate"] = 1000 * g["ga"] / g["approaches"]
    rows = [["Month", "Approaches", "Go-arounds", "Rate /1,000"]]
    for m, r in g.iterrows():
        rows.append([m, f"{int(r['approaches']):,}", f"{int(r['ga'])}",
                     f"{r['rate']:.1f}"])
    story.append(_style_table(rows))
    story.append(Image(str(figdir / "monthly_go_arounds.png"),
                       width=6.9 * inch, height=6.9 * inch * 3.8 / 11))

    story.append(PageBreak())
    story.append(Paragraph("By runway", h2))
    g = att[att.runway.notna() & (att.runway != "")].groupby("runway").agg(
        approaches=("is_ga", "size"), ga=("is_ga", "sum"))
    g["rate"] = 1000 * g["ga"] / g["approaches"]
    g = g.sort_values("approaches", ascending=False)
    rows = [["Runway", "Approaches", "Go-arounds", "Rate /1,000"]]
    for m, r in g.iterrows():
        rows.append([m, f"{int(r['approaches']):,}", f"{int(r['ga'])}",
                     f"{r['rate']:.1f}"])
    story.append(_style_table(rows))
    story.append(Image(str(figdir / "runway_rate.png"),
                       width=6.4 * inch, height=6.4 * inch * 3.6 / 8))

    story.append(Paragraph("Go-around event characteristics", h2))
    d = ga[["min_agl_ft", "init_agl_ft", "peak_climb_fpm",
            "alt_regain_ft"]].describe()
    rows = [["", "Min AGL (ft)", "Initiation AGL (ft)",
             "Peak climb (fpm)", "Altitude regained (ft)"]]
    for stat in ["mean", "50%", "25%", "75%", "min", "max"]:
        label = {"50%": "median", "25%": "25th pct",
                 "75%": "75th pct"}.get(stat, stat)
        rows.append([label] + [f"{d.loc[stat, c]:,.0f}" for c in d.columns])
    story.append(_style_table(rows))
    nx = ga["next_action"].value_counts()
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"After going around, {nx.get('reapproach', 0):,} aircraft "
        f"({100 * nx.get('reapproach', 0) / len(ga):.0f}%) flew another "
        "approach in the same flight; the rest end their data record "
        "(departure or coverage loss).", body))

    story.append(Paragraph("Where go-arounds begin", h2))
    story.append(Image(str(figdir / "go_around_map.png"),
                       width=5.6 * inch, height=5.6 * inch * 6.4 / 8.6))

    story.append(PageBreak())
    story.append(Paragraph("Method in brief", h2))
    for txt in [
        f"<b>Detection.</b> Each aircraft-day is split into flight legs on "
        f"{config.SEGMENT_GAP_MINUTES:.0f}-minute gaps. An approach attempt "
        f"is a &ge;{config.APPROACH_MIN_DURATION_S:.0f} s interval aligned "
        f"with a runway extended centerline "
        f"(&lt;{config.APPROACH_MAX_DIST_NM:.0f} NM out, "
        f"&lt;{config.APPROACH_MAX_XTRACK_NM:.2f} NM cross-track, track "
        f"within {config.APPROACH_MAX_TRACK_DELTA_DEG:.0f}&deg; of the "
        f"runway heading, below {config.APPROACH_MAX_AGL_FT:,.0f} ft AGL, "
        f"descending). Parallel-runway assignment uses a "
        f"low-altitude-weighted vote.",
        f"<b>Classification.</b> Touchdown evidence (on-ground flag, "
        f"AGL &le; {config.TOUCHDOWN_AGL_FT:.0f} ft, a "
        f"&ge;{config.TOUCHDOWN_PLATEAU_MIN_S:.0f} s flat stretch below "
        f"{config.TOUCHDOWN_PLATEAU_MAX_AGL_FT:.0f} ft AGL, groundspeed "
        f"&le; {config.TOUCHDOWN_GS_KT:.0f} kt, or a data dropout below "
        f"{config.TOUCHDOWN_GAP_AGL_FT:.0f} ft) separates landings and "
        f"touch-and-goes. A go-around is a no-touchdown approach whose low "
        f"point (below {config.GA_MAX_LOW_AGL_FT:,.0f} ft AGL, after "
        f"&ge;{config.GA_MIN_OBSERVED_DESCENT_FT:.0f} ft of observed "
        f"descent) reverses into a sustained climb "
        f"(&ge;{config.GA_CLIMB_VRATE_FPM:.0f} fpm for "
        f"&ge;{config.GA_CLIMB_DURATION_S:.0f} s, regaining "
        f"&ge;{config.GA_ALT_REGAIN_FT:.0f} ft) with "
        f"&le;{config.LEVEL_GA_MAX_S:.0f} s spent at the profile low "
        f"point. Level segments of &ge;{config.LEVEL_LOWAPP_MIN_S:.0f} s "
        f"classify as deliberate low approaches; "
        f"{config.LEVEL_GA_MAX_S:.0f}-{config.LEVEL_LOWAPP_MIN_S:.0f} s "
        f"cases are flagged ambiguous. "
        "The two populations are separated by an empty gap in the "
        "plateau-duration distribution (figure below).",
        "<b>Validation.</b> Stratified manual review of event plots; "
        "cross-agreement with the traffic library's published go-around "
        "detector (all disagreements definitional); one-at-a-time "
        "threshold sensitivity: counts stable within ±5% for most "
        "parameters, ±10-20% only for the two parameters defining the "
        "class boundary at flare height.",
        "<b>Limitations.</b> Pilot intent is invisible in surveillance "
        "data (practice and directed go-arounds are counted alike, as "
        "specified). Go-arounds initiated entirely below the local ADS-B "
        "coverage floor (~50-250 ft AGL) are undetectable. Scope is "
        "go-arounds from established, runway-aligned finals; circling "
        "breakoffs are out of scope.",
    ]:
        story.append(Paragraph(txt, body))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 4))
    story.append(Image(str(figdir / "plateau_separation.png"),
                       width=6.4 * inch, height=6.4 * inch * 3.8 / 8.5))
    story.append(Spacer(1, 2))
    story.append(Image(str(figdir / "go_around_profiles.png"),
                       width=6.9 * inch, height=6.9 * inch * 3.6 / 11))

    out = config.OUTPUT_DIR / "summary_report.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=letter,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=f"Go-Around Detection at {config.AIRPORT_ICAO} - Summary Report",
    )
    doc.build(story)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
