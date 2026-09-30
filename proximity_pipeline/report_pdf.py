"""
summary.pdf (per run) and baseline_report.pdf (per stored baseline), in
the go-around pipeline's report style: exact numbers first, figures
embedded, every verdict on page 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import config
from . import goaround_adapter as ga
from .baseline import PRIMARY_METRIC

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d8d7d2")
RED = colors.HexColor("#b3261e")


def _styles():
    st = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=st["Title"], fontSize=16,
                             textColor=INK, spaceAfter=2, alignment=0),
        "sub": ParagraphStyle("sub", parent=st["Normal"], fontSize=9.5,
                              textColor=MUTED, spaceAfter=12),
        "h2": ParagraphStyle("h2", parent=st["Heading2"], fontSize=12,
                             textColor=INK, spaceBefore=14, spaceAfter=5),
        "body": ParagraphStyle("body", parent=st["Normal"], fontSize=9.5,
                               textColor=INK, leading=13.5),
        "big": ParagraphStyle("big", parent=st["Normal"], fontSize=11,
                              textColor=INK, leading=16),
        "warn": ParagraphStyle("warn", parent=st["Normal"], fontSize=9.5,
                               textColor=RED, leading=13.5),
    }


def _table(data, header=True, col_widths=None):
    t = Table(data, hAlign="LEFT", colWidths=col_widths)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]
    if header:
        style.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def _f(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{nd}f}"
    return str(v)


def _image(path: Path, width: float):
    if not Path(path).exists():
        return Spacer(1, 1)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def _doc(path: Path, title: str):
    return SimpleDocTemplate(
        str(path), pagesize=letter, leftMargin=0.75 * inch,
        rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=title)


# ------------------------------------------------------------ summary ----

def summary(run_dir: Path, res: dict, chk: dict, ev: pd.DataFrame,
            sens: pd.DataFrame | None = None) -> Path:
    S = _styles()
    bl = res["baseline"]; bl_inv = res["baseline_involved"]
    results = res["results"]; eq = res["equilibrium"]
    risk = res["risk"]
    figdir = run_dir / "figures"
    icao = ga.config.AIRPORT_ICAO
    story = []
    story.append(Paragraph(f"Proximity risk after go-arounds — {icao}", S["h1"]))
    wm = res["window_metrics"]
    story.append(Paragraph(
        f"{run_dir.parent.parent.name} / {run_dir.name}  ·  "
        f"{wm['day'].nunique()} days, {len(ev)} go-arounds, "
        f"{len(res['control_windows']) // 2} matched landing controls  ·  "
        f"window {config.WINDOW_MIN:g} min, radius "
        f"{config.VOLUME_RADIUS_NM:g} NM", S["sub"]))

    # verdicts
    prim = results[(results["endpoint"] == PRIMARY_METRIC)
                   & (results["event_set"] == "all")].set_index("scope")
    lines = []
    for scope, name in (("airspace", "Airspace-wide"),
                        ("ga_involved", "Go-around-involved")):
        if scope not in prim.index:
            continue
        r = prim.loc[scope]
        lines.append(
            f"<b>{name} T1 encounters:</b> rate ratio post/pre "
            f"<b>{_f(r['irr'])}</b> [{_f(r['irr_lo'])}, {_f(r['irr_hi'])}]"
            f"; SIR post vs baseline <b>{_f(r['sir_post'])}</b> "
            f"[{_f(r['sir_post_lo'])}, {_f(r['sir_post_hi'])}]; "
            f"SIR pre {_f(r['sir_pre'])} [{_f(r['sir_pre_lo'])}, "
            f"{_f(r['sir_pre_hi'])}]; n = {int(r['n_events'])}; minimum "
            f"detectable IRR {_f(r['mde_irr'])}.")
    for line in lines:
        story.append(Paragraph(line, S["big"]))
        story.append(Spacer(1, 4))

    bad = [t for t, c in res["calibration"].items()
           if not np.isfinite(c.get("ece", np.nan))
           or c["ece"] > config.PROB_MAX_ECE]
    verdict_rows = [["check", "result"],
                    ["baseline (airspace-wide)", bl.verdict],
                    ["baseline (go-around-involved)", bl_inv.verdict]]
    for _, e in eq.iterrows():
        verdict_rows.append([f"equilibrium ({e['scope']})",
                             f"{e['verdict']}: pre in control "
                             f"{100 * e['in_control_share']:.0f}%, SIR_pre "
                             f"{_f(e['sir_pre'])} [{_f(e['sir_pre_ci'][0])}, "
                             f"{_f(e['sir_pre_ci'][1])}], KS p {_f(e['ks_p'])}"])
    verdict_rows.append([
        "ceiling", f"charted {chk['charted_ceiling_ft_agl']:.0f} ft above "
                   f"field; effective {chk['effective_ceiling_ft_agl']:.0f} ft"
                   + (f" (RAISED: {chk['reason']})" if chk["override"]
                      else " (charted used)")])
    verdict_rows.append([
        "probability model",
        "poorly calibrated for " + ", ".join(bad) + " (not used for claims)"
        if bad else "calibrated (ECE within limit)"])
    story.append(Paragraph("Verdicts", S["h2"]))
    story.append(_table(verdict_rows, col_widths=[2.2 * inch, 4.6 * inch]))
    if bl.verdict != "VALID":
        story.append(Spacer(1, 4))
        story.append(Paragraph("Baseline warnings: " + "; ".join(bl.warnings),
                               S["warn"]))
    # conclusions differ between all events and in-control-pre events?
    a = results[(results["endpoint"] == PRIMARY_METRIC)
                & (results["scope"] == "airspace")].set_index("event_set")
    if {"all", "in_control_pre"} <= set(a.index):
        sa = a.loc["all"]; si = a.loc["in_control_pre"]
        def sig(r): return (r["irr_lo"] > 1) or (r["irr_hi"] < 1)
        if sig(sa) != sig(si):
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                "The conclusion on the primary IRR DIFFERS between all events "
                f"(IRR {_f(sa['irr'])} [{_f(sa['irr_lo'])}, {_f(sa['irr_hi'])}]) "
                f"and events with an in-control pre window (IRR {_f(si['irr'])} "
                f"[{_f(si['irr_lo'])}, {_f(si['irr_hi'])}]).", S["warn"]))

    story.append(Paragraph("Ceiling check", S["h2"]))
    story.append(_table([
        ["item", "value"],
        ["charted ceiling", f"{chk['charted_ceiling_ft_msl']:.0f} ft MSL = "
                            f"{chk['charted_ceiling_ft_agl']:.0f} ft above field"],
        ["go-around max height P95 (post window)",
         f"{_f(chk['p95_ga_max_height_ft_agl'], 0)} ft above field "
         f"(n={chk['n_go_arounds']})"],
        ["go-around T1 encounters above charted ceiling",
         f"{100 * chk['spill_share_at_charted']:.1f}% of "
         f"{chk['n_ga_t1_encounters']} (limit {100 * chk['spill_limit']:.0f}%)"],
        ["post-window time beyond radius",
         f"{100 * chk['ga_post_time_beyond_radius_share']:.1f}%"],
        ["effective ceiling", f"{chk['effective_ceiling_ft_agl']:.0f} ft above "
                              f"field ({chk['reason']})"],
    ], col_widths=[2.6 * inch, 4.2 * inch]))

    story.append(Paragraph("Data quality", S["h2"]))
    q = wm["quality"].value_counts()
    story.append(_table([["window quality", "clock windows"]] +
                        [[k, f"{int(v):,}"] for k, v in q.items()] +
                        [["excluded events (bad or partial windows)",
                          f"{int(risk['excluded'].sum())}"]]))

    story.append(PageBreak())
    story.append(Paragraph("Results per endpoint", S["h2"]))
    cols = ["scope", "event_set", "endpoint", "n_events", "sum_pre",
            "sum_post", "irr", "irr_lo", "irr_hi", "wilcoxon_paired_p",
            "sir_post", "sir_post_lo", "sir_post_hi", "wilcoxon_post_p",
            "mde_irr"]
    have = [c for c in cols if c in results.columns]
    rows = [have]
    for _, r in results.iterrows():
        rows.append([_f(r[c]) if not isinstance(r[c], str) else r[c]
                     for c in have])
    story.append(_table(rows))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "IRR: conditional-likelihood rate ratio post/pre with exposure "
        "offset, day-clustered bootstrap CI. SIR: sum observed / sum expected "
        "under the baseline, day-block bootstrap CI. Wilcoxon p-values for "
        "secondary endpoints are Holm-corrected in results_primary.csv. "
        f"Endpoints labelled 'pooled' use an exposure-adjusted Poisson rate "
        f"(fewer than {config.GLM_MIN_EVENTS} baseline events).", S["body"]))

    for name, cap in (("superposed_epoch.png", "Headline: T1 encounter rate "
                       "O/E around the go-around"),
                      ("forest.png", "IRR and SIR per endpoint and scope"),
                      ("equilibrium.png", "Equilibrium of pre windows"),
                      ("control_chart.png", "Control chart of every window"),
                      ("encounter_density.png", "Where encounters happen"),
                      ("ceiling_check.png", "Ceiling check")):
        p = figdir / name
        if p.exists():
            story.append(Paragraph(cap, S["h2"]))
            story.append(_image(p, 6.9 * inch))
    if sens is not None and len(sens) and (figdir / "sensitivity.png").exists():
        story.append(PageBreak())
        story.append(Paragraph("Sensitivity", S["h2"]))
        story.append(_image(figdir / "sensitivity.png", 6.9 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Go-around events", S["h2"]))
    cols = ["event_id", "callsign", "runway", "t0_utc", "pre_T1_any",
            "pre_T1_any_expected", "pre_T1_any_z", "post_T1_any",
            "post_T1_any_expected", "post_T1_any_z", "excluded"]
    have = [c for c in cols if c in risk.columns]
    rows = [[c.replace("T1_any_", "").replace("T1_any", "T1") for c in have]]
    for _, r in risk.iterrows():
        rows.append([f"{r[c]:%Y-%m-%d %H:%M}" if c == "t0_utc"
                     else (_f(r[c]) if not isinstance(r[c], str) else r[c])
                     for c in have])
    story.append(_table(rows))

    story.append(PageBreak())
    story.append(Paragraph("Method in brief", S["h2"]))
    tiers = config.TIERS
    for txt in [
        f"<b>Volume.</b> Cylinder of radius {config.VOLUME_RADIUS_NM:g} NM "
        f"around the airport, from the surface (airborne aircraft only: "
        f"onground false and groundspeed at least {config.AIRBORNE_MIN_GS_KT:.0f} kt) "
        f"to the effective ceiling. Encounters are computed to the ceiling "
        f"plus {config.CEILING_BUFFER_FT:.0f} ft and tagged inside/outside.",
        f"<b>Encounters.</b> Every pair of airborne aircraft is evaluated "
        f"each second: the analytic zone-entry interval (a quadratic in time "
        f"horizontally, linear vertically) says whether the pair is inside a "
        f"tier zone during the step (observed) or enters it within "
        f"{config.T_LOOKAHEAD_S:.0f} s by straight-line projection "
        f"(predicted). Tiers: T1 {tiers['T1'][0]:g} NM / {tiers['T1'][1]:.0f} ft, "
        f"T2 {tiers['T2'][0]:g} NM / {tiers['T2'][1]:.0f} ft, T3 "
        f"{tiers['T3'][0] * ga.FT_PER_NM:.0f} ft / {tiers['T3'][1]:.0f} ft. "
        f"Consecutive flagged seconds form one encounter (gaps up to "
        f"{config.EPISODE_MERGE_GAP_S:.0f} s merged).",
        f"<b>Baseline.</b> Negative binomial regression of encounter counts "
        f"per {config.WINDOW_MIN:g}-min clock window on hour, day type, month, "
        f"runway flow and traffic, with pair-hours as exposure; windows within "
        f"{config.BASELINE_EXCLUSION_MIN:.0f} min of a go-around and bad-quality "
        f"windows excluded; Phase I trimming; out-of-sample calibration.",
        "<b>Comparison.</b> Pre and post windows of each go-around are scored "
        "against the baseline (z-scores, control limits). Post vs pre: rate "
        "ratio with day-clustered bootstrap CI and Wilcoxon signed-rank test. "
        "Post vs baseline: standardised incidence ratio with day-block "
        "bootstrap CI. Go-around-involved encounters are compared with "
        "matched full-stop landings.",
        "<b>Limitations.</b> Proximity risk is a surrogate for safety, not a "
        "collision probability. Surveillance only; straight-line prediction "
        "ignores intent; tier thresholds are regulatory reference values, not "
        "the separation actually required for each pair; go-around intent is "
        "invisible; observational design (association, not causation).",
    ]:
        story.append(Paragraph(txt, S["body"]))
        story.append(Spacer(1, 6))
    out = run_dir / "summary.pdf"
    _doc(out, f"Proximity risk after go-arounds - {icao}").build(story)
    return out


# ---------------------------------------------------- baseline report ----

def baseline_report(bl, bl_inv, calib: dict, em_table: pd.DataFrame,
                    folder: Path) -> Path:
    S = _styles()
    icao = ga.config.AIRPORT_ICAO
    val = folder / "validation"
    story = [Paragraph(f"Baseline report — {icao} / {folder.name}", S["h1"]),
             Paragraph(f"{bl.info.get('n_days')} days, {bl.info.get('n_windows'):,} "
                       f"baseline windows, {bl.info.get('date_first')} to "
                       f"{bl.info.get('date_last')}", S["sub"])]
    story.append(Paragraph(f"<b>VERDICT (airspace-wide): {bl.verdict}</b>",
                           S["big"]))
    for w in bl.warnings:
        story.append(Paragraph(f"- {w}", S["warn"]))
    story.append(Paragraph(f"<b>VERDICT (go-around-involved, matched landing "
                           f"controls): {bl_inv.verdict}</b>", S["big"]))
    for w in bl_inv.warnings:
        story.append(Paragraph(f"- {w}", S["warn"]))

    story.append(Paragraph("Exclusions", S["h2"]))
    log = bl.exclusion_log
    story.append(_table([["reason", "windows"]] + [
        [k.replace("_", " "), _f(v)] for k, v in log.items()
        if not isinstance(v, dict)]))

    story.append(Paragraph("Models and out-of-sample calibration", S["h2"]))
    rows = [["metric", "model", "events", "rate /pair-h [CI]", "folds",
             "95% exceed [CI]", "99% exceed [CI]", "pass", "KS p"]]
    for m, v in bl.validation.items():
        ci = v.get("rate_ci") or [np.nan, np.nan]
        e95 = v.get("exceed95_ci") or [np.nan, np.nan]
        e99 = v.get("exceed99_ci") or [np.nan, np.nan]
        rows.append([
            m, v.get("kind", ""), _f(v.get("n_events")),
            f"{_f(v.get('rate_per_pair_hour'))} [{_f(ci[0])}, {_f(ci[1])}]",
            f"{v.get('n_folds', '-')} ({v.get('fold_kind', '-')})",
            f"{_f(100 * v['exceed95_rate'], 1)}% [{_f(100 * e95[0], 1)}, "
            f"{_f(100 * e95[1], 1)}]" if "exceed95_rate" in v else "-",
            f"{_f(100 * v['exceed99_rate'], 2)}% [{_f(100 * e99[0], 2)}, "
            f"{_f(100 * e99[1], 2)}]" if "exceed99_rate" in v else "-",
            ("yes" if v.get("pass95") and v.get("pass99") else "no")
            if "pass95" in v else "-",
            _f(v.get("pit_ks_p")),
        ])
    story.append(_table(rows))
    story.append(Spacer(1, 4))
    notes = [f"{m}: {'; '.join(v['notes'])}" for m, v in bl.validation.items()
             if v.get("notes")]
    if notes:
        story.append(Paragraph("Model notes: " + " | ".join(notes), S["body"]))

    prim = bl.validation.get(PRIMARY_METRIC, {})
    model = bl.count_models.get(PRIMARY_METRIC)
    if model is not None and model.coef is not None:
        story.append(Paragraph("Primary model coefficients (T1 encounters)",
                               S["h2"]))
        rows = [["term", "coefficient", "rate multiplier"]]
        for n, c in zip(model.coef_names, model.coef):
            rows.append([n, _f(c, 3), _f(np.exp(c), 2)])
        rows.append(["dispersion alpha", _f(model.alpha, 3),
                     f"p = {_f(model.dispersion_p, 3)}"])
        story.append(_table(rows))
    st = prim.get("stability", {})
    if st:
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Stability: estimates within ±{100 * config.STABILITY_PLATEAU_TOL:.0f}% "
            f"of the full-data value from "
            f"{st.get('stable_from_days', 'not reached')} days "
            f"(of {st.get('n_days')}).", S["body"]))
    suff = prim.get("sufficiency_table")
    if suff is not None and len(suff):
        story.append(Paragraph("Data sufficiency per stratum (hour block x "
                               "traffic tercile x flow)", S["h2"]))
        rows = [["stratum", "windows", "T1 encounters", "pair-hours", "flagged"]]
        for _, r in suff.iterrows():
            rows.append([r["stratum"], _f(int(r["n_windows"])),
                         _f(int(r["n_events"])), _f(r["pair_hours"], 1),
                         "yes" if r["flagged"] else ""])
        story.append(_table(rows))
    if len(bl.phase1_removed):
        story.append(Paragraph("Phase I: windows removed", S["h2"]))
        rows = [["metric", "window", "observed", "expected", "P(Y >= y)"]]
        for _, r in bl.phase1_removed.iterrows():
            rows.append([r["metric"], r["window_id"], _f(r[r["metric"]]),
                         _f(r["expected"]), f"{r['p_upper']:.2e}"])
        story.append(_table(rows))

    for name, cap in (("calibration_exceedance.png",
                       "Out-of-sample exceedance vs nominal"),
                      ("calibration_pit_qq.png", "PIT histogram and QQ-plot"),
                      ("stability_curve.png", "Stability curve"),
                      ("probability_model.png",
                       "Probability model: reliability and error growth")):
        p = val / name
        if p.exists():
            story.append(Paragraph(cap, S["h2"]))
            story.append(_image(p, 6.9 * inch))
    c = calib.get("T1", {})
    story.append(Paragraph(
        f"Probability model (T1): Brier {_f(c.get('brier'), 3)}, Brier skill "
        f"{_f(c.get('bss'), 3)}, expected calibration error "
        f"{_f(c.get('ece'), 3)} (limit {config.PROB_MAX_ECE}), "
        f"{_f(c.get('n'))} pair-seconds.", S["body"]))
    out = folder / "baseline_report.pdf"
    _doc(out, f"Baseline report - {icao} - {folder.name}").build(story)
    return out
