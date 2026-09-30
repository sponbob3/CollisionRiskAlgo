"""
Stage 4: run the full pipeline over daily files and build the event tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .approaches import detect_approaches, truncated_final
from .classify import ClassifiedApproach, classify_approach
from .loading import load_day, split_into_legs


def _chain_postprocess(leg_results: list[ClassifiedApproach]) -> None:
    """Link consecutive approaches on the same leg: tag what followed each
    one, and fold 'unresolved' fragments followed shortly by another aligned
    segment into a single continued approach."""
    leg_results.sort(key=lambda r: r.approach.start)
    for i, r in enumerate(leg_results):
        if i + 1 < len(leg_results):
            nxt = leg_results[i + 1]
            gap = (nxt.approach.start - r.approach.end).total_seconds()
            r.next_gap_s = gap
            r.next_action = "reapproach"
            if (r.outcome == "unresolved"
                    and gap <= config.CONTINUED_APPROACH_MAX_GAP_S):
                r.outcome = "continued_approach"
        else:
            r.next_action = "leg_end"


def process_day(path: Path) -> list[ClassifiedApproach]:
    df = load_day(path)
    results: list[ClassifiedApproach] = []
    for leg in split_into_legs(df):
        apps = detect_approaches(leg)
        fallback = truncated_final(leg, apps)
        if fallback is not None:
            apps.append(fallback)
        leg_results = [classify_approach(app) for app in apps]
        _chain_postprocess(leg_results)
        results.extend(leg_results)
    return results


def results_to_frame(results: list[ClassifiedApproach]) -> pd.DataFrame:
    rows = []
    for r in results:
        a = r.approach
        rows.append({
            "leg_id": a.leg_id,
            "icao24": a.icao24,
            "callsign": a.callsign,
            "runway": a.runway,
            "approach_start_utc": a.start,
            "approach_end_utc": a.end,
            "outcome": r.outcome,
            "touchdown": r.touchdown,
            "coverage_truncated": r.coverage_truncated,
            "min_agl_ft": round(r.min_agl_ft, 1),
            "t_low_utc": r.t_low,
            "lat_low": round(r.lat_low, 6),
            "lon_low": round(r.lon_low, 6),
            "level_low_duration_s": round(r.level_low_duration_s, 1),
            "climb_start_utc": r.climb_start,
            "init_agl_ft": round(r.init_agl_ft, 1),
            "init_dist_thr_nm": round(r.init_dist_thr_nm, 3),
            "peak_climb_fpm": round(r.peak_climb_fpm, 0),
            "alt_regain_ft": round(r.alt_regain_ft, 1),
            "next_action": r.next_action,
            "next_gap_s": r.next_gap_s,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["t_local"] = (
            pd.to_datetime(df["t_low_utc"], utc=True)
            .dt.tz_convert(config.LOCAL_TZ)
        )
    return df


PLOTTED_OUTCOMES = ("go_around", "ga_ambiguous", "low_approach")
GA_OUTCOMES = ("go_around", "ga_ambiguous")
NON_ATTEMPTS = ("continued_approach",)


def data_files(data_dir: Path) -> list[Path]:
    """Daily files in a dataset folder; when the same day exists as both
    parquet and csv, the parquet wins."""
    by_stem: dict[str, Path] = {}
    for f in sorted(data_dir.glob("*.csv")) + sorted(
            data_dir.glob("*.parquet")):
        by_stem[f.stem] = f  # parquet sorted second -> overrides csv
    return [by_stem[k] for k in sorted(by_stem)]


def run(data_dir: Path, out_dir: Path, plots: bool = True,
        limit: int | None = None) -> pd.DataFrame:
    """Process every daily file in data_dir; write event tables (and
    per-event plots) into out_dir. Returns the full approach table."""
    files = data_files(Path(data_dir))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no .parquet/.csv files in {data_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    if plots:
        from .viz import plot_event
        plot_dir.mkdir(exist_ok=True)

    # accumulate table rows only - ClassifiedApproach objects hold their
    # leg's full trajectory (for plotting), which must not outlive the day
    day_frames: list[pd.DataFrame] = []
    df = pd.DataFrame()
    for i, f in enumerate(files):
        day_results = process_day(f)
        if plots:
            for r in day_results:
                if r.outcome in PLOTTED_OUTCOMES:
                    name = (f"{r.outcome}_{r.t_low:%Y%m%d_%H%M%S}Z_"
                            f"{r.approach.callsign}_{r.approach.runway}.png")
                    if (plot_dir / name).exists():
                        continue
                    try:
                        plot_event(r, plot_dir / name)
                    except Exception as e:
                        print(f"  plot failed {name}: {e}", flush=True)
        print(f"[{i + 1}/{len(files)}] {f.name}: "
              f"{len(day_results)} approaches", flush=True)
        frame = results_to_frame(day_results)
        if len(frame):
            day_frames.append(frame)
        del day_results
        # checkpoint the tables after every day so an interrupted run still
        # leaves complete outputs for the days processed so far
        # (silence pandas' all-NA-column concat FutureWarning: columns like
        # climb_start_utc are legitimately all-NA on some days)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat(day_frames, ignore_index=True)
        df.to_csv(out_dir / "all_approaches.csv", index=False)
        ga = df[df["outcome"].isin(GA_OUTCOMES)]
        ga.to_csv(out_dir / "go_around_events.csv", index=False)
    return df


def write_summaries(df: pd.DataFrame, out_dir: Path) -> str:
    """Exact-numbers summary: outcome counts and percentages, per-runway
    and per-month tables. Written as CSVs and returned as console text."""
    out_dir = Path(out_dir)
    n_all = len(df)
    attempts = df[~df["outcome"].isin(NON_ATTEMPTS)]
    n_att = len(attempts)
    n_ga = int((df["outcome"] == "go_around").sum())

    # ---- outcomes ----
    vc = df["outcome"].value_counts()
    outcomes = pd.DataFrame({
        "outcome": vc.index,
        "count": vc.values,
        "pct_of_segments": (100 * vc.values / n_all).round(3),
        "pct_of_attempts": [
            round(100 * v / n_att, 3)
            if o not in NON_ATTEMPTS else float("nan")
            for o, v in vc.items()
        ],
    })
    outcomes.to_csv(out_dir / "summary_outcomes.csv", index=False)

    # ---- per runway (attempts only) ----
    rw = attempts[attempts["runway"].notna() & (attempts["runway"] != "")]
    by_rwy = rw.groupby("runway").agg(
        attempts=("outcome", "size"),
        go_around=("outcome", lambda s: int((s == "go_around").sum())),
        ga_ambiguous=("outcome", lambda s: int((s == "ga_ambiguous").sum())),
        touch_and_go=("outcome", lambda s: int((s == "touch_and_go").sum())),
        low_approach=("outcome", lambda s: int((s == "low_approach").sum())),
        full_stop=("outcome", lambda s: int((s == "full_stop").sum())),
    ).sort_values("attempts", ascending=False)
    by_rwy["ga_per_1000"] = (1000 * by_rwy["go_around"]
                             / by_rwy["attempts"]).round(2)
    by_rwy.to_csv(out_dir / "summary_by_runway.csv")

    # ---- per month (attempts only) ----
    t_local = pd.to_datetime(attempts["t_low_utc"], utc=True,
                             format="mixed").dt.tz_convert(config.LOCAL_TZ)
    month = t_local.dt.strftime("%Y-%m")
    by_month = attempts.assign(month=month).groupby("month").agg(
        attempts=("outcome", "size"),
        go_around=("outcome", lambda s: int((s == "go_around").sum())),
        ga_ambiguous=("outcome", lambda s: int((s == "ga_ambiguous").sum())),
    )
    by_month["ga_per_1000"] = (1000 * by_month["go_around"]
                               / by_month["attempts"]).round(2)
    by_month.to_csv(out_dir / "summary_by_month.csv")

    headline = (f"segments: {n_all}   independent attempts: {n_att}   "
                f"go-arounds: {n_ga}   "
                f"rate: {1000 * n_ga / n_att:.2f} per 1,000 attempts")
    _summary_pdf(out_dir, headline, outcomes, by_rwy, by_month)

    lines = [
        headline,
        "",
        outcomes.to_string(index=False),
        "",
        "BY RUNWAY",
        by_rwy.to_string(),
        "",
        "BY MONTH",
        by_month.to_string(),
    ]
    return "\n".join(lines)


def _summary_pdf(out_dir: Path, headline: str, outcomes: pd.DataFrame,
                 by_rwy: pd.DataFrame, by_month: pd.DataFrame) -> None:
    """summary.pdf: the exact summary tables, nothing else."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    ink = colors.HexColor("#0b0b0b")
    rule = colors.HexColor("#d8d7d2")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=14,
                        textColor=ink, alignment=0, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9.5,
                         textColor=colors.HexColor("#52514e"), spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11,
                        textColor=ink, spaceBefore=12, spaceAfter=4)

    def table(frame: pd.DataFrame, index_name: str | None = None):
        f = frame.reset_index() if index_name else frame
        data = [list(f.columns)] + [
            ["" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else v)
             for v in row]
            for row in f.itertuples(index=False)
        ]
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), ink),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, ink),
            ("LINEBELOW", (0, 1), (-1, -2), 0.25, rule),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        return t

    story = [
        Paragraph(f"Approach outcomes — {config.AIRPORT_ICAO}", h1),
        Paragraph(f"{out_dir.parent.name} / {out_dir.name}   ·   "
                  f"{headline}", sub),
        Paragraph("Outcomes", h2), table(outcomes),
        Paragraph("By runway", h2), table(by_rwy, "runway"),
        Paragraph("By month", h2), table(by_month, "month"),
        Spacer(1, 8),
    ]
    doc = SimpleDocTemplate(
        str(out_dir / "summary.pdf"), pagesize=letter,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=f"Approach outcomes - {config.AIRPORT_ICAO} - "
              f"{out_dir.parent.name} {out_dir.name}",
    )
    doc.build(story)
