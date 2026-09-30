"""
Remove event plots whose event no longer exists in the current tables
(outcomes can change when classifier rules are refined; plot filenames are
outcome-prefixed, so stale files must be pruned rather than overwritten).

Usage:
    python -m goaround_pipeline.clean_orphan_plots
"""

from __future__ import annotations

import pandas as pd

from . import config
from .pipeline import PLOTTED_OUTCOMES


def main() -> None:
    df = pd.read_csv(config.OUTPUT_DIR / "all_approaches.csv")
    df = df[df["outcome"].isin(PLOTTED_OUTCOMES)]
    t = pd.to_datetime(df["t_low_utc"], utc=True)
    valid = {
        f"{o}_{ts:%Y%m%d_%H%M%S}Z_{cs}_{rw}.png"
        for o, ts, cs, rw in zip(df["outcome"], t, df["callsign"],
                                 df["runway"].fillna(""))
    }
    plot_dir = config.OUTPUT_DIR / "plots"
    removed = kept = 0
    for p in plot_dir.glob("*.png"):
        if p.name in valid:
            kept += 1
        else:
            p.unlink()
            removed += 1
    print(f"kept {kept}, removed {removed} stale plots "
          f"({len(valid)} events expected)")


if __name__ == "__main__":
    main()
