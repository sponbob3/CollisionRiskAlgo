"""
Proximity-risk pipeline: does proximity risk in the terminal airspace
change after a go-around? Method and build specification: FRAMEWORK.md.
Entry point: run_proximity.py.
"""

__version__ = "0.1.0"


def code_version() -> dict:
    """Version of this code for provenance: the release version, the git
    commit, and whether tracked files had uncommitted changes. Commit and
    changes read "unknown" when the code is not a git checkout (e.g. a
    ZIP download) or git is not installed; a run never fails over it."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    info = {"version": __version__, "commit": "unknown", "modified": "unknown"}
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12",
                              "HEAD"], capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            info["commit"] = out.stdout.strip()
            st = subprocess.run(["git", "-C", str(root), "status",
                                 "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10)
            if st.returncode == 0:
                info["modified"] = "yes" if st.stdout.strip() else "no"
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def code_version_text() -> str:
    """e.g. 'v0.1.0 (commit 3f2a9c1d0b7e)' or '... with local changes'."""
    v = code_version()
    text = f"v{v['version']} (commit {v['commit']}"
    if v["modified"] == "yes":
        text += ", with local changes"
    return text + ")"
