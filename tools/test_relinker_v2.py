"""Smoke / regression tests for relinker_v2. Plain asserts, no pytest needed:

    python tools/test_relinker_v2.py          # runs every test_* function, prints PASS/FAIL
    python -m pytest tools/test_relinker_v2.py   # also works if you have pytest

Covers: every bundle in tools/models loads; relink on both repo CSVs; the headline repo-clip numbers;
input-format auto-detection (column-name variants, MOTChallenge txt, headerless, TSV, the repo tracker's
header bug, normalised coords -- all must give the same chains); frame-stride handling; edge cases
(empty input, one detection, string ids, ...); the CLI end to end. All outputs go to a temp directory.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings

import numpy as np
import pandas as pd

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:             # so `pytest` from the repo root finds the modules too
    sys.path.insert(0, TOOLS_DIR)
import relinker_v2 as rl
import repo_data as rd

warnings.filterwarnings("ignore")
REPO_CSVS = {"parking": os.path.join(rd.REPO_ROOT, "better_tests", "tracksid3duplicate.csv"),
             "airport": os.path.join(rd.REPO_ROOT, "better_tests", "airport_tracks.csv")}
TMP = tempfile.mkdtemp(prefix="relinker_v2_tests_")
AIRPORT = rd.SCENES["airport"]
_REF = {}


def _chains_sig(chains):
    return sorted(map(tuple, chains))


def _airport_reference():
    """Chains of the repo airport CSV with the default model (the reference for the format tests)."""
    if "sig" not in _REF:
        _, _, chains = rl.relink(AIRPORT["raw"], AIRPORT["fps"], AIRPORT["W"], AIRPORT["H"])
        _REF["sig"] = _chains_sig(chains)
    return _REF["sig"]


def _relink_file(path, **kw):
    df = rl.read_tracks(path, **kw)
    _, _, chains = rl.relink(df, AIRPORT["fps"], AIRPORT["W"], AIRPORT["H"])
    return df, _chains_sig(chains)


# ------------------------------------------------------------------------------------------
def test_models_load():
    paths = sorted(glob.glob(os.path.join(rd.MODELS_DIR, "*.joblib")))
    assert len(paths) >= 3, f"expected the three relinker_v2 bundles in {rd.MODELS_DIR}"
    X = np.zeros((2, len(rl.FEATURES["norm+"])))
    for p in paths:
        b = rl.load_model(p)
        assert {"model", "feature_cols", "config"} <= set(b), p
        assert list(b["feature_cols"]) == list(rl.FEATURES[b["config"]["feature_set"]]), p
        assert b["model"].predict_proba(X[:, :len(b["feature_cols"])]).shape == (2, 2), p


def test_relink_repo_csvs():
    for sc, path in REPO_CSVS.items():
        S = rd.SCENES[sc]
        df = rl.read_tracks(path)
        links, t2v, chains = rl.relink(df, S["fps"], S["W"], S["H"])
        n_tracklets = rl.split_into_tracklets(df, 6)["tracklet_id"].nunique()
        assert len(t2v) == n_tracklets, (sc, len(t2v), n_tracklets)          # every tracklet mapped
        assert sorted(t2v.values()) == sorted(v for v, c in enumerate(chains) for _ in c)
        assert 0 < len(chains) < df["track_id"].nunique(), sc                  # fewer vehicles than ids
        out = rl.apply_relink(df, t2v)
        assert len(out) == len(df) and (out["vehicle_id"] >= 0).all(), sc


def test_repo_clip_numbers():
    """Headline numbers of the frozen models (see eval_repo_clips.py)."""
    import eval_repo_clips as ev
    got = {(r["scene"], r["kind"]): r["idf1"] for r in ev.evaluate(rd.MODELS_DIR, boot=0)
           if r["method"] != "tracklets (no relink)"}
    want = {("parking", "-"): 0.885, ("parking", "cross-scene"): 0.960, ("parking", "IN-SAMPLE"): 0.975,
            ("airport", "-"): 0.736, ("airport", "cross-scene"): 0.795, ("airport", "IN-SAMPLE"): 0.945}
    for k, v in want.items():
        assert abs(got[k] - v) < 1e-9, (k, got[k], v)


def test_format_variants():
    raw = AIRPORT["raw"]
    ref = _airport_reference()
    w, h = raw.x2 - raw.x1, raw.y2 - raw.y1
    cases = {}
    cases["repo csv (xyxy)"] = (raw, {}, ".csv", {})
    mot = pd.DataFrame({"f": raw.frame, "id": raw.track_id, "l": raw.x1, "t": raw.y1, "w": w, "h": h,
                        "c": 1.0, "x": -1, "y": -1, "z": -1})
    cases["MOTChallenge txt, headerless (tlwh)"] = (mot, {"header": False}, ".txt", {})
    cases["MOT txt + id=-1 rows"] = (pd.concat([mot, mot.head(5).assign(id=-1)]), {"header": False}, ".txt", {})
    cases["Frame,ID,X,Y,Width,Height"] = (pd.DataFrame({"Frame": raw.frame, "ID": raw.track_id, "X": raw.x1,
                                                        "Y": raw.y1, "Width": w, "Height": h, "score": 0.9,
                                                        "label": "car"}), {}, ".csv", {})
    cases["frame_id,track_id,xc,yc,w,h"] = (pd.DataFrame({"frame_id": raw.frame, "track_id": raw.track_id,
                                                          "xc": (raw.x1 + raw.x2) / 2, "yc": (raw.y1 + raw.y2) / 2,
                                                          "w": w, "h": h}), {}, ".csv", {})
    cases["x1..y2 names holding w,h"] = (pd.DataFrame({"frame": raw.frame, "track_id": raw.track_id, "x1": raw.x1,
                                                       "y1": raw.y1, "x2": w, "y2": h}), {}, ".csv", {})
    nor = raw[["frame", "track_id"]].copy()
    for c, d in [("x1", AIRPORT["W"]), ("y1", AIRPORT["H"]), ("x2", AIRPORT["W"]), ("y2", AIRPORT["H"])]:
        nor[c] = raw[c] / d
    cases["normalised coords"] = (nor, {}, ".csv", {"frame_w": AIRPORT["W"], "frame_h": AIRPORT["H"]})
    cases["TSV, shuffled rows"] = (raw.sample(frac=1, random_state=0), {"sep": "\t"}, ".tsv", {})
    for i, (name, (d, to_csv_kw, ext, read_kw)) in enumerate(cases.items()):
        p = os.path.join(TMP, f"fmt{i}{ext}")
        d.to_csv(p, index=False, **to_csv_kw)
        _, sig = _relink_file(p, **read_kw)
        assert sig == ref, f"{name}: chains differ from the reference"
    # the repo tracker's header bug: 9 header names, 10 fields per row
    p = os.path.join(TMP, "headerbug.csv")
    with open(p, "w") as fh:
        fh.write("frame,track_id,x1,y1,x2,y2,velocity_x,velocity_yconf,class\n")
        for r in raw.itertuples(index=False):
            fh.write(f"{r.frame},{r.track_id},{r.x1},{r.y1},{r.x2},{r.y2},{r.velocity_x},{r.velocity_y},{r.conf},{r[9]}\n")
    _, sig = _relink_file(p)
    assert sig == ref, "repo header bug: chains differ from the reference"


def test_stride():
    """Every 3rd frame with the original frame numbers: the split gap scales with the frame step."""
    raw = AIRPORT["raw"]
    st = rl.normalize_tracks(raw[raw.frame % 3 == 0])
    assert rl.frame_step(st) == 3
    scaled = rl.split_into_tracklets(st, 6, scale_by_step=True)["tracklet_id"].nunique()
    unscaled = rl.split_into_tracklets(st, 6, scale_by_step=False)["tracklet_id"].nunique()
    assert scaled < unscaled, (scaled, unscaled)


def _box(f, tid, x, y=300, w=80, h=50):
    return {"frame": f, "track_id": tid, "x1": x, "y1": y, "x2": x + w, "y2": y + h}


def test_edge_cases():
    two = [_box(f, 1, 100 + 3 * f) for f in range(1, 40)] + [_box(f, 2, 100 + 3 * f) for f in range(50, 90)]
    cases = {
        "two fragments of one car": (pd.DataFrame(two), {}),
        "string ids": (pd.DataFrame([{**r, "track_id": f"car_{r['track_id']}"} for r in two]), {}),
        "one track": (pd.DataFrame([_box(f, 7, 10 + f) for f in range(1, 30)]), {}),
        "one detection": (pd.DataFrame([_box(5, 7, 10)]), {}),
        "no candidate pairs": (pd.DataFrame([_box(f, 1, 10) for f in range(1, 10)]
                                            + [_box(f, 2, 1100, 600) for f in range(500, 510)]), {}),
        "overlapping tracks only": (pd.DataFrame([_box(f, 1, 10) for f in range(1, 50)]
                                                 + [_box(f, 2, 400) for f in range(1, 50)]), {}),
        "empty": (pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2"]), {}),
        "float frames + unsorted": (pd.DataFrame(two).sample(frac=1, random_state=0)
                                    .assign(frame=lambda d: d.frame.astype(float)), {}),
        "MOT-style column names": (pd.DataFrame([{"frame": r["frame"], "id": r["track_id"], "bb_left": r["x1"],
                                                  "bb_top": r["y1"], "bb_width": 80, "bb_height": 50} for r in two]), {}),
        "greedy assembly": (pd.DataFrame(two), {"assembly": "greedy"}),
        "mincostflow assembly": (pd.DataFrame(two), {"assembly": "mincostflow"}),
        "merge none": (pd.DataFrame(two), {"merge": "none"}),
    }
    for name, (df, kw) in cases.items():
        links, t2v, chains = rl.relink(df, 30.0, 1280, 720, **kw)
        out = rl.apply_relink(df if "frame" in df.columns and "x1" in df.columns else rl.normalize_tracks(df), t2v)
        assert len(out) == len(df), name
        assert (out["vehicle_id"] >= 0).all(), name
        assert len(chains) == len(set(t2v.values())), name
        if name in ("two fragments of one car", "string ids", "float frames + unsorted", "MOT-style column names"):
            assert len(chains) == 1, f"{name}: the two fragments should be one vehicle, got {chains}"
        if name == "empty":
            assert len(links) == 0 and t2v == {} and chains == [], name
    # empty file with only a header line, read from disk
    p = os.path.join(TMP, "empty.csv")
    pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2"]).to_csv(p, index=False)
    links, t2v, chains = rl.relink(rl.read_tracks(p), 30.0, 1280, 720)
    assert len(links) == 0 and t2v == {} and chains == []


def test_cli():
    raw = AIRPORT["raw"]
    mot = pd.DataFrame({"f": raw.frame, "id": raw.track_id, "l": raw.x1, "t": raw.y1, "w": raw.x2 - raw.x1,
                        "h": raw.y2 - raw.y1, "c": 1.0, "x": -1, "y": -1, "z": -1})
    src = os.path.join(TMP, "cli_input.txt")
    mot.to_csv(src, index=False, header=False)
    for merge in [None, "none"]:
        prefix = os.path.join(TMP, "cli_out", f"airport_{merge or 'default'}")
        cmd = [sys.executable, os.path.join(TOOLS_DIR, "relinker_v2.py"), src, "--fps", str(AIRPORT["fps"]),
               "--width", str(AIRPORT["W"]), "--height", str(AIRPORT["H"]), "--out-prefix", prefix]
        if merge:
            cmd += ["--merge", merge]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=TMP)
        assert r.returncode == 0, r.stderr[-2000:]
        for suffix in ["_links.csv", "_relinked.csv", "_chains.json"]:
            assert os.path.exists(prefix + suffix), prefix + suffix
        with open(prefix + "_chains.json") as fh:
            ch = json.load(fh)
        if merge is None:
            assert _chains_sig(ch["chains"]) == _airport_reference()
        rel = pd.read_csv(prefix + "_relinked.csv")
        assert len(rel) == len(raw) and (rel["vehicle_id"] >= 0).all()


# ------------------------------------------------------------------------------------------
def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
