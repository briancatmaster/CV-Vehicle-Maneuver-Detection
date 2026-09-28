"""relinker_v2 -- tracker-agnostic, scale/fps-normalised tracklet relinker.

Input: any per-frame track table with a frame number, a track id and a box.
Output: pairwise link scores, a tracklet -> vehicle mapping and the vehicle chains.

    import relinker_v2 as rl
    df = rl.read_tracks("tracks.csv")                 # CSV / MOTChallenge txt, auto-detected
    links, t2v, chains = rl.relink(df, fps=29.97, frame_w=1920, frame_h=1080)
    out = rl.apply_relink(df, t2v)                     # adds tracklet_id + vehicle_id per row

CLI:
    python tools/relinker_v2.py tracks.csv --fps 29.97 --width 1920 --height 1080 --out-prefix out/run1
    python tools/relinker_v2.py tracks.txt --video clip.mp4          # fps / size read from the video

Only `frame, track_id` and a box are used. Kalman velocities, conf and class columns are ignored
(velocities are re-estimated from box centres, so every tracker is treated the same way).

Pipeline (defaults come from the model bundle, see DEFAULT_CONFIG):
  1. split each track id into tracklets at frame gaps > split_gap frames (6, multiplied by the
     median frame step when the input is frame-subsampled);
  2. candidate pairs A -> B (B starts after A ends): gap <= max_gap_s seconds and
     centre distance <= gate_a + gate_b * gap_seconds  (distances in box diagonals);
  3. scale-free features (seconds, box diagonals, box-diagonals per second);
  4. every candidate pair (same-ID continuations included) is scored by the random forest
     (optional rule mode: consecutive same-id tracklets get a fixed prior instead);
  5. chain assembly: one-to-one assignment (Hungarian on the predecessor/successor bipartite
     graph, weight log(p/threshold)), then optional merge of chains that share a track id.

API details
-----------
relink(df, fps, frame_w, frame_h, model_path=None, threshold=None, assembly=None, merge=None)
  df        any DataFrame (or use read_tracks(path)); column variants are matched case-insensitively
            (frame/frame_id/frame_idx/img_id..., track_id/id/tid/object_id/tracker_id...,
            x1/xmin/left/bb_left..., w/width/bb_width..., cx/xc/x_center...). Box format auto:
            x1,y1,x2,y2 -> xyxy (treated as tlwh if x2<=x1 or y2<=y1 in >50% rows); left/top+w/h or
            bare x,y,w,h -> tlwh (MOT); cx,cy,w,h -> centre. Override with box_format= in read_tracks /
            normalize_tracks. Headerless files = MOTChallenge frame,id,bb_left,bb_top,bb_width,bb_height,
            conf,x,y,z. id<0 rows dropped; normalised coords (<=1.5) scaled by frame_w/frame_h;
            duplicate (frame,id) rows keep the first; extra columns (conf, class, velocities) ignored.
  fps       fps of the ORIGINAL frame numbering (stride-k output with original frame numbers ->
            pass the source video fps). The tracklet split gap (6 frames) is multiplied by the
            detected median frame step.
  returns   links_df: every candidate pair (id1, id2, p, accepted, features, same_id, next_same_id)
            tracklet_to_vehicle: {"<track_id>_<k>": vehicle int (0.. by first appearance)}, all tracklets
            chains: list of time-ordered tracklet-id lists (one per vehicle, singletons included)
  threshold/assembly/merge: None -> values stored in the model bundle (0.5 / 'hungarian' / 'sameid').
apply_relink(df, tracklet_to_vehicle) -> rows + tracklet_id + vehicle_id (-1 = not in mapping).

Models (tools/models/, joblib bundles carrying model + feature list + config; sklearn 1.8.0 pickles):
  relinker_v2_both.joblib     parking + airport (default; IN-SAMPLE on those two clips)
  relinker_v2_parking.joblib  parking only -> use on airport for honest cross-scene numbers
  relinker_v2_airport.joblib  airport only -> use on parking for honest cross-scene numbers
Bundle config: FD velocity 0.25 s, norm gate a=3 b=3 10 s, 'norm+' features, successor-only labels
(non-successor same-vehicle pairs dropped), no same-id rule, threshold 0.5, Hungarian, 'sameid' merge,
RF 200 trees depth 5 leaf 2 split 5 balanced. train_relinker_v2.py rebuilds these bundles from the repo
clips (identical predictions); eval_repo_clips.py reproduces the numbers below. Measured on the repo GT
with the 282_2/282_3 label fix (parking 282_2/282_3 = static car, removed from the White SUV group;
labeled tracklets, detection IDF1): cross-scene parking 0.960 / airport 0.795 vs raw DeepSORT ids
0.885 / 0.736. The config and threshold were chosen on these same two clips.

Known limitation of the default 'sameid' merge (held-out test, 8 unseen UA-DETRAC sequences, frozen
v2.0 models): on DeepSORT output (repo settings) the default config LOWERED IDF1 by 0.024 (95% CI
-0.040 to -0.008; 6 of 8 sequences hurt), because DeepSORT sometimes hands one track id from one car
to another and the 'sameid' merge then glues both cars' chains together. With the same model and
merge='none', DeepSORT improved by +0.024 (7 of 8 sequences) -- a post-hoc check, not pre-registered;
it is not free either (one DeepSORT sequence dropped 0.046, BoT-SORT -0.0005). On ByteTrack and
BoT-SORT output relinking was neutral (+0.0005 / +0.0007 IDF1); on deliberately fragmented ByteTrack
output it helped every sequence (+0.060 IDF1, 8 of 8). So: for DeepSORT output consider --merge none.
A checked same-id join (join only when the link itself passes the model / a motion check) is planned
for v2.1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

__version__ = "2.0"

DEFAULT_CONFIG = {
    "split_gap": 6,             # frames (at the input's own frame step), as in relinker_script.py
    "vel_source": "fd",         # 'fd' (finite differences of box centres) or 'kalman' (velocity_x/y columns)
    "fd_window_s": 0.25,        # FD velocity window in seconds (None -> use fd_window_dets)
    "fd_window_dets": 5,
    "gate": {"type": "norm", "max_gap_s": 10.0, "a": 3.0, "b": 3.0},
    "feature_set": "norm+",
    "same_id_rule": False,      # True: consecutive tracklets of one id get p = same_id_p instead of the model
    "same_id_p": 0.95,
    "threshold": 0.5,
    "assembly": "hungarian",    # 'hungarian' | 'greedy' | 'mincostflow'
    "merge": "sameid",          # 'none' | 'flicker2' | 'sameid'
}

# ----------------------------------------------------------------------------------------------
# Input normalisation
# ----------------------------------------------------------------------------------------------
_ALIASES = {
    "frame": ["frame", "frame_id", "frameid", "frame_idx", "frame_index", "frame_no", "frame_num",
              "frame_number", "framenum", "img_id", "image_id", "fn", "t", "timestep", "time_step"],
    "track_id": ["track_id", "trackid", "track", "id", "tid", "object_id", "objectid", "obj_id",
                 "target_id", "tracker_id", "track_idx", "identity", "vehicle_id"],
    "x1": ["x1", "xmin", "x_min", "left", "bb_left", "bbox_left", "l", "xtl", "x_tl"],
    "y1": ["y1", "ymin", "y_min", "top", "bb_top", "bbox_top", "ytl", "y_tl"],
    "x2": ["x2", "xmax", "x_max", "right", "bb_right", "bbox_right", "r", "xbr", "x_br"],
    "y2": ["y2", "ymax", "y_max", "bottom", "bb_bottom", "bbox_bottom", "b", "ybr", "y_br"],
    "w": ["w", "width", "bb_width", "bbox_width", "bw", "box_w"],
    "h": ["h", "height", "bb_height", "bbox_height", "bh", "box_h"],
    "cx": ["cx", "xc", "x_c", "x_center", "center_x", "centre_x", "xcenter", "x_centre", "ctr_x"],
    "cy": ["cy", "yc", "y_c", "y_center", "center_y", "centre_y", "ycenter", "y_centre", "ctr_y"],
    "x": ["x"], "y": ["y"],
    "velocity_x": ["velocity_x", "vx", "vel_x"], "velocity_y": ["velocity_y", "vy", "vel_y"],
}
MOT_COLS = ["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]


def _norm_name(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_tracks(path, box_format="auto", **kw):
    """Read a track file (CSV with header, headerless CSV, or MOTChallenge txt) and return
    a DataFrame with canonical columns frame, track_id, x1, y1, x2, y2 (+ velocity_x/y if present).
    Extra columns are dropped. See normalize_tracks for box_format."""
    with open(path, "r") as fh:
        first = fh.readline()
    sep = "," if first.count(",") >= first.count("\t") else "\t"
    if sep == "\t" and first.count("\t") == 0:
        sep = r"\s+"
    toks = [t for t in (first.strip().split(",") if sep == "," else first.split()) if t != ""]
    headerless = len(toks) > 0 and all(_is_number(t) for t in toks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # tolerate the repo tracker's 9-name / 10-field header bug
        if headerless:
            raw = pd.read_csv(path, header=None, sep=sep, engine="python" if sep != "," else "c")
            names = MOT_COLS[: raw.shape[1]] + [f"extra{i}" for i in range(max(0, raw.shape[1] - 10))]
            raw.columns = names
            if box_format == "auto":
                box_format = "tlwh"          # MOTChallenge convention
        else:
            raw = pd.read_csv(path, sep=sep, index_col=False, engine="python" if sep != "," else "c")
    return normalize_tracks(raw, box_format=box_format, **kw)


def normalize_tracks(raw, box_format="auto", frame_w=None, frame_h=None, verbose=False):
    """Map column-name variants to frame, track_id, x1, y1, x2, y2.

    box_format: 'auto' | 'xyxy' | 'tlwh' | 'cxcywh'. In auto mode:
      x1,y1,x2,y2-like names -> xyxy (but if x2<=x1 or y2<=y1 in >50% of rows, the last two are
      treated as width/height, i.e. tlwh); left/top + w/h -> tlwh; cx,cy + w/h -> cxcywh;
      bare x,y + w/h -> tlwh (MOT convention; pass box_format='cxcywh' for YOLO-style centres).
    Normalised coordinates (all <= 1.5) are scaled by frame_w/frame_h when those are given.
    Rows with track_id < 0 (MOT detections without identity) are dropped.
    """
    df = raw.copy()
    df.columns = [_norm_name(c) for c in df.columns]
    found = {}
    for canon, alist in _ALIASES.items():
        for a in alist:
            if a in df.columns and a not in found.values():
                found[canon] = a
                break
    if "frame" not in found or "track_id" not in found:
        raise ValueError(f"need frame and track id columns; got {list(raw.columns)}")
    out = pd.DataFrame({"frame": pd.to_numeric(df[found["frame"]], errors="coerce"),
                        "track_id": df[found["track_id"]]})
    g = lambda k: pd.to_numeric(df[found[k]], errors="coerce").astype(float)
    has = lambda *ks: all(k in found for k in ks)
    fmt = box_format
    if fmt == "auto":
        if has("x1", "y1", "x2", "y2"):
            x1, y1, x2, y2 = g("x1"), g("y1"), g("x2"), g("y2")
            bad = ((x2 <= x1) | (y2 <= y1)).mean()
            fmt = "tlwh_from_xyxy_names" if bad > 0.5 else "xyxy"
        elif has("x1", "y1", "w", "h"):
            fmt = "tlwh"
        elif has("cx", "cy", "w", "h"):
            fmt = "cxcywh"
        elif has("x", "y", "w", "h"):
            fmt = "tlwh"
        else:
            raise ValueError(f"cannot find a box in columns {list(raw.columns)}")
    if fmt == "xyxy":
        out["x1"], out["y1"], out["x2"], out["y2"] = g("x1"), g("y1"), g("x2"), g("y2")
    elif fmt == "tlwh_from_xyxy_names":
        out["x1"], out["y1"] = g("x1"), g("y1")
        out["x2"], out["y2"] = g("x1") + g("x2"), g("y1") + g("y2")
        fmt = "tlwh"
    elif fmt == "tlwh":
        L = g("x1") if "x1" in found else g("x")
        T = g("y1") if "y1" in found else g("y")
        out["x1"], out["y1"], out["x2"], out["y2"] = L, T, L + g("w"), T + g("h")
    elif fmt == "cxcywh":
        cx = g("cx") if "cx" in found else g("x")
        cy = g("cy") if "cy" in found else g("y")
        out["x1"], out["y1"] = cx - g("w") / 2, cy - g("h") / 2
        out["x2"], out["y2"] = cx + g("w") / 2, cy + g("h") / 2
    else:
        raise ValueError(f"unknown box_format {box_format}")
    for k in ("velocity_x", "velocity_y"):
        if k in found:
            out[k] = pd.to_numeric(df[found[k]], errors="coerce")
    out = out.dropna(subset=["frame", "track_id", "x1", "y1", "x2", "y2"])
    tid_num = pd.to_numeric(out["track_id"], errors="coerce")
    if tid_num.notna().all():
        out = out[tid_num >= 0]
        out["track_id"] = tid_num[tid_num >= 0].astype(np.int64)
    out["frame"] = out["frame"].round().astype(np.int64)
    if frame_w and frame_h and out[["x1", "y1", "x2", "y2"]].abs().max().max() <= 1.5:
        out[["x1", "x2"]] *= frame_w
        out[["y1", "y2"]] *= frame_h
    # one row per (frame, track_id)
    out = out.drop_duplicates(subset=["frame", "track_id"], keep="first")
    out = out.sort_values(["track_id", "frame"]).reset_index(drop=True)
    out.attrs["box_format"] = fmt
    out.attrs["columns_used"] = found
    if verbose:
        print(f"[relinker_v2] box format={fmt}, columns={found}", file=sys.stderr)
    return out


# ----------------------------------------------------------------------------------------------
# Tracklets
# ----------------------------------------------------------------------------------------------
def frame_step(df):
    """Median frame step within tracks (1 for full-rate input, k for stride-k input)."""
    d = df.sort_values(["track_id", "frame"]).groupby("track_id")["frame"].diff().dropna()
    d = d[d > 0]
    return int(max(1, round(float(d.median())))) if len(d) else 1


def split_into_tracklets(df, split_gap=6, scale_by_step=True):
    """Adds cx, cy and 'tracklet_id' = '<track_id>_<k>' (new k after a frame gap > split_gap)."""
    df = df.copy()
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["cy"] = (df["y1"] + df["y2"]) / 2
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    gap = split_gap * (frame_step(df) if scale_by_step else 1)
    fd = df.groupby("track_id")["frame"].diff()
    new_seg = (fd > gap).fillna(False).astype(int)
    sub = new_seg.groupby(df["track_id"]).cumsum()
    df["tracklet_id"] = df["track_id"].astype(str) + "_" + sub.astype(str)
    return df


def _slope(f, v):
    """Least-squares slope of v against f (per frame). 0 if <2 distinct frames."""
    if len(f) < 2:
        return 0.0
    f = f - f.mean()
    den = float((f * f).sum())
    return float((f * (v - v.mean())).sum() / den) if den > 0 else 0.0


def summarize_tracklets(df_split, fps, frame_w=None, frame_h=None, vel_source="fd",
                        fd_window_s=0.25, fd_window_dets=5):
    """Per-tracklet endpoint statistics used by the pair features. Returns a DataFrame indexed
    0..N-1, sorted by first frame."""
    use_kalman = vel_source == "kalman"
    if use_kalman and not {"velocity_x", "velocity_y"} <= set(df_split.columns):
        raise ValueError("vel_source='kalman' needs velocity_x/velocity_y columns")
    step = frame_step(df_split)
    if fd_window_s is not None:
        k = max(1, int(round(fd_window_s * fps / step)))    # detections spanned by the window
    else:
        k = max(1, int(fd_window_dets))
    rows = []
    d = df_split.sort_values(["tracklet_id", "frame"])
    cols = ["frame", "cx", "cy", "x1", "y1", "x2", "y2"] + (["velocity_x", "velocity_y"] if use_kalman else [])
    arrs = {c: d[c].to_numpy(dtype=float) for c in cols}
    tids = d["tracklet_id"].to_numpy()
    bases = d["track_id"].astype(str).to_numpy()
    bounds = np.flatnonzero(np.r_[True, tids[1:] != tids[:-1], True])
    W = float(frame_w) if frame_w else np.nan
    H = float(frame_h) if frame_h else np.nan
    for s, e in zip(bounds[:-1], bounds[1:]):
        f = arrs["frame"][s:e]; cx = arrs["cx"][s:e]; cy = arrs["cy"][s:e]
        w = arrs["x2"][s:e] - arrs["x1"][s:e]; h = arrs["y2"][s:e] - arrs["y1"][s:e]
        n = e - s
        r = {"tid": tids[s], "base": bases[s], "first": f[0], "last": f[-1], "n": n,
             "fx": cx[0], "fy": cy[0], "lx": cx[-1], "ly": cy[-1]}
        # stable boxes (mean of 5 detections), as in relinker_script.get_stable_box
        r["fw"], r["fh"] = w[:5].mean(), h[:5].mean()
        r["lw"], r["lh"] = w[-5:].mean(), h[-5:].mean()
        if use_kalman:
            vx = arrs["velocity_x"][s:e]; vy = arrs["velocity_y"][s:e]
            r["fvx"], r["fvy"], r["lvx"], r["lvy"] = vx[0], vy[0], vx[-1], vy[-1]
            r["fhead"] = math.atan2(vy[:5].mean(), vx[:5].mean())
            r["lhead"] = math.atan2(vy[-5:].mean(), vx[-5:].mean())
        else:
            m = min(n, k + 1)
            r["fvx"], r["fvy"] = _slope(f[:m], cx[:m]), _slope(f[:m], cy[:m])
            r["lvx"], r["lvy"] = _slope(f[-m:], cx[-m:]), _slope(f[-m:], cy[-m:])
            r["fhead"] = math.atan2(r["fvy"], r["fvx"])
            r["lhead"] = math.atan2(r["lvy"], r["lvx"])
        # legacy area growth: (area_last - area_first) / count over first/last 10 detections
        aL = (w[-10:] * h[-10:]); aF = (w[:10] * h[:10])
        r["lgrow"] = (aL[-1] - aL[0]) / len(aL) if len(aL) > 1 else 0.0
        r["fgrow"] = (aF[-1] - aF[0]) / len(aF) if len(aF) > 1 else 0.0
        # scale-free growth: d log(area) / d seconds over first/last 10 detections
        def lgs(ff, aa):
            if len(ff) < 2 or ff[-1] == ff[0] or aa[0] <= 0 or aa[-1] <= 0:
                return 0.0
            return (math.log(aa[-1]) - math.log(aa[0])) / ((ff[-1] - ff[0]) / fps)
        r["lgrow_n"] = lgs(f[-10:], aL); r["fgrow_n"] = lgs(f[:10], aF)
        # distance of the last / first box to the frame border (pixels)
        r["ledge"] = min(arrs["x1"][e - 1], arrs["y1"][e - 1], W - arrs["x2"][e - 1], H - arrs["y2"][e - 1])
        r["fedge"] = min(arrs["x1"][s], arrs["y1"][s], W - arrs["x2"][s], H - arrs["y2"][s])
        rows.append(r)
    T = pd.DataFrame(rows).sort_values(["first", "last"]).reset_index(drop=True)
    T.attrs["fps"] = fps
    T.attrs["fd_window_dets"] = k
    return T


# ----------------------------------------------------------------------------------------------
# Candidate pairs + features
# ----------------------------------------------------------------------------------------------
FEATURES = {
    "legacy9": ["traj_sqrt", "velocity_error", "speed", "time_diff", "pix_dist", "box_area_ratio",
                "area_growth_diff", "heading_diff", "aspect_ratio_diff"],
    "norm9": ["traj_sqrt_n", "vel_err_n", "speed_n", "dt_s", "dist_n", "box_area_ratio",
              "area_growth_diff_n", "heading_diff", "log_aspect_diff"],
}
FEATURES["norm+"] = FEATURES["norm9"] + ["speedA_n", "speedB_n", "log_durA", "log_durB", "edgeA_n", "edgeB_n"]
FEATURES["norm+ctx"] = FEATURES["norm+"] + ["rankA", "rankB"]


def _diag(w, h):
    return np.sqrt(np.asarray(w) ** 2 + np.asarray(h) ** 2)


def candidate_pairs(T, fps, gate):
    """Return index arrays (ia, ib) of candidate pairs A->B with B.first > A.last.
    gate: {'type': 'legacy', 'max_gap_frames': F, 'max_px': 300} or
          {'type': 'norm', 'max_gap_s': 10, 'a': 3, 'b': 3}  (dist <= (a + b*gap_s) box diagonals)
          {'type': 'none', 'max_gap_s': ...}  (time gate only)"""
    first = T["first"].to_numpy(); last = T["last"].to_numpy()
    order = np.argsort(first, kind="stable")
    fs = first[order]
    if gate["type"] == "legacy":
        maxf = gate["max_gap_frames"]
    else:
        maxf = gate["max_gap_s"] * fps
    IA, IB = [], []
    lx, ly = T["lx"].to_numpy(), T["ly"].to_numpy()
    fx, fy = T["fx"].to_numpy(), T["fy"].to_numpy()
    sA = _diag(T["lw"], T["lh"]); sB = _diag(T["fw"], T["fh"])
    for a in range(len(T)):
        lo = np.searchsorted(fs, last[a], side="right")
        hi = np.searchsorted(fs, last[a] + maxf, side="right")
        if hi <= lo:
            continue
        b = order[lo:hi]
        b = b[b != a]
        dt = first[b] - last[a]
        dist = np.hypot(fx[b] - lx[a], fy[b] - ly[a])
        if gate["type"] == "legacy":
            keep = (dt <= maxf) & (dist < gate.get("max_px", 300))
        elif gate["type"] == "norm":
            s = (sA[a] + sB[b]) / 2
            keep = dist / np.maximum(s, 1e-6) <= gate["a"] + gate["b"] * dt / fps
        else:
            keep = np.ones(len(b), bool)
        IA.append(np.full(keep.sum(), a)); IB.append(b[keep])
    if not IA:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(IA).astype(int), np.concatenate(IB).astype(int)


def _wrap(d):
    d = np.abs(d)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def pair_features(T, ia, ib, fps):
    """All features (legacy + normalised) for pairs ia -> ib. Returns a DataFrame."""
    A = T.iloc[ia].reset_index(drop=True); B = T.iloc[ib].reset_index(drop=True)
    out = pd.DataFrame({"id1": A["tid"].values, "id2": B["tid"].values,
                        "base1": A["base"].values, "base2": B["base"].values})
    out["same_id"] = (out["base1"] == out["base2"]).astype(int)
    dt = (B["first"] - A["last"]).to_numpy(float)
    dxy = np.hypot(B["fx"] - A["lx"], B["fy"] - A["ly"]).to_numpy()
    vAx, vAy = A["lvx"].to_numpy(float), A["lvy"].to_numpy(float)
    vBx, vBy = B["fvx"].to_numpy(float), B["fvy"].to_numpy(float)
    terr = np.hypot(A["lx"] + dt * vAx - B["fx"], A["ly"] + dt * vAy - B["fy"]).to_numpy()
    verr = np.hypot(vAx - vBx, vAy - vBy)
    aA = (A["lw"] * A["lh"]).to_numpy(); aB = (B["fw"] * B["fh"]).to_numpy()
    mx = np.maximum(aA, aB)
    with np.errstate(divide="ignore", invalid="ignore"):
        bar = np.where(mx > 0, np.minimum(aA, aB) / np.where(mx > 0, mx, 1), 0.0)
        aspA = np.where(A["lh"] > 0, A["lw"] / A["lh"].where(A["lh"] > 0, 1), 0.0)
        aspB = np.where(B["fh"] > 0, B["fw"] / B["fh"].where(B["fh"] > 0, 1), 0.0)
    head = _wrap(A["lhead"].to_numpy() - B["fhead"].to_numpy())
    # ---- legacy (pixels / frames), identical to relinker_script.py when vel_source='kalman'
    out["time_diff"] = dt
    out["pix_dist"] = dxy
    out["traj_sqrt"] = terr / np.sqrt(dt)
    out["velocity_error"] = verr
    out["speed"] = dxy / dt
    out["box_area_ratio"] = bar
    out["aspect_ratio_diff"] = np.abs(aspA - aspB)
    out["heading_diff"] = head
    out["area_growth_diff"] = np.abs(A["lgrow"].to_numpy() - B["fgrow"].to_numpy())
    # ---- normalised (seconds / box diagonals)
    s = np.maximum((_diag(A["lw"], A["lh"]) + _diag(B["fw"], B["fh"])) / 2, 1e-6)
    dts = dt / fps
    out["dt_s"] = dts
    out["dist_n"] = dxy / s
    out["speed_n"] = out["dist_n"] / dts
    out["traj_sqrt_n"] = (terr / s) / np.sqrt(dts)
    out["vel_err_n"] = verr * fps / s
    with np.errstate(divide="ignore", invalid="ignore"):
        la = np.log(np.clip(aspA, 1e-3, None)) - np.log(np.clip(aspB, 1e-3, None))
    out["log_aspect_diff"] = np.abs(la)
    out["area_growth_diff_n"] = np.abs(A["lgrow_n"].to_numpy() - B["fgrow_n"].to_numpy())
    out["speedA_n"] = np.hypot(vAx, vAy) * fps / s
    out["speedB_n"] = np.hypot(vBx, vBy) * fps / s
    out["log_durA"] = np.log1p((A["last"] - A["first"]).to_numpy() / fps)
    out["log_durB"] = np.log1p((B["last"] - B["first"]).to_numpy() / fps)
    ea = A["ledge"].to_numpy(float) / _diag(A["lw"], A["lh"]).clip(1e-6)
    eb = B["fedge"].to_numpy(float) / _diag(B["fw"], B["fh"]).clip(1e-6)
    out["edgeA_n"] = np.nan_to_num(np.clip(ea, -1, 5), nan=5.0)
    out["edgeB_n"] = np.nan_to_num(np.clip(eb, -1, 5), nan=5.0)
    # context: rank of this pair's normalised distance among A's successors / B's predecessors
    out["rankA"] = out.groupby("id1")["dist_n"].rank(method="min") - 1
    out["rankB"] = out.groupby("id2")["dist_n"].rank(method="min") - 1
    out["first2"] = B["first"].values
    out["last1"] = A["last"].values
    return out


def build_pairs(df, fps, frame_w=None, frame_h=None, cfg=None):
    """df (canonical columns) -> (df_split, tracklet table T, pairs DataFrame with features)."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    dsplit = split_into_tracklets(df, cfg["split_gap"])
    T = summarize_tracklets(dsplit, fps, frame_w, frame_h, cfg["vel_source"], cfg["fd_window_s"],
                            cfg["fd_window_dets"])
    ia, ib = candidate_pairs(T, fps, cfg["gate"])
    P = pair_features(T, ia, ib, fps)
    # 'next same-id tracklet' flag: B is the tracklet that directly follows A within the same id
    nxt = {}
    for base, g in T.groupby("base"):
        tl = g.sort_values("first")["tid"].tolist()
        for x, y in zip(tl[:-1], tl[1:]):
            nxt[x] = y
    P["next_same_id"] = [int(nxt.get(a) == b) for a, b in zip(P["id1"], P["id2"])]
    return dsplit, T, P


# ----------------------------------------------------------------------------------------------
# Chain assembly
# ----------------------------------------------------------------------------------------------
def assemble_greedy(pairs, p, threshold):
    """Best-first one-to-one links (relinker_script.py @ 11e6a17 build_relinked_chains)."""
    order = np.argsort(-np.asarray(p), kind="stable")
    used_s, used_t, links = set(), set(), []
    for i in order:
        if p[i] < threshold:
            break
        a, b = pairs["id1"].iat[i], pairs["id2"].iat[i]
        if a not in used_s and b not in used_t:
            links.append((a, b, float(p[i]))); used_s.add(a); used_t.add(b)
    return links


def assemble_hungarian(pairs, p, threshold):
    """Global one-to-one assignment maximising sum log(p/threshold) over accepted links
    (equivalently minimising sum -log p with an entry/exit cost -log threshold)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    p = np.asarray(p, float)
    keep = p > threshold
    if not keep.any():
        return []
    sub = pairs.loc[keep, ["id1", "id2"]].reset_index(drop=True)
    w = np.log(p[keep] / threshold)
    srcs = {t: i for i, t in enumerate(pd.unique(sub["id1"]))}
    tgts = {t: i for i, t in enumerate(pd.unique(sub["id2"]))}
    si = sub["id1"].map(srcs).to_numpy(); ti = sub["id2"].map(tgts).to_numpy()
    ns, nt = len(srcs), len(tgts)
    # split into connected components of the bipartite graph to keep matrices small
    g = coo_matrix((np.ones(len(si)), (si, ns + ti)), shape=(ns + nt, ns + nt))
    _, lab = connected_components(g, directed=False)
    inv_s = np.array(list(srcs.keys()), dtype=object); inv_t = np.array(list(tgts.keys()), dtype=object)
    links = []
    comp_of_edge = lab[si]
    for c in np.unique(comp_of_edge):
        e = np.flatnonzero(comp_of_edge == c)
        us, ut = np.unique(si[e]), np.unique(ti[e])
        rs = {v: k for k, v in enumerate(us)}; rt = {v: k for k, v in enumerate(ut)}
        M = np.zeros((len(us), len(ut)))
        for k in e:
            M[rs[si[k]], rt[ti[k]]] = max(M[rs[si[k]], rt[ti[k]]], w[k])
        r, cidx = linear_sum_assignment(-M)
        for a, b in zip(r, cidx):
            if M[a, b] > 0:
                links.append((inv_s[us[a]], inv_t[ut[b]], float(threshold * math.exp(M[a, b]))))
    return links


def assemble_mincostflow(pairs, p, threshold, scale=100000):
    """Min-cost-flow formulation (Zhang et al. 2008 style): every tracklet is covered by exactly
    one path; a path pays an entry + exit cost (together -log threshold) and each link costs -log p.
    Lower-bound-1 'observation' edges are encoded as node demands (in-node absorbs 1 unit,
    out-node supplies 1 unit); S->T bypass keeps the flow balanced. Solved with networkx network
    simplex on integer costs. With a fixed per-tracklet cost this is mathematically the same
    optimisation as assemble_hungarian (kept to verify that equivalence)."""
    import networkx as nx
    p = np.asarray(p, float)
    keep = p > threshold
    sub = pairs.loc[keep, ["id1", "id2"]].reset_index(drop=True); pk = p[keep]
    nodes = pd.unique(pd.concat([sub["id1"], sub["id2"]]))
    n = len(nodes)
    if n == 0:
        return []
    G = nx.DiGraph()
    half = int(round(scale * -math.log(threshold) / 2))
    G.add_node("S", demand=-n); G.add_node("T", demand=n)
    G.add_edge("S", "T", weight=0, capacity=n)
    for t in nodes:
        G.add_node(("in", t), demand=1); G.add_node(("out", t), demand=-1)
        G.add_edge("S", ("in", t), weight=half, capacity=1)
        G.add_edge(("out", t), "T", weight=half, capacity=1)
    for (a, b), pp in zip(sub.itertuples(index=False, name=None), pk):
        G.add_edge(("out", a), ("in", b), weight=int(round(-scale * math.log(pp))), capacity=1)
    flow = nx.network_simplex(G)[1]
    links = []
    for (a, b), pp in zip(sub.itertuples(index=False, name=None), pk):
        if flow[("out", a)].get(("in", b), 0) > 0:
            links.append((a, b, float(pp)))
    return links


def chains_from_links(links, all_tids, first_frame, last_frame, base_of, merge="none"):
    """Follow links into ordered chains (every tracklet appears in exactly one chain, singletons
    included). merge: 'none' | 'flicker2' (11e6a17 rule: merge chains sharing >= 2 track ids when
    no temporal overlap between different ids) | 'sameid' (merge chains sharing >= 1 track id when
    no two tracklets of the merged chains overlap in time)."""
    succ = {a: b for a, b, _ in links}
    has_pred = {b for _, b, _ in links}
    chains = []
    for t in sorted(all_tids, key=lambda x: (first_frame[x], last_frame[x], x)):
        if t in has_pred:
            continue
        c = [t]
        while c[-1] in succ:
            c.append(succ[c[-1]])
        chains.append(c)
    if merge == "none":
        return chains

    def overlap(c1, c2, ignore_same_base):
        for a in c1:
            for b in c2:
                if ignore_same_base and base_of[a] == base_of[b]:
                    continue
                if first_frame[a] <= last_frame[b] and first_frame[b] <= last_frame[a]:
                    return True
        return False

    need = 2 if merge == "flicker2" else 1
    changed = True
    while changed:
        changed = False
        for i in range(len(chains)):
            if chains[i] is None:
                continue
            bi = {base_of[t] for t in chains[i]}
            for j in range(i + 1, len(chains)):
                if chains[j] is None:
                    continue
                bj = {base_of[t] for t in chains[j]}
                if len(bi & bj) < need:
                    continue
                if overlap(chains[i], chains[j], ignore_same_base=(merge == "flicker2")):
                    continue
                chains[i] = sorted(set(chains[i]) | set(chains[j]), key=lambda x: (first_frame[x], x))
                bi = {base_of[t] for t in chains[i]}
                chains[j] = None
                changed = True
        chains = [c for c in chains if c is not None]
    return chains


def assemble(pairs, p, threshold, T, method="hungarian", merge="none"):
    if method == "greedy":
        links = assemble_greedy(pairs, p, threshold)
    elif method == "hungarian":
        links = assemble_hungarian(pairs, p, threshold)
    elif method == "mincostflow":
        links = assemble_mincostflow(pairs, p, threshold)
    else:
        raise ValueError(method)
    ff = dict(zip(T["tid"], T["first"])); lf = dict(zip(T["tid"], T["last"])); bo = dict(zip(T["tid"], T["base"]))
    chains = chains_from_links(links, list(T["tid"]), ff, lf, bo, merge=merge)
    return links, chains


# ----------------------------------------------------------------------------------------------
# Scoring + public API
# ----------------------------------------------------------------------------------------------
def score_pairs(P, bundle):
    """Probability that A->B is the same vehicle. Same-id continuations use the rule prior when
    cfg['same_id_rule'] is on; everything else goes through the model."""
    cfg = {**DEFAULT_CONFIG, **bundle.get("config", {})}
    feats = bundle["feature_cols"]
    p = np.zeros(len(P))
    if len(P) == 0:
        return p
    if cfg["same_id_rule"]:
        rule = P["next_same_id"].to_numpy() == 1
        other_same = (P["same_id"].to_numpy() == 1) & ~rule
        model_rows = ~(rule | other_same)
        p[rule] = cfg["same_id_p"]
        p[other_same] = 0.0      # skip links within one id: left to the chain / merge step
    else:
        model_rows = np.ones(len(P), bool)
    if model_rows.any():
        X = P.loc[model_rows, feats].to_numpy(float)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        p[model_rows] = bundle["model"].predict_proba(X)[:, 1]
    return p


def load_model(model_path=None):
    import joblib
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "relinker_v2_both.joblib")
    return joblib.load(model_path)


def relink(df, fps, frame_w, frame_h, model_path=None, threshold=None, assembly=None, merge=None,
           bundle=None, return_all=False):
    """Relink fragmented tracks.

    df: any DataFrame with frame, track id and a box (column variants auto-detected; see
        normalize_tracks). fps: frames per second of the ORIGINAL frame numbering (for stride-k
        output with original frame numbers pass the source fps). frame_w/frame_h: pixels.
    Returns (links_df, tracklet_to_vehicle, chains):
      links_df: every candidate pair id1 -> id2 with p (probability) and accepted (bool) + features;
      tracklet_to_vehicle: {tracklet_id: vehicle_id (int, 0.. ordered by first appearance)};
      chains: list of lists of tracklet ids in time order (one list per vehicle, singletons included).
    """
    if bundle is None:
        bundle = load_model(model_path)
    cfg = {**DEFAULT_CONFIG, **bundle.get("config", {})}
    if threshold is not None:
        cfg["threshold"] = threshold
    if assembly is not None:
        cfg["assembly"] = assembly
    if merge is not None:
        cfg["merge"] = merge
    if not {"frame", "track_id", "x1", "y1", "x2", "y2"} <= set(df.columns) or df.attrs.get("box_format") is None:
        df = normalize_tracks(df, frame_w=frame_w, frame_h=frame_h)
    if len(df) == 0:                     # nothing to relink (e.g. a clip with no detections)
        P = pd.DataFrame(columns=["id1", "id2", "base1", "base2", "same_id", "next_same_id", "p", "accepted"]
                         + list(bundle["feature_cols"]))
        if return_all:
            return P, {}, [], df.assign(cx=[], cy=[], tracklet_id=[]), pd.DataFrame()
        return P, {}, []
    dsplit, T, P = build_pairs(df, fps, frame_w, frame_h, cfg)
    p = score_pairs(P, bundle)
    links, chains = assemble(P, p, cfg["threshold"], T, cfg["assembly"], cfg["merge"])
    acc = {(a, b) for a, b, _ in links}
    P = P.copy()
    P["p"] = p
    P["accepted"] = [(a, b) in acc for a, b in zip(P["id1"], P["id2"])]
    ff = dict(zip(T["tid"], T["first"]))
    chains = sorted(chains, key=lambda c: min(ff[t] for t in c))
    t2v = {t: v for v, c in enumerate(chains) for t in c}
    if return_all:
        return P, t2v, chains, dsplit, T
    return P, t2v, chains


def apply_relink(df, tracklet_to_vehicle, split_gap=None):
    """Return df with tracklet_id and vehicle_id columns (rows of unknown tracklets keep -1)."""
    split_gap = DEFAULT_CONFIG["split_gap"] if split_gap is None else split_gap
    if df.attrs.get("box_format") is None:
        df = normalize_tracks(df)
    d = split_into_tracklets(df, split_gap)
    d["vehicle_id"] = d["tracklet_id"].map(tracklet_to_vehicle).fillna(-1).astype(int)
    return d


def _video_meta(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tracker-agnostic tracklet relinker (relinker_v2)")
    ap.add_argument("tracks", help="track CSV / MOTChallenge txt")
    ap.add_argument("--fps", type=float); ap.add_argument("--width", type=int); ap.add_argument("--height", type=int)
    ap.add_argument("--video", help="read fps/width/height from this video")
    ap.add_argument("--model", default=None, help="joblib bundle (default: models/relinker_v2_both.joblib next to this script)")
    ap.add_argument("--box-format", default="auto", choices=["auto", "xyxy", "tlwh", "cxcywh"])
    ap.add_argument("--threshold", type=float); ap.add_argument("--assembly", choices=["hungarian", "greedy", "mincostflow"])
    ap.add_argument("--merge", choices=["none", "flicker2", "sameid"],
                    help="chain merge step (default from the bundle: sameid). For DeepSORT output consider "
                         "'none': on 8 held-out UA-DETRAC sequences 'sameid' lowered DeepSORT IDF1 by 0.024 "
                         "(6/8 hurt) because DeepSORT sometimes moves one id to another car; 'none' gave "
                         "+0.024 in a post-hoc check. Neutral on ByteTrack/BoT-SORT. A checked same-id join "
                         "is planned for v2.1")
    ap.add_argument("--out-prefix", default=None, help="writes <prefix>_links.csv, _relinked.csv, _chains.json")
    a = ap.parse_args(argv)
    fps, W, H = a.fps, a.width, a.height
    if a.video:
        vf, vw, vh = _video_meta(a.video)
        fps, W, H = fps or vf, W or vw, H or vh
    if not fps:
        ap.error("--fps (or --video) is required")
    df = read_tracks(a.tracks, box_format=a.box_format, frame_w=W, frame_h=H, verbose=True)
    links, t2v, chains = relink(df, fps, W, H, model_path=a.model, threshold=a.threshold,
                                assembly=a.assembly, merge=a.merge)
    n_ids = df["track_id"].nunique()
    print(f"track ids: {n_ids}  tracklets: {len(t2v)}  vehicles: {len(chains)}  "
          f"candidate pairs: {len(links)}  accepted links: {int(links['accepted'].sum())}")
    prefix = a.out_prefix or os.path.splitext(a.tracks)[0] + "_relinked"
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    links.to_csv(prefix + "_links.csv", index=False)
    apply_relink(df, t2v).to_csv(prefix + "_relinked.csv", index=False)
    with open(prefix + "_chains.json", "w") as fh:
        json.dump({"chains": chains, "tracklet_to_vehicle": t2v}, fh, indent=1)
    print(f"wrote {prefix}_links.csv, {prefix}_relinked.csv, {prefix}_chains.json")


if __name__ == "__main__":
    main()
