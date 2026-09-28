"""zones.py - tracking-free zone occupancy (no track IDs at all).

Idea: a parked / curb-stopped vehicle produces nearly the same box frame after frame.  So
  1. sample frames at ``sample_fps`` (1 fps is plenty),
  2. flag a detection "stationary" when the sampled frame ``stat_dt_s`` before OR after it contains a
     box with IoU >= ``stat_iou`` (``stat_require_both=True``: before AND after),
  3. cluster stationary boxes into zones (leader clustering on IoU, then 2 refinement passes with the
     per-zone median box, then merging of zones whose median boxes overlap >= ``zone_iou``),
  4. zone occupied at sample t  <=>  a stationary box of that frame overlaps the zone box >= ``zone_iou``;
     fill gaps <= ``gap_fill_s``, drop runs < ``min_interval_s``  -> occupancy intervals = dwell per zone.

Input: any per-frame detections ``frame, x1, y1, x2, y2`` (track ids, if present, are ignored).
Everything is in seconds and IoU, so it is independent of resolution and frame rate.

What it cannot do: identify WHICH vehicle was there (two cars using a stall back-to-back within
``gap_fill_s`` merge into one interval), through-traffic counts, pull-in / back-out maneuver timing,
or zones that are occluded most of the time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:                                   # imported as tools.zones (from the repo root)
    from .mot_eval import iou_matrix
except ImportError:                    # run / imported from inside tools/
    from mot_eval import iou_matrix

ZONE_DEFAULTS = dict(sample_fps=1.0, stat_iou=0.7, stat_dt_s=1.0, zone_iou=0.5, min_zone_s=2.0,
                     gap_fill_s=3.0, min_interval_s=2.0, offset_s=0.0, stat_require_both=False)


def _sample_frames(clip, fps, sample_fps, offset_s=0.0):
    f0, f1 = clip
    if sample_fps is None or sample_fps >= fps:
        return np.arange(f0, f1 + 1)
    t = np.arange(offset_s + (f0 - 1) / fps, (f1 - 1) / fps + 1e-9, 1.0 / sample_fps)
    fr = np.unique(np.round(t * fps).astype(int) + 1)
    return fr[(fr >= f0) & (fr <= f1)]


def zone_occupancy(det_df, fps, clip=None, **params):
    """Returns (zones, intervals, info).
    zones:     DataFrame[zone, x1, y1, x2, y2, support_s]
    intervals: DataFrame[zone, t_start, t_end, f_start, f_end, dwell_s, left_censored, right_censored]
    info:      dict (frames analysed, stationary boxes, parameters)."""
    P = dict(ZONE_DEFAULTS)
    P.update(params)
    if clip is None:
        clip = (int(det_df.frame.min()), int(det_df.frame.max()))
    frames = _sample_frames(clip, fps, P["sample_fps"], P["offset_s"])
    dt = (np.median(np.diff(frames)) / fps) if len(frames) > 1 else 1.0 / fps
    D = det_df[det_df.frame.isin(frames)].sort_values("frame")
    fr = D.frame.to_numpy()
    B = D[["x1", "y1", "x2", "y2"]].to_numpy(float)
    idx = {f: (np.searchsorted(fr, f, "left"), np.searchsorted(fr, f, "right")) for f in frames}
    k = max(1, int(round(P["stat_dt_s"] / dt)))       # neighbour offset in samples
    stat = np.zeros(len(D), bool)
    for i, f in enumerate(frames):
        a, b = idx[f]
        if a == b:
            continue
        ok = []
        for j in (i - k, i + k):
            if 0 <= j < len(frames):
                c, d = idx[frames[j]]
                if c == d:
                    ok.append(np.zeros(b - a, bool))
                else:
                    ok.append(iou_matrix(B[a:b], B[c:d]).max(1) >= P["stat_iou"])
        if not ok:
            continue
        # "any": one matching neighbour suffices (a single missed detection must not break a stop;
        # a moving vehicle matches neither side).  stat_require_both=True gives the strict variant.
        stat[a:b] = (np.logical_and.reduce(ok) if P["stat_require_both"] else np.logical_or.reduce(ok))
    SB, SF = B[stat], fr[stat]
    # --- leader clustering + refinement
    reps = []
    lab = np.full(len(SB), -1)
    for i, b in enumerate(SB):
        if reps:
            m = iou_matrix(b[None], np.array(reps))[0]
            j = int(np.argmax(m))
            if m[j] >= P["zone_iou"]:
                lab[i] = j
                continue
        reps.append(b.copy()); lab[i] = len(reps) - 1
    for _ in range(2):
        if not len(reps):
            break
        reps = np.array([np.median(SB[lab == z], axis=0) for z in range(len(reps)) if np.any(lab == z)])
        M = iou_matrix(SB, reps)
        lab = np.where(M.max(1) >= P["zone_iou"], M.argmax(1), -1)
    # merge overlapping zones
    if len(reps):
        changed = True
        reps = [r for r in reps]
        while changed and len(reps) > 1:
            changed = False
            M = iou_matrix(np.array(reps), np.array(reps))
            np.fill_diagonal(M, 0)
            i, j = np.unravel_index(np.argmax(M), M.shape)
            if M[i, j] >= P["zone_iou"]:
                lab = np.where(lab == j, i, lab)
                lab = np.where(lab > j, lab - 1, lab)
                reps.pop(j)
                reps[i] = np.median(SB[lab == i], axis=0)
                changed = True
        reps = np.array(reps)
        M = iou_matrix(SB, reps)
        lab = np.where(M.max(1) >= P["zone_iou"], M.argmax(1), -1)
    zones, intervals = [], []
    zid = 0
    t_of = (frames - 1) / fps
    fpos = {f: i for i, f in enumerate(frames)}
    for z in range(len(reps)):
        mz = lab == z
        occ = np.zeros(len(frames), bool)
        occ[[fpos[f] for f in np.unique(SF[mz])]] = True
        support = occ.sum() * dt
        if support < P["min_zone_s"]:
            continue
        # gap filling
        on = np.where(occ)[0]
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a and (t_of[b] - t_of[a]) <= P["gap_fill_s"] + 1e-9:
                occ[a:b] = True
        d = np.diff(np.concatenate([[0], occ.astype(int), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
        n_int = 0
        for s, e in zip(starts, ends):
            dur = t_of[e] - t_of[s] + dt
            if dur < P["min_interval_s"]:
                continue
            intervals.append(dict(zone=zid, t_start=t_of[s], t_end=t_of[e], f_start=int(frames[s]),
                                  f_end=int(frames[e]), dwell_s=dur, left_censored=bool(s == 0),
                                  right_censored=bool(e == len(frames) - 1)))
            n_int += 1
        if n_int:
            r = np.median(SB[mz], axis=0)
            zones.append(dict(zone=zid, x1=r[0], y1=r[1], x2=r[2], y2=r[3], support_s=support))
            zid += 1
    info = dict(n_frames=int(len(frames)), sample_dt_s=float(dt), n_boxes=int(len(D)),
                n_stationary_boxes=int(stat.sum()), params=P)
    iv_cols = ["zone", "t_start", "t_end", "f_start", "f_end", "dwell_s", "left_censored", "right_censored"]
    return (pd.DataFrame(zones, columns=["zone", "x1", "y1", "x2", "y2", "support_s"]),
            pd.DataFrame(intervals, columns=iv_cols), info)


def evaluate_zones(zones, intervals, ref_stops, gt_df, fps, tol_s=2.0, match_iou=0.5):
    """Compare zone occupancy with reference (oracle) stops.

    ref_stops: behavior.oracle_result(...).stops (needs gt_id, t_start, t_end, dwell_s, x1..y2 and
    censoring flags).  A GT stop is FOUND if some zone box overlaps its median box with IoU >=
    ``match_iou`` and one of that zone's intervals overlaps the stop in time.  Dwell estimate = the
    longest such interval.  An interval is attributed to a GT vehicle if that vehicle's GT box at the
    interval midpoint overlaps the zone (IoU >= match_iou); attributed intervals overlapping none of
    that vehicle's stops are counted as FALSE stops (e.g. queueing traffic).  Unattributed intervals are
    unlabeled vehicles and are ignored (partial GT).
    Start / end events (not censored) are matched to GT stop starts / ends within ``tol_s``."""
    per, errs, rels = [], [], []
    start_err, end_err = [], []
    Z = zones[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(zones) else np.zeros((0, 4))
    used = set()
    for s in ref_stops.itertuples():
        found, best, npieces, zmatch = False, 0.0, 0, None
        if len(Z):
            m = iou_matrix(np.array([[s.x1, s.y1, s.x2, s.y2]]), Z)[0]
            for z in np.argsort(-m):
                if m[z] < match_iou:
                    break
                zi = intervals[(intervals.zone == zones.zone.iloc[z]) & (intervals.t_end >= s.t_start) &
                               (intervals.t_start <= s.t_end)]
                if len(zi):
                    found, zmatch = True, int(zones.zone.iloc[z])
                    npieces = len(zi)
                    j = zi.dwell_s.idxmax()
                    best = float(zi.dwell_s.max())
                    used.update(zi.index.tolist())
                    iv = intervals.loc[j]
                    if not s.left_censored and not s.starts_at_track_start and not iv.left_censored:
                        start_err.append(abs(iv.t_start - s.t_start))
                    if not s.right_censored and not s.ends_at_track_end and not iv.right_censored:
                        end_err.append(abs(iv.t_end - s.t_end))
                    break
        errs.append(abs(best - s.dwell_s)); rels.append(abs(best - s.dwell_s) / s.dwell_s)
        per.append(dict(gt_id=s.gt_id, t_start=s.t_start, ref_dwell=s.dwell_s, zone=zmatch, zone_dwell=best,
                        n_pieces=npieces, found=found,
                        censored=bool(s.left_censored or s.right_censored)))
    # attribute every interval to a GT vehicle (if any) to count false stops
    false_stops, attributed = 0, 0
    for iv in intervals.itertuples():
        if iv.Index in used:
            attributed += 1
            continue
        zb = zones.loc[zones.zone == iv.zone, ["x1", "y1", "x2", "y2"]].to_numpy(float)
        mid = int(round((iv.f_start + iv.f_end) / 2))
        g = gt_df[(gt_df.frame >= mid - int(fps)) & (gt_df.frame <= mid + int(fps))]
        if len(g) == 0:
            continue
        m = iou_matrix(zb, g[["x1", "y1", "x2", "y2"]].to_numpy(float))[0]
        if m.max() >= match_iou:
            attributed += 1
            false_stops += 1
    n = len(per)
    ends_ok = [e for e in end_err if e <= tol_s]
    starts_ok = [e for e in start_err if e <= tol_s]
    return dict(n_ref_stops=n, stop_recall=float(np.mean([p["found"] for p in per])) if n else np.nan,
                dwell_mae_s=float(np.mean(errs)) if n else np.nan,
                dwell_rel_err_median=float(np.median(rels)) if n else np.nan,
                pieces_mean=float(np.mean([p["n_pieces"] for p in per])) if n else np.nan,
                n_intervals=int(len(intervals)), n_zones=int(len(zones)),
                n_intervals_attributed=int(attributed), false_stops=int(false_stops),
                stop_start_abs_err_s=start_err, stop_end_abs_err_s=end_err,
                stop_start_within_tol=f"{len(starts_ok)}/{len(start_err)}",
                stop_end_within_tol=f"{len(ends_ok)}/{len(end_err)}",
                per_stop=per)
