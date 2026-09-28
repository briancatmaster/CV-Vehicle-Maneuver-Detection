"""behavior.py - per-vehicle behavior from ANY trajectories (id -> boxes over time).

Input: a DataFrame with ``frame, <id_col>, x1, y1, x2, y2`` (1-based frame numbers; frames may be
subsampled, e.g. 1 fps) and the video's ``fps``.  Nothing here depends on DeepSORT or on a particular
detector: a relinked track, a ByteTrack output or pseudo-GT all work the same way.

All thresholds are expressed in seconds and in *box lengths* (L = rolling median of max(w, h) of the
vehicle's own box), so they do not depend on the frame rate, the resolution or the camera distance.

Per track the analysis produces
  * first / last seen (s) and whether they touch the clip boundaries (censoring),
  * stationary segments ("stops"): normalised speed < ``stop_speed`` L/s for >= ``min_stop_s``,
    short motion blips (< ``merge_gap_s``) inside a stop are merged, and two stops of the same track
    whose positions differ by < ``bridge_disp`` L are merged across holes / jitter up to ``max_bridge_s``
    (a parked car missed by the detector for a few seconds is still one stop),
  * dwell = stop duration (observed span + one sample period, which is unbiased for sampled data),
  * maneuvers around each stop:
      pull-in  (arrive): time from the last moment the vehicle was > ``maneuver_disp`` L away from the
               stop position until the stop starts (censored if the track starts closer than that),
      pull-out (depart): time from the stop end until the vehicle is first > ``maneuver_disp`` L away
               from the stop position (censored if the track ends first),
  * events: ``arrival`` (track starts after the clip start), ``departure`` (track ends before the clip
    end), ``stop_start`` and ``stop_end`` (only when observed inside the track - a track that begins
    already stopped has no stop_start, which is exactly how a fragment boundary shows up).

Main entry points
  analyze(df, fps, id_col='track_id', clip=(first_frame, last_frame), **params) -> BehaviorResult
      .tracks  DataFrame, one row per id (first/last seen, n_stops, total dwell, ...)
      .stops   DataFrame, one row per stop (id, t_start, t_end, dwell_s, cx, cy, L, pull-in/out, flags)
      .events  DataFrame, one row per event (id, type, t, frame, maneuver_s, censored)
  compare(ref, test, ref_map, test_map, tol_s=2.0) -> dict   (count / dwell / event metrics; used with
      the GT-vehicle attribution helpers below)
  attribute_to_gt(result, pred_df, scene, id_col) -> BehaviorResult with a gt_id on stops / events
  subsample(df, fps, target_fps) -> df restricted to frames nearest to k/target_fps (original numbers)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULTS = dict(
    smooth_s=0.5,        # rolling-median window for center and size (s)
    speed_window_s=1.0,  # displacement is measured over this window (s), centred
    stop_speed=0.10,     # L per second below which a vehicle counts as stationary
    min_stop_s=2.0,      # minimum dwell to report a stop (s)
    merge_gap_s=1.0,     # motion blips shorter than this between two stationary runs are merged (s)
    max_gap_s=3.0,       # a track with a hole longer than this is analysed as separate pieces (s)
    maneuver_disp=1.0,   # pull-in / pull-out ends when displacement from the stop position exceeds this (L)
    edge_tol_s=0.5,      # a track starting/ending within this of the clip start/end is "censored"
    bridge_disp=0.3,     # two stops of one track closer than this (L) are one stop, even across a
    max_bridge_s=5.0,    #   detection hole / jitter blip of up to max_bridge_s (s); longer holes are
                         #   ambiguous (White SUV backs out unseen during a 7.7 s hole, 105.7-113.4 s)
)


@dataclass
class BehaviorResult:
    tracks: pd.DataFrame
    stops: pd.DataFrame
    events: pd.DataFrame
    fps: float
    clip: tuple
    params: dict = field(default_factory=dict)


def subsample(df, fps, target_fps, offset_s=0.0):
    """Keep only rows whose frame is the frame nearest to ``offset_s + k / target_fps`` seconds.
    Frame numbers stay the original 1-based numbers.  Returns (df_sub, kept_frames)."""
    fmin, fmax = int(df.frame.min()), int(df.frame.max())
    if target_fps is None or target_fps >= fps:
        frames = np.arange(fmin, fmax + 1)
    else:
        t = np.arange(offset_s + (fmin - 1) / fps, (fmax - 1) / fps + 1e-9, 1.0 / target_fps)
        frames = np.unique(np.round(t * fps).astype(int) + 1)
        frames = frames[(frames >= fmin) & (frames <= fmax)]
    return df[df.frame.isin(frames)].copy(), frames


def _runs(mask):
    """Start/end indices (inclusive) of True runs."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0] - 1
    return list(zip(s, e))


def _rolling_median_time(t, x, win):
    """Centred rolling median over a time window (works for irregular sampling)."""
    if win <= 0 or len(x) < 3:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    lo = np.searchsorted(t, t - win / 2, side="left")
    hi = np.searchsorted(t, t + win / 2, side="right")
    return np.array([np.median(x[a:b]) for a, b in zip(lo, hi)])


def _analyze_piece(t, cx, cy, L, P):
    """Speed (L/s) for one gap-free piece."""
    W = P["speed_window_s"]
    n = len(t)
    if n == 1:
        return np.array([np.nan])
    ta = np.clip(t - W / 2, t[0], t[-1])
    tb = np.clip(t + W / 2, t[0], t[-1])
    # make sure the window spans at least one sample interval
    same = (tb - ta) <= 1e-9
    ta = np.where(same, np.maximum(t[0], t - W), ta)
    tb = np.where(same, np.minimum(t[-1], t + W), tb)
    xa, xb = np.interp(ta, t, cx), np.interp(tb, t, cx)
    ya, yb = np.interp(ta, t, cy), np.interp(tb, t, cy)
    dt = np.maximum(tb - ta, 1e-9)
    return np.hypot(xb - xa, yb - ya) / dt / np.maximum(L, 1.0)


def analyze(df, fps, id_col="track_id", clip=None, **params):
    """Per-track behavior.  ``clip`` = (first_frame, last_frame) of the analysed video segment; defaults
    to df's frame range.  Extra keyword arguments override ``DEFAULTS``."""
    P = dict(DEFAULTS)
    P.update(params)
    if clip is None:
        clip = (int(df.frame.min()), int(df.frame.max()))
    t_clip0, t_clip1 = (clip[0] - 1) / fps, (clip[1] - 1) / fps
    all_frames = np.unique(df.frame.to_numpy())
    dt_sample = float(np.median(np.diff(all_frames))) / fps if len(all_frames) > 1 else 1.0 / fps
    max_gap = max(P["max_gap_s"], 2.5 * dt_sample)
    tol = max(P["edge_tol_s"], 1.5 * dt_sample)
    track_rows, stop_rows, event_rows = [], [], []
    for tid, g in df.groupby(id_col, sort=False):
        g = g.sort_values("frame").drop_duplicates("frame")
        fr = g.frame.to_numpy()
        t = (fr - 1) / fps
        cx = ((g.x1 + g.x2) / 2).to_numpy(float)
        cy = ((g.y1 + g.y2) / 2).to_numpy(float)
        Lraw = np.maximum((g.x2 - g.x1).to_numpy(float), (g.y2 - g.y1).to_numpy(float))
        pieces = np.split(np.arange(len(t)), np.where(np.diff(t) > max_gap)[0] + 1)
        speed = np.full(len(t), np.nan)
        scx, scy, sL = cx.copy(), cy.copy(), Lraw.copy()
        # at low sampling rates widen the windows so they always span a few samples
        Pe = dict(P, smooth_s=max(P["smooth_s"], 2.5 * dt_sample),
                  speed_window_s=max(P["speed_window_s"], 2.0 * dt_sample))
        for idx in pieces:
            tt = t[idx]
            scx[idx] = _rolling_median_time(tt, cx[idx], Pe["smooth_s"])
            scy[idx] = _rolling_median_time(tt, cy[idx], Pe["smooth_s"])
            sL[idx] = _rolling_median_time(tt, Lraw[idx], max(Pe["smooth_s"], 2.0))
            speed[idx] = _analyze_piece(tt, scx[idx], scy[idx], sL[idx], Pe)
        # a single-sample piece has no speed -> treat as moving (unknown) so it cannot create a stop
        stat = np.where(np.isnan(speed), False, speed < P["stop_speed"])
        stops = []
        for idx in pieces:
            for a, b in _runs(stat[idx]):
                stops.append([idx[a], idx[b]])
        # merge stops separated by short motion blips (same piece only, i.e. no long hole between)
        merged = []
        for s in stops:
            if merged:
                a0, b0 = merged[-1]
                gap = t[s[0]] - t[b0]
                short_blip = gap <= P["merge_gap_s"] + dt_sample + 1e-9 and \
                    not np.any(np.diff(t[b0:s[0] + 1]) > max_gap)
                d = np.hypot(np.median(scx[a0:b0 + 1]) - np.median(scx[s[0]:s[1] + 1]),
                             np.median(scy[a0:b0 + 1]) - np.median(scy[s[0]:s[1] + 1]))
                same_place = gap <= P["max_bridge_s"] and \
                    d < P["bridge_disp"] * max(np.median(sL[a0:b0 + 1]), 1.0)
                if short_blip or same_place:
                    merged[-1][1] = s[1]
                    continue
            merged.append(list(s))
        t0, t1 = t[0], t[-1]
        left_c = t0 - t_clip0 <= tol
        right_c = t_clip1 - t1 <= tol
        n_stops, dwell_tot = 0, 0.0
        for a, b in merged:
            dur = t[b] - t[a] + dt_sample
            if dur < P["min_stop_s"]:
                continue
            n_stops += 1
            dwell_tot += dur
            px, py, pL = np.median(scx[a:b + 1]), np.median(scy[a:b + 1]), np.median(sL[a:b + 1])
            dist = np.hypot(scx - px, scy - py) / max(pL, 1.0)
            # pull-in
            far_before = np.where(dist[:a] > P["maneuver_disp"])[0]
            if len(far_before):
                t_far = t[far_before[-1]]
                pin, pin_c = t[a] - t_far, False
            else:
                pin, pin_c = t[a] - t0, True
            far_after = np.where(dist[b + 1:] > P["maneuver_disp"])[0]
            if len(far_after):
                pout, pout_c = t[b + 1 + far_after[0]] - t[b], False
            else:
                pout, pout_c = t1 - t[b], True
            starts_at_track_start = (t[a] - t0) <= tol
            ends_at_track_end = (t1 - t[b]) <= tol
            stop_rows.append(dict(id=tid, t_start=t[a], t_end=t[b], f_start=int(fr[a]), f_end=int(fr[b]),
                                  dwell_s=dur, cx=px, cy=py, L=pL,
                                  x1=float(np.median(g.x1.to_numpy()[a:b + 1])),
                                  y1=float(np.median(g.y1.to_numpy()[a:b + 1])),
                                  x2=float(np.median(g.x2.to_numpy()[a:b + 1])),
                                  y2=float(np.median(g.y2.to_numpy()[a:b + 1])),
                                  starts_at_track_start=bool(starts_at_track_start),
                                  ends_at_track_end=bool(ends_at_track_end),
                                  left_censored=bool(starts_at_track_start and left_c),
                                  right_censored=bool(ends_at_track_end and right_c),
                                  pull_in_s=pin, pull_in_censored=bool(pin_c),
                                  pull_out_s=pout, pull_out_censored=bool(pout_c)))
            if not starts_at_track_start:
                event_rows.append(dict(id=tid, type="stop_start", t=t[a], frame=int(fr[a]),
                                       maneuver_s=pin, censored=bool(pin_c)))
            if not ends_at_track_end:
                event_rows.append(dict(id=tid, type="stop_end", t=t[b], frame=int(fr[b]),
                                       maneuver_s=pout, censored=bool(pout_c)))
        if not left_c:
            event_rows.append(dict(id=tid, type="arrival", t=t0, frame=int(fr[0]), maneuver_s=np.nan,
                                   censored=False))
        if not right_c:
            event_rows.append(dict(id=tid, type="departure", t=t1, frame=int(fr[-1]), maneuver_s=np.nan,
                                   censored=False))
        track_rows.append(dict(id=tid, first_s=t0, last_s=t1, first_frame=int(fr[0]), last_frame=int(fr[-1]),
                               duration_s=t1 - t0 + dt_sample, n_samples=len(t), present_at_start=bool(left_c),
                               present_at_end=bool(right_c), n_stops=n_stops, dwell_total_s=dwell_tot,
                               median_speed_Lps=float(np.nanmedian(speed)) if np.isfinite(speed).any() else np.nan))
    ev_cols = ["id", "type", "t", "frame", "maneuver_s", "censored"]
    return BehaviorResult(tracks=pd.DataFrame(track_rows), stops=pd.DataFrame(stop_rows),
                          events=pd.DataFrame(event_rows, columns=ev_cols), fps=fps, clip=tuple(clip), params=P)


# ----------------------------------------------------------------------------- GT attribution + comparison
def attribute_to_gt(res, pred_df, scene, id_col="track_id", iou=0.5, max_frame_dist_s=1.0):
    """Attach a ``gt_id`` to every stop / event / track of ``res`` using per-frame IoU matching to the
    scene's pseudo GT (mot_eval.match_frames).  A stop gets the GT vehicle matched most often inside it;
    an event gets the GT vehicle matched at the nearest matched frame of that id within
    ``max_frame_dist_s``; a track gets its majority GT vehicle.  Unmatched -> NaN (unlabeled vehicle)."""
    try:                               # imported as tools.behavior (from the repo root)
        from . import mot_eval as me
    except ImportError:                # run / imported from inside tools/
        import mot_eval as me
    gt = me.load_gt(scene)
    m = me.match_frames(pred_df, gt, iou=iou, id_col=id_col, frames=np.unique(pred_df.frame))
    by_id = {k: g.sort_values("frame") for k, g in m.groupby("pred_id")}
    fps = res.fps

    def at(pid, frame):
        g = by_id.get(pid)
        if g is None:
            return np.nan
        fr = g.frame.to_numpy()
        i = np.searchsorted(fr, frame)
        cand = [j for j in (i - 1, i) if 0 <= j < len(fr)]
        j = min(cand, key=lambda j: abs(fr[j] - frame))
        if abs(fr[j] - frame) / fps > max_frame_dist_s:
            return np.nan
        return g.gt_id.iloc[j]

    def within(pid, f0, f1):
        g = by_id.get(pid)
        if g is None:
            return np.nan
        s = g[(g.frame >= f0) & (g.frame <= f1)]
        return s.gt_id.mode().iloc[0] if len(s) else np.nan

    stops = res.stops.copy()
    if len(stops):
        stops["gt_id"] = [within(r.id, r.f_start, r.f_end) for r in stops.itertuples()]
    else:
        stops["gt_id"] = []
    ev = res.events.copy()
    ev["gt_id"] = [at(r.id, r.frame) for r in ev.itertuples()] if len(ev) else []
    tr = res.tracks.copy()
    maj = {k: g.gt_id.mode().iloc[0] for k, g in by_id.items()}
    tr["gt_id"] = tr.id.map(maj)
    return BehaviorResult(tracks=tr, stops=stops, events=ev, fps=res.fps, clip=res.clip, params=res.params)


def _match_events(ref_t, test_t, tol):
    """Greedy one-to-one matching of two sorted time lists within tol. Returns list of (i, j)."""
    pairs = sorted(((abs(a - b), i, j) for i, a in enumerate(ref_t) for j, b in enumerate(test_t)
                    if abs(a - b) <= tol))
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j); out.append((i, j))
    return out


def compare(ref, test, tol_s=2.0, event_types=("arrival", "departure", "stop_start", "stop_end")):
    """Compare a test BehaviorResult with a reference one; both must carry ``gt_id`` on stops / events
    (use ``attribute_to_gt``; for the oracle, gt_id == id).  Only items with a gt_id are compared.

    Returns dict with
      n_ref_vehicles, n_test_ids, count_inflation   (distinct test ids attributed to GT / GT vehicles)
      dwell: per reference stop, the test stops of the same GT vehicle overlapping it in time:
        n_ref_stops, stop_recall (>=1 overlapping test stop), pieces_mean / max (test stops per ref stop),
        dwell_mae_s, dwell_rel_err_median / mean (|largest overlapping test stop - ref| / ref; a missed
        stop counts as error = ref dwell), dwell_ratio_median (largest piece / ref),
        n_test_stops_unmatched (test stops overlapping no ref stop of their vehicle)
      events: per type and overall precision / recall / F1 at tolerance ``tol_s``,
        plus maneuver MAE for matched stop_start (pull-in) / stop_end (pull-out) pairs uncensored on both sides
    """
    out = {}
    rs = ref.stops[ref.stops.gt_id.notna()] if len(ref.stops) else ref.stops
    ts = test.stops[test.stops.gt_id.notna()] if len(test.stops) else test.stops
    out["n_ref_vehicles"] = int(ref.tracks.gt_id.nunique())
    out["n_test_ids"] = int(test.tracks.gt_id.notna().sum())
    out["count_inflation"] = out["n_test_ids"] / max(out["n_ref_vehicles"], 1)
    errs, rels, ratios, pieces = [], [], [], []
    used_test = set()
    per_stop = []
    for r in rs.itertuples():
        cand = ts[(ts.gt_id == r.gt_id) & (ts.t_end >= r.t_start) & (ts.t_start <= r.t_end)] if len(ts) else ts
        pieces.append(len(cand))
        used_test.update(cand.index.tolist())
        best = float(cand.dwell_s.max()) if len(cand) else 0.0
        errs.append(abs(best - r.dwell_s))
        rels.append(abs(best - r.dwell_s) / r.dwell_s)
        ratios.append(best / r.dwell_s)
        per_stop.append(dict(gt_id=r.gt_id, t_start=r.t_start, ref_dwell=r.dwell_s, test_best=best,
                             n_pieces=len(cand), censored=bool(r.left_censored or r.right_censored)))
    n = len(rs)
    out["dwell"] = dict(
        n_ref_stops=n,
        stop_recall=float(np.mean(np.array(pieces) > 0)) if n else np.nan,
        pieces_mean=float(np.mean(pieces)) if n else np.nan,
        pieces_max=int(np.max(pieces)) if n else 0,
        dwell_mae_s=float(np.mean(errs)) if n else np.nan,
        dwell_rel_err_median=float(np.median(rels)) if n else np.nan,
        dwell_rel_err_mean=float(np.mean(rels)) if n else np.nan,
        dwell_ratio_median=float(np.median(ratios)) if n else np.nan,
        n_test_stops=int(len(ts)),
        n_test_stops_unmatched=int(len(ts) - len(used_test)),
        per_stop=per_stop,
    )
    ev = {}
    tp_all = nr_all = nt_all = 0
    for typ in event_types:
        re_ = ref.events[(ref.events.type == typ) & ref.events.gt_id.notna()]
        te_ = test.events[(test.events.type == typ) & test.events.gt_id.notna()]
        tp, man_err = 0, []
        for gid in set(re_.gt_id) | set(te_.gt_id):
            a = re_[re_.gt_id == gid].sort_values("t")
            b = te_[te_.gt_id == gid].sort_values("t")
            mm = _match_events(a.t.tolist(), b.t.tolist(), tol_s)
            tp += len(mm)
            for i, j in mm:
                ra, tb = a.iloc[i], b.iloc[j]
                if typ in ("stop_start", "stop_end") and not ra.censored and not tb.censored:
                    man_err.append(abs(ra.maneuver_s - tb.maneuver_s))
        nr, nt = len(re_), len(te_)
        p = tp / nt if nt else np.nan
        r_ = tp / nr if nr else np.nan
        ev[typ] = dict(n_ref=nr, n_test=nt, tp=tp, precision=p, recall=r_,
                       f1=(2 * p * r_ / (p + r_)) if nt and nr and (p + r_) > 0 else np.nan,
                       maneuver_mae_s=float(np.mean(man_err)) if man_err else np.nan,
                       n_maneuver_pairs=len(man_err))
        tp_all += tp; nr_all += nr; nt_all += nt
    P = tp_all / nt_all if nt_all else np.nan
    R = tp_all / nr_all if nr_all else np.nan
    ev["all"] = dict(n_ref=nr_all, n_test=nt_all, tp=tp_all, precision=P, recall=R,
                     f1=(2 * P * R / (P + R)) if nt_all and nr_all and (P + R) > 0 else np.nan)
    out["events"] = ev
    return out


def oracle_result(gt_df, fps, clip, **params):
    """Behavior of the pseudo-GT vehicles themselves (id == gt_id), with gt_id columns attached."""
    res = analyze(gt_df, fps, id_col="gt_id", clip=clip, **params)
    for d in (res.tracks, res.stops, res.events):
        d["gt_id"] = d["id"] if len(d) else []
    return res
