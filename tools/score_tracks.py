"""score_tracks.py - one command to score a tracks file: relink it, then compare per-vehicle behaviour
before and after relinking.

    python tools/score_tracks.py better_tests/tracksid3duplicate.csv --fps 60000/1001 --width 1920 --height 1080
    python tools/score_tracks.py my_tracks.txt --video my_clip.mp4 --out-dir out/my_clip
    python tools/score_tracks.py better_tests/airport_tracks.csv --scene airport     # + scores vs repo labels

Input: any file relinker_v2.read_tracks accepts (tracker CSV with a header, headerless CSV or
MOTChallenge txt; column names and box format are auto-detected) plus the video's fps and frame size
(--fps/--width/--height, or --video to read them from the clip; --fps accepts "60000/1001").
Frame numbers must be the ORIGINAL video frame numbers, also for frame-subsampled trackers.

Steps
  1. read the tracks and relink them with relinker_v2 (model / threshold / merge default to the
     settings stored in the model bundle, tools/models/relinker_v2_both.joblib);
  2. run behavior.analyze on the raw track IDs and on the relinked vehicle IDs (stops = stationary
     < 0.1 box lengths/s for >= 2 s; arrival / departure = track starts / ends away from the clip
     edges; stop_start / stop_end = observed inside a track);
  3. print a short summary: raw IDs vs relinked vehicles, share of short IDs (< 2 s), accepted links,
     events and dwell, per-vehicle stops;
  4. with --scene parking|airport (only for the repo's two labelled clips, i.e. other trackers run on
     assets/00009_Trim.mp4 or assets/Airport_DropOff_Footage_STOCK.mp4): IDF1 and event F1 against the
     repo's partial pseudo ground truth (mot_eval.py), raw vs relinked.

Outputs in --out-dir (default ./score_out/<input name>/):
  relinked.csv      input rows + tracklet_id + vehicle_id
  vehicles.csv      one row per relinked vehicle (first/last seen, stops, dwell, raw IDs it merged)
  summary.json      everything that is printed

Caveats
  * Behaviour numbers are only as good as the identities: without relinking a fragmented track turns
    one parked car into several arrivals / departures and cuts its dwell into pieces.
  * --scene scores use PARTIAL labels (10 parking / 21 airport vehicles): boxes that match no
    labelled vehicle are ignored, so a wrong merge into an unlabelled car is invisible to IDF1.
  * The default model was trained on both labelled clips, so --scene scores with it are in-sample.
    For honest numbers use the other scene's model, e.g. on parking
    --model tools/models/relinker_v2_airport.joblib.
"""
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction

import numpy as np
import pandas as pd

try:                                   # imported as tools.score_tracks (from the repo root)
    from . import behavior as bh
    from . import relinker_v2 as rl
except ImportError:                    # run as a script / imported from inside tools/
    import behavior as bh
    import relinker_v2 as rl

SHORT_S = 2.0                          # an ID seen for less than this many seconds counts as "short"
EVENT_TYPES = ("arrival", "departure", "stop_start", "stop_end")


# ----------------------------------------------------------------------------- helpers
def parse_fps(s):
    """'29.97', '30' or '30000/1001' -> float."""
    return float(Fraction(s)) if "/" in str(s) else float(s)


def video_meta(path):
    """(fps, width, height) of a video, read with OpenCV."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def behaviour_summary(res):
    """Headline numbers of one behavior.analyze result."""
    tr, st, ev = res.tracks, res.stops, res.events
    n = len(tr)
    counts = ev.type.value_counts() if len(ev) else pd.Series(dtype=int)
    return dict(
        n_ids=int(n),
        n_short_ids=int((tr.duration_s < SHORT_S).sum()) if n else 0,
        short_id_share=float((tr.duration_s < SHORT_S).mean()) if n else float("nan"),
        median_id_duration_s=float(tr.duration_s.median()) if n else float("nan"),
        events={t: int(counts.get(t, 0)) for t in EVENT_TYPES},
        n_stops=int(len(st)),
        n_ids_with_stop=int(st.id.nunique()) if len(st) else 0,
        dwell_median_s=float(st.dwell_s.median()) if len(st) else float("nan"),
        dwell_max_s=float(st.dwell_s.max()) if len(st) else float("nan"),
        dwell_total_s=float(st.dwell_s.sum()) if len(st) else 0.0,
    )


def per_vehicle_table(res, rel):
    """One row per relinked vehicle: behaviour + which raw IDs / tracklets it is made of."""
    tr = res.tracks.rename(columns={"id": "vehicle_id"}).copy()
    comp = rel.groupby("vehicle_id").agg(n_raw_ids=("track_id", "nunique"), n_tracklets=("tracklet_id", "nunique"),
                                         raw_ids=("track_id", lambda s: " ".join(str(x) for x in pd.unique(s))))
    tr = tr.merge(comp, left_on="vehicle_id", right_index=True, how="left")
    st = res.stops
    ivl = {}
    for vid, g in (st.groupby("id") if len(st) else []):
        ivl[vid] = "; ".join(f"{a:.1f}-{b:.1f}s ({d:.1f}s)" for a, b, d in zip(g.t_start, g.t_end, g.dwell_s))
    tr["stops"] = tr.vehicle_id.map(ivl).fillna("")
    cols = ["vehicle_id", "first_s", "last_s", "duration_s", "present_at_start", "present_at_end", "n_stops",
            "dwell_total_s", "stops", "n_raw_ids", "n_tracklets", "raw_ids", "first_frame", "last_frame",
            "n_samples", "median_speed_Lps"]
    return tr[cols].sort_values("first_s").reset_index(drop=True)


# ----------------------------------------------------------------------------- labelled clips
def score_against_labels(df, rel, scene, fps, tol_s=2.0):
    """IDF1 (mot_eval) and behaviour-event F1 (behavior.compare vs the pseudo-GT vehicles) for the raw
    IDs and the relinked vehicles.  Frames: every frame the tracker analysed inside the labelled clip
    (stride-aware), as in the feasibility study, so a frame with no output still counts its misses."""
    try:
        from . import mot_eval as me
    except ImportError:
        import mot_eval as me
    S = me.SCENES[scene]
    if abs(fps - S["fps"]) > 0.01:
        print(f"WARNING: --fps {fps:.3f} differs from the {scene} clip's {S['fps']:.3f} fps; "
              "is this really a track file of that clip?")
    raw_repo = me.load_tracks(scene)
    clip = (int(raw_repo.frame.min()), int(raw_repo.frame.max()))
    frames = np.arange(1, S["n_frames"] + 1, rl.frame_step(df))
    frames = frames[(frames >= clip[0]) & (frames <= clip[1])]
    oracle = bh.oracle_result(me.load_gt(scene), S["fps"], clip)
    out = {}
    for name, d, col in [("raw", df, "track_id"), ("relinked", rel, "vehicle_id")]:
        d = d[d.frame.isin(set(frames.tolist()))]
        ev = me.evaluate(d, scene, id_col=col, frames=frames, also_iou=())
        res = bh.attribute_to_gt(bh.analyze(d, S["fps"], id_col=col, clip=clip), d, scene, id_col=col)
        c = bh.compare(oracle, res, tol_s=tol_s)
        E = c["events"]["all"]
        out[name] = dict(idf1=ev["idf1"], coverage=ev["coverage"], id_switches=ev["id_switches"],
                         ids_per_gt_vehicle=ev["frag_mean"], event_precision=E["precision"],
                         event_recall=E["recall"], event_f1=E["f1"],
                         events_by_type={t: {k: c["events"][t][k] for k in ("n_ref", "n_test", "tp")}
                                         for t in EVENT_TYPES},
                         stop_recall=c["dwell"]["stop_recall"], dwell_mae_s=c["dwell"]["dwell_mae_s"],
                         behaviour_count_inflation=c["count_inflation"])
    out["n_gt_vehicles"] = int(ev["n_gt_vehicles"])
    out["n_frames_eval"] = int(len(frames))
    return out


# ----------------------------------------------------------------------------- printing
def _fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def print_summary(s, vehicles, max_rows=15):
    r, v, lk = s["raw"], s["relinked"], s["relink"]
    print(f"\n{s['input']}: {s['n_rows']} boxes, frames {s['frames'][0]}-{s['frames'][1]} "
          f"({s['clip_s']:.1f} s at {s['fps']:.3f} fps, frame step {s['frame_step']})")
    print(f"relinker: {lk['tracklets']} tracklets, {lk['candidate_pairs']} candidate pairs, "
          f"{lk['accepted_links']} accepted links -> {lk['vehicles']} vehicles "
          f"(model {lk['model']}, threshold {lk['threshold']}, merge {lk['merge']})")
    rows = [("IDs / vehicles", r["n_ids"], v["n_ids"]),
            (f"short (< {SHORT_S:g} s)", f"{r['n_short_ids']} ({100 * r['short_id_share']:.0f}%)",
             f"{v['n_short_ids']} ({100 * v['short_id_share']:.0f}%)"),
            ("median ID duration (s)", _fmt(r["median_id_duration_s"], 1), _fmt(v["median_id_duration_s"], 1))]
    rows += [(t.replace("_", " ") + "s", r["events"][t], v["events"][t]) for t in EVENT_TYPES]
    rows += [("stops (>= 2 s)", r["n_stops"], v["n_stops"]),
             ("dwell median / max (s)", f"{_fmt(r['dwell_median_s'], 1)} / {_fmt(r['dwell_max_s'], 1)}",
              f"{_fmt(v['dwell_median_s'], 1)} / {_fmt(v['dwell_max_s'], 1)}")]
    print(f"\n{'':26s}{'raw IDs':>16s}{'relinked':>16s}")
    for name, a, b in rows:
        print(f"{name:26s}{str(a):>16s}{str(b):>16s}")
    stopped = vehicles[vehicles.n_stops > 0]
    print(f"\nrelinked vehicles with a stop ({len(stopped)} of {len(vehicles)}; all in vehicles.csv):")
    for x in stopped.head(max_rows).itertuples():
        ids = x.raw_ids if len(x.raw_ids) <= 30 else x.raw_ids[:27] + "..."
        print(f"  vehicle {x.vehicle_id:>4}: seen {x.first_s:6.1f}-{x.last_s:6.1f} s, stops {x.stops}  "
              f"[raw IDs {ids}]")
    if len(stopped) > max_rows:
        print(f"  ... {len(stopped) - max_rows} more")
    if "labels" in s:
        L = s["labels"]
        print(f"\nvs repo labels ({s['scene']}: {L['n_gt_vehicles']} labelled vehicles, partial GT, "
              f"{L['n_frames_eval']} frames):")
        print(f"{'':26s}{'raw IDs':>16s}{'relinked':>16s}")
        for key, name in [("idf1", "IDF1"), ("coverage", "coverage"), ("id_switches", "ID switches"),
                          ("ids_per_gt_vehicle", "IDs per labelled vehicle"), ("event_precision", "event precision"),
                          ("event_recall", "event recall"), ("event_f1", "event F1 (+-2 s)"),
                          ("stop_recall", "stop recall"), ("dwell_mae_s", "dwell MAE (s)")]:
            a, b = L["raw"][key], L["relinked"][key]
            fa = str(a) if isinstance(a, int) else _fmt(a, 3)
            fb = str(b) if isinstance(b, int) else _fmt(b, 3)
            print(f"{name:26s}{fa:>16s}{fb:>16s}")
        if s["relink"]["model"] == "relinker_v2_both.joblib":
            other = {"parking": "airport", "airport": "parking"}[s["scene"]]
            print(f"note: relinker_v2_both was trained on this clip (in-sample); for a cross-scene score "
                  f"use --model tools/models/relinker_v2_{other}.joblib")


# ----------------------------------------------------------------------------- main
def score(path, fps, width, height, model=None, threshold=None, merge=None, scene=None, box_format="auto"):
    """Run the whole pipeline; returns (summary dict, relinked DataFrame, per-vehicle DataFrame)."""
    df = rl.read_tracks(path, box_format=box_format, frame_w=width, frame_h=height)
    bundle = rl.load_model(model)
    cfg = {**rl.DEFAULT_CONFIG, **bundle.get("config", {})}
    links, t2v, chains = rl.relink(df, fps, width, height, bundle=bundle, threshold=threshold, merge=merge)
    rel = rl.apply_relink(df, t2v)
    rel = rel[rel.vehicle_id >= 0]            # every tracklet is mapped; kept as a safety net
    clip = (int(df.frame.min()), int(df.frame.max()))
    res_raw = bh.analyze(df, fps, id_col="track_id", clip=clip)
    res_rel = bh.analyze(rel, fps, id_col="vehicle_id", clip=clip)
    vehicles = per_vehicle_table(res_rel, rel)
    s = dict(
        input=os.path.basename(path), n_rows=int(len(df)), frames=list(clip), fps=float(fps),
        width=width, height=height, frame_step=rl.frame_step(df), clip_s=(clip[1] - clip[0] + 1) / fps,
        relink=dict(model=os.path.basename(model) if model else "relinker_v2_both.joblib",
                    threshold=threshold if threshold is not None else cfg["threshold"],
                    merge=merge or cfg["merge"], assembly=cfg["assembly"], tracklets=len(t2v),
                    candidate_pairs=int(len(links)), accepted_links=int(links["accepted"].sum()) if len(links) else 0,
                    vehicles=len(chains)),
        raw=behaviour_summary(res_raw), relinked=behaviour_summary(res_rel),
        behaviour_params=res_raw.params,
    )
    if scene:
        s["scene"] = scene
        s["labels"] = score_against_labels(df, rel, scene, fps)
    return s, rel, vehicles


def main(argv=None):
    ap = argparse.ArgumentParser(description="Relink a tracks file and compare per-vehicle behaviour raw vs relinked")
    ap.add_argument("tracks", help="tracker CSV / MOTChallenge txt")
    ap.add_argument("--fps", help="video fps, e.g. 29.97 or 30000/1001")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--video", help="read fps / width / height from this video (needs opencv)")
    ap.add_argument("--scene", choices=["parking", "airport"],
                    help="the input is a track file of one of the repo's labelled clips: also score it "
                         "against the repo labels (fps / size default to that clip's)")
    ap.add_argument("--model", default=None, help="relinker bundle (default tools/models/relinker_v2_both.joblib)")
    ap.add_argument("--threshold", type=float, default=None, help="link probability threshold (default: bundle's, 0.5)")
    ap.add_argument("--merge", choices=["none", "flicker2", "sameid"], default=None,
                    help="chain merge rule (default: bundle's, sameid)")
    ap.add_argument("--box-format", default="auto", choices=["auto", "xyxy", "tlwh", "cxcywh"])
    ap.add_argument("--out-dir", default=None, help="output folder (default ./score_out/<input name>)")
    a = ap.parse_args(argv)

    fps = parse_fps(a.fps) if a.fps else None
    W, H = a.width, a.height
    if a.video:
        vf, vw, vh = video_meta(a.video)
        fps, W, H = fps or vf, W or vw, H or vh
    if a.scene:
        try:
            from . import mot_eval as me
        except ImportError:
            import mot_eval as me
        S = me.SCENES[a.scene]
        fps, W, H = fps or S["fps"], W or S["width"], H or S["height"]
    if not fps or not W or not H:
        ap.error("need --fps, --width and --height (or --video / --scene)")

    s, rel, vehicles = score(a.tracks, fps, W, H, model=a.model, threshold=a.threshold, merge=a.merge,
                             scene=a.scene, box_format=a.box_format)
    print_summary(s, vehicles)

    out = a.out_dir or os.path.join("score_out", os.path.splitext(os.path.basename(a.tracks))[0])
    os.makedirs(out, exist_ok=True)
    rel.to_csv(os.path.join(out, "relinked.csv"), index=False)
    vehicles.to_csv(os.path.join(out, "vehicles.csv"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(jsonable(s), fh, indent=1)
    print(f"\nwrote {out}/relinked.csv, vehicles.csv, summary.json")


if __name__ == "__main__":
    main()
