"""mot_eval.py - tracker-independent evaluation against the repo's partial pseudo ground truth.

Pseudo ground truth (GT)
------------------------
The repo has no hand-drawn boxes. Its labels (hard-coded in ``relinker_script.py``) say which
DeepSORT IDs / tracklets are the same physical vehicle.  We turn them into per-frame GT boxes with a
*vehicle* identity:

* parking  (``better_tests/tracksid3duplicate.csv`` <-> ``assets/00009_Trim.mp4``, 59.94 fps):
  every tracklet of every base ID in a ``parking_vehicles`` group -> that group's vehicle
  (``parking_lot_not_same`` only says ID 238 is *not* vehicle 4; 238 is otherwise unlabeled, so it is
  simply left out of the GT).
* airport  (``better_tests/airport_tracks.csv`` <-> ``assets/Airport_DropOff_Footage_STOCK.mp4``,
  23.976 fps): ``airport_chains`` are mapped to tracklets with the relinker's own
  ``split_into_tracklets`` (gap > 6) + ``map_chains_to_tracklets`` (IDs 16 and 36 appear in two chains;
  the mapping gives each chain a *different* tracklet: 16_0 / 16_1, 36_0 / 36_1, so no tracklet is in two
  chains).  Every tracklet of an ``airport_isolated`` ID becomes one vehicle.  Tracklets of chain IDs
  that the chain mapping did not pick (e.g. 21_1, 21_3, 21_4) are *unlabeled*.
* When one GT vehicle has two boxes in the same frame (two of its DeepSORT IDs alive at once, e.g. a
  duplicate track), we keep the box of the longer tracklet (more rows) and drop the other.  The
  count is reported by ``gt_summary``.

Caveats: the GT is PARTIAL (only some vehicles are labeled) and its boxes are one DeepSORT run's
boxes, so any tracker whose boxes resemble that run's will match more easily.  Predictions that
match no GT box are therefore IGNORED (they may be real but unlabeled vehicles).

API
---
``load_gt(scene)``                     -> DataFrame[frame, gt_id, x1, y1, x2, y2, src_track_id, src_tracklet]
``gt_summary(scene)``                  -> dict of coverage numbers
``evaluate(pred_df, scene, iou=0.5)``  -> dict of metrics (see ``evaluate`` docstring)
``assign_tracks_to_gt(pred_df, scene)``-> DataFrame, one row per predicted id: majority GT vehicle
``match_frames(pred_df, gt_df, iou)``  -> per-frame matches (DataFrame[frame, gt_id, pred_id, iou])
``SCENES``                             -> dict with csv / video / fps / frame range per scene

Validation: on the repo CSVs IDF1 and ID switches agree exactly with py-motmetrics 1.4.0 (raw DeepSORT
IDs: parking IDF1 0.885 / 50 switches, airport 0.736 / 44), per-vehicle fragmentation equals the known
label group sizes, and synthetic split / swap tests give the analytic IDF1.

Frame numbers are the repo's 1-based frame numbers.  ``pred_df`` needs columns
``frame, x1, y1, x2, y2`` plus an id column (``track_id`` by default; pass ``id_col='tracklet_id'``
etc.).  IDs may be ints or strings.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # repo root = parent of tools/

SCENES = {
    "parking": dict(csv=os.path.join(REPO, "better_tests", "tracksid3duplicate.csv"),
                    video=os.path.join(REPO, "assets", "00009_Trim.mp4"), fps=60000 / 1001, width=1920, height=1080,
                    n_frames=7390),
    "airport": dict(csv=os.path.join(REPO, "better_tests", "airport_tracks.csv"),
                    video=os.path.join(REPO, "assets", "Airport_DropOff_Footage_STOCK.mp4"), fps=24000 / 1001,
                    width=1280, height=720, n_frames=469),
}

# Visually verified label correction (checked by eye on frames 7240, 7300, 7351 and 7380 of
# assets/00009_Trim.mp4, plus the box coordinates): DeepSORT ID 282 is in the White SUV group, but its
# last two tracklets (frames 7286-7311 and 7351-7390, x~196 px, static) sit on a car parked behind the
# white minivan while the White SUV exits left as IDs 294/296/298/302.  Keeping them would make the
# White SUV "park" at the end of the clip.  load_gt(scene, corrected=False) reproduces the uncorrected
# labels.
GT_EXCLUDE_TRACKLETS = {"parking": {"282_2", "282_3"}, "airport": set()}

PARKING_NAMES = ["Black SUV", "White SUV", "Vehicle A", "Stationary(4,263,287)",
                 "Iso 1", "Iso 3", "Iso 6", "Iso 7", "Iso 8", "Iso 9"]


# ----------------------------------------------------------------------------- repo label access
def _repo_labels():
    """Read the GT constants and the two helper functions from relinker_script.py WITHOUT running it
    (the script trains a model at import).  Uses ``ast`` so it stays in sync with the repo."""
    with open(os.path.join(REPO, "relinker_script.py")) as fh:
        src = fh.read()
    tree = ast.parse(src)
    consts, funcs = {}, []
    want_c = {"parking_vehicles", "parking_lot_not_same", "airport_chains", "airport_isolated"}
    want_f = {"split_into_tracklets", "map_chains_to_tracklets"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and node.targets[0].id in want_c:
            consts[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name in want_f:
            funcs.append(node)
    ns = {"pd": pd, "np": np, "defaultdict": defaultdict, "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "relinker_funcs", "exec"), ns)
    return consts, ns["split_into_tracklets"], ns["map_chains_to_tracklets"]


def load_tracks(scene):
    """Raw repo DeepSORT CSV for a scene with a ``tracklet_id`` column (gap>6 split, as the relinker)."""
    _, split, _ = _repo_labels()
    df = pd.read_csv(SCENES[scene]["csv"])
    return split(df.copy(), max_gap=6)


def _build_gt(scene, corrected=True):
    consts, split, map_chains = _repo_labels()
    df = pd.read_csv(SCENES[scene]["csv"])
    ds = split(df.copy(), max_gap=6)
    t2v, names = {}, {}
    if scene == "parking":
        for v, ids in enumerate(consts["parking_vehicles"]):
            for t in ds.loc[ds.track_id.isin(ids), "tracklet_id"].unique():
                t2v[t] = v
            names[v] = PARKING_NAMES[v] if v < len(PARKING_NAMES) else f"group{v}"
    elif scene == "airport":
        chains = map_chains(consts["airport_chains"], ds)
        assert len(chains) == len(consts["airport_chains"]), "a chain failed to map"
        for v, ch in enumerate(chains):
            for t in ch:
                assert t not in t2v, f"tracklet {t} in two chains"
                t2v[t] = v
            names[v] = "chain " + "-".join(str(i) for i in consts["airport_chains"][v])
        v0 = len(chains)
        for k, i in enumerate(sorted(consts["airport_isolated"])):
            for t in ds.loc[ds.track_id == i, "tracklet_id"].unique():
                assert t not in t2v
                t2v[t] = v0 + k
            names[v0 + k] = f"isolated {i}"
    else:
        raise ValueError(scene)
    if corrected:
        for t in GT_EXCLUDE_TRACKLETS.get(scene, ()):
            t2v.pop(t, None)
    g = ds[ds.tracklet_id.isin(t2v)].copy()
    g["gt_id"] = g.tracklet_id.map(t2v).astype(int)
    tl_len = g.groupby("tracklet_id").size()
    g["_len"] = g.tracklet_id.map(tl_len)
    g = g.sort_values(["gt_id", "frame", "_len"], ascending=[True, True, False])
    dup_mask = g.duplicated(["gt_id", "frame"], keep="first")
    dups = g[dup_mask | g.duplicated(["gt_id", "frame"], keep=False)]
    # IoU between the kept box and each dropped box
    dup_ious = []
    for (_, _), grp in dups.groupby(["gt_id", "frame"]):
        b = grp[["x1", "y1", "x2", "y2"]].to_numpy(float)
        for j in range(1, len(b)):
            dup_ious.append(float(iou_matrix(b[:1], b[j:j + 1])[0, 0]))
    out = g[~dup_mask].rename(columns={"track_id": "src_track_id", "tracklet_id": "src_tracklet"})
    out = out[["frame", "gt_id", "x1", "y1", "x2", "y2", "src_track_id", "src_tracklet"]]
    out = out.sort_values(["frame", "gt_id"]).reset_index(drop=True)
    info = dict(n_dropped_duplicate_boxes=int(dup_mask.sum()),
                dup_frames_by_vehicle={names[int(k)]: int(v) for k, v in
                                       g[dup_mask].groupby("gt_id").size().items()},
                dup_iou_median=float(np.median(dup_ious)) if dup_ious else None,
                dup_iou_min=float(np.min(dup_ious)) if dup_ious else None,
                dup_iou_frac_below_0_5=float(np.mean(np.array(dup_ious) < 0.5)) if dup_ious else None)
    return out, names, info, ds, t2v


_CACHE = {}


def _gt_bundle(scene, corrected=True):
    key = (scene, bool(corrected))
    if key not in _CACHE:
        _CACHE[key] = _build_gt(scene, corrected)
    return _CACHE[key]


def load_gt(scene, corrected=True):
    """Per-frame pseudo-GT boxes: DataFrame[frame, gt_id, x1, y1, x2, y2, src_track_id, src_tracklet].
    ``corrected`` applies GT_EXCLUDE_TRACKLETS (default True)."""
    return _gt_bundle(scene, corrected)[0].copy()


def gt_names(scene):
    """{gt_id: human-readable name}."""
    return dict(_gt_bundle(scene)[1])


def gt_summary(scene, corrected=True):
    """Coverage of the pseudo GT relative to the raw DeepSORT CSV (vehicles, IDs, tracklets, frames, boxes)."""
    gt, names, info, ds, t2v = _gt_bundle(scene, corrected)
    per_v = gt.groupby("gt_id").agg(first=("frame", "min"), last=("frame", "max"), boxes=("frame", "size"),
                                    ids=("src_track_id", "nunique"), tracklets=("src_tracklet", "nunique"))
    return dict(
        scene=scene,
        gt_vehicles=int(gt.gt_id.nunique()),
        gt_boxes=int(len(gt)),
        raw_boxes=int(len(ds)),
        gt_box_fraction_of_raw=float(len(gt) / len(ds)),
        frames_with_gt=int(gt.frame.nunique()),
        raw_frames=int(ds.frame.nunique()),
        frame_range=[int(ds.frame.min()), int(ds.frame.max())],
        deep_sort_ids_labeled=int(gt.src_track_id.nunique()), deep_sort_ids_total=int(ds.track_id.nunique()),
        tracklets_labeled=int(len(t2v)), tracklets_total=int(ds.tracklet_id.nunique()),
        mean_gt_boxes_per_frame=float(gt.groupby("frame").size().mean()),
        mean_raw_boxes_per_frame=float(ds.groupby("frame").size().mean()),
        duplicates=info,
        per_vehicle={names[int(k)]: {kk: int(vv) for kk, vv in r.items()} for k, r in per_v.iterrows()},
    )


# ----------------------------------------------------------------------------- matching
def iou_matrix(a, b):
    """IoU between boxes a (N,4) and b (M,4) in x1,y1,x2,y2."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_frames(pred_df, gt_df, iou=0.5, id_col="track_id", frames=None, sticky=True):
    """Per-frame one-to-one matching of predicted boxes to GT boxes (IoU >= ``iou``).

    CLEAR-MOT style: if ``sticky`` a GT box keeps last frame's predicted id when that id is present
    with IoU >= thr; the remaining boxes are matched with the Hungarian algorithm on (1 - IoU).
    Returns DataFrame[frame, gt_id, pred_id, iou] (matched pairs only)."""
    if frames is None:
        frames = np.unique(pred_df["frame"].to_numpy())
    frames = np.sort(np.asarray(list(frames)))
    P = pred_df[pred_df.frame.isin(frames)].sort_values("frame", kind="stable")
    G = gt_df[gt_df.frame.isin(frames)].sort_values("frame", kind="stable")
    # numpy views + per-frame slices (pandas per-frame indexing is ~50x slower)
    pf, gf = P.frame.to_numpy(), G.frame.to_numpy()
    PB, GB = P[["x1", "y1", "x2", "y2"]].to_numpy(float), G[["x1", "y1", "x2", "y2"]].to_numpy(float)
    PI, GI = P[id_col].to_numpy(), G["gt_id"].to_numpy()
    last = {}
    rows = []
    for f in frames:
        pa, pb_ = np.searchsorted(pf, f, "left"), np.searchsorted(pf, f, "right")
        ga, gb_ = np.searchsorted(gf, f, "left"), np.searchsorted(gf, f, "right")
        if pa == pb_ or ga == gb_:
            continue
        gb, pb = GB[ga:gb_], PB[pa:pb_]
        gids, pids = GI[ga:gb_], PI[pa:pb_]
        M = iou_matrix(gb, pb)
        used_g, used_p = set(), set()
        if sticky:
            pidx = defaultdict(list)
            for j, pid in enumerate(pids):
                pidx[pid].append(j)
            for i, gid in enumerate(gids):
                if gid in last and last[gid] in pidx:
                    js = [j for j in pidx[last[gid]] if j not in used_p]
                    if js:
                        j = max(js, key=lambda jj: M[i, jj])
                        if M[i, j] >= iou:
                            used_g.add(i); used_p.add(j)
                            rows.append((f, gid, pids[j], M[i, j]))
        gi = [i for i in range(len(gids)) if i not in used_g]
        pj = [j for j in range(len(pids)) if j not in used_p]
        if gi and pj:
            sub = M[np.ix_(gi, pj)]
            cost = np.where(sub >= iou, 1 - sub, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if sub[a, b] >= iou:
                    rows.append((f, gids[gi[a]], pids[pj[b]], sub[a, b]))
        # update sticky memory (GT ids unmatched in this frame keep their previous pred id)
        k = len(rows) - 1
        while k >= 0 and rows[k][0] == f:
            last[rows[k][1]] = rows[k][2]
            k -= 1
    return pd.DataFrame(rows, columns=["frame", "gt_id", "pred_id", "iou"])


def _idf1(m, n_gt, n_pred):
    """Identity F1 given matched pairs m (frame, gt_id, pred_id). Global bipartite id matching."""
    if len(m) == 0:
        return 0.0, 0.0, 0.0, 0
    co = m.groupby(["gt_id", "pred_id"]).size()
    gids = sorted(co.index.get_level_values(0).unique(), key=str)
    pids = sorted(co.index.get_level_values(1).unique(), key=str)
    gi = {g: i for i, g in enumerate(gids)}
    pi = {p: j for j, p in enumerate(pids)}
    W = np.zeros((len(gids), len(pids)))
    for (g, p), v in co.items():
        W[gi[g], pi[p]] = v
    r, c = linear_sum_assignment(-W)
    idtp = int(W[r, c].sum())
    idp = idtp / n_pred if n_pred else 0.0
    idr = idtp / n_gt if n_gt else 0.0
    idf1 = 2 * idtp / (n_gt + n_pred) if (n_gt + n_pred) else 0.0
    return idf1, idp, idr, idtp


def evaluate(pred_df, scene, iou=0.5, id_col="track_id", frames=None, also_iou=(0.3,), min_frames_frag=1,
             corrected=True):
    """Evaluate a tracker output against the partial pseudo GT of ``scene``.

    Only frames present in ``pred_df`` are evaluated (or the explicit ``frames`` iterable - pass it
    when a frame-subsampled tracker legitimately output nothing on some analysed frames).  Predicted
    boxes that match no GT box are ignored (partial GT).

    Returns a dict:
      coverage            matched GT boxes / GT boxes in evaluated frames
      idf1, idp, idr      identity F1 over GT boxes and the predicted boxes that matched a GT box
                          (IDF1 = 2*IDTP / (#GT boxes + #matched pred boxes))
      id_switches         CLEAR-MOT ID switches (sticky matching; change of the pred id matched to a GT
                          vehicle relative to its previous matched pred id)
      frag_mean/median/max  distinct predicted ids matched to each GT vehicle (>= ``min_frames_frag`` frames)
      interruptions       times a GT vehicle's matched run is interrupted and resumes (motmetrics 'frag')
      count_inflation     distinct predicted ids matched to any GT vehicle / number of GT vehicles
      n_gt_vehicles, n_gt_boxes, n_frames_eval, n_pred_matched_ids
      per_vehicle         {gt_id: {'frag': k, 'matched': n, 'boxes': n}}
      iou_<t>             the same headline numbers at the alternative IoU thresholds in ``also_iou``
    """
    gt = load_gt(scene, corrected)
    if frames is None:
        frames = np.unique(pred_df["frame"].to_numpy())
    frames = np.asarray(sorted(set(int(f) for f in frames)))
    res = _evaluate_core(pred_df, gt, frames, iou, id_col, min_frames_frag)
    for t in also_iou:
        r2 = _evaluate_core(pred_df, gt, frames, t, id_col, min_frames_frag)
        res[f"iou_{t}"] = {k: r2[k] for k in ["coverage", "idf1", "id_switches", "frag_mean", "frag_median",
                                              "frag_max", "count_inflation"]}
    res["scene"] = scene
    res["iou"] = iou
    return res


def _evaluate_core(pred_df, gt, frames, iou, id_col, min_frames_frag):
    G = gt[gt.frame.isin(frames)]
    m = match_frames(pred_df, gt, iou=iou, id_col=id_col, frames=frames)
    n_gt = len(G)
    n_match = len(m)
    idf1, idp, idr, idtp = _idf1(m, n_gt, n_match)
    # id switches + interruptions
    idsw = 0
    interrupts = 0
    frame_pos = {f: i for i, f in enumerate(frames)}
    for gid, grp in m.sort_values("frame").groupby("gt_id"):
        prev = None
        seq = grp.pred_id.tolist()
        for pid in seq:
            if prev is not None and pid != prev:
                idsw += 1
            prev = pid
        # interruptions: GT present in evaluated frame but unmatched between two matched frames
        gframes = G.loc[G.gt_id == gid, "frame"].to_numpy()
        matched = set(grp.frame.tolist())
        state = []
        for f in gframes:
            state.append(f in matched)
        s = np.array(state, bool)
        if s.any():
            first, lastm = np.argmax(s), len(s) - 1 - np.argmax(s[::-1])
            seg = s[first:lastm + 1]
            interrupts += int(np.sum((~seg[:-1]) & seg[1:])) if len(seg) > 1 else 0
    cnt = m.groupby(["gt_id", "pred_id"]).size()
    cnt = cnt[cnt >= min_frames_frag]
    frag = cnt.groupby(level=0).size()
    gt_ids = sorted(G.gt_id.unique())
    frag_full = pd.Series({g: int(frag.get(g, 0)) for g in gt_ids})
    matched_ids = set(cnt.index.get_level_values(1))
    per_vehicle = {}
    for g in gt_ids:
        per_vehicle[int(g)] = dict(frag=int(frag_full[g]), matched=int((m.gt_id == g).sum()),
                                   boxes=int((G.gt_id == g).sum()))
    nv = len(gt_ids)
    return dict(
        coverage=float(n_match / n_gt) if n_gt else float("nan"),
        idf1=float(idf1), idp=float(idp), idr=float(idr), idtp=int(idtp),
        id_switches=int(idsw), interruptions=int(interrupts),
        frag_mean=float(frag_full.mean()) if nv else float("nan"),
        frag_median=float(frag_full.median()) if nv else float("nan"),
        frag_max=int(frag_full.max()) if nv else 0,
        count_inflation=float(len(matched_ids) / nv) if nv else float("nan"),
        n_gt_vehicles=int(nv), n_gt_boxes=int(n_gt), n_matched=int(n_match),
        n_frames_eval=int(len(frames)), n_pred_matched_ids=int(len(matched_ids)),
        per_vehicle=per_vehicle,
    )


def assign_tracks_to_gt(pred_df, scene, iou=0.5, id_col="track_id", frames=None, corrected=True):
    """Map each predicted track / tracklet to the GT vehicle it matches most often.

    Returns DataFrame[pred_id, gt_id, n_rows, n_matched, n_majority, purity, match_frac] with
      n_rows      boxes of the predicted id in evaluated frames
      n_matched   of those, boxes matched to ANY GT vehicle
      n_majority  boxes matched to the majority GT vehicle
      purity      n_majority / n_matched  (1.0 = never matched to another vehicle)
      match_frac  n_majority / n_rows     (low = mostly on unlabeled vehicles / background)
    ``gt_id`` is NaN for ids that never match a GT box (unlabeled - do not treat as negatives
    blindly: the GT is partial).  Suggested label-transfer rule: trust rows with
    purity >= 0.9 and match_frac >= 0.5."""
    gt = load_gt(scene, corrected)
    if frames is None:
        frames = np.unique(pred_df["frame"].to_numpy())
    P = pred_df[pred_df.frame.isin(frames)]
    m = match_frames(P, gt, iou=iou, id_col=id_col, frames=frames)
    n_rows = P.groupby(id_col).size()
    co = m.groupby(["pred_id", "gt_id"]).size().reset_index(name="n")
    rows = []
    for pid, nr in n_rows.items():
        c = co[co.pred_id == pid]
        if len(c) == 0:
            rows.append((pid, np.nan, int(nr), 0, 0, np.nan, 0.0))
            continue
        best = c.loc[c.n.idxmax()]
        nm = int(c.n.sum())
        rows.append((pid, int(best.gt_id), int(nr), nm, int(best.n), best.n / nm, best.n / nr))
    return pd.DataFrame(rows, columns=["pred_id", "gt_id", "n_rows", "n_matched", "n_majority",
                                       "purity", "match_frac"])


def relabel_with_gt(pred_df, scene, iou=0.5, id_col="track_id"):
    """Convenience: pred_df with a ``gt_id`` column from ``assign_tracks_to_gt`` (majority vehicle)."""
    a = assign_tracks_to_gt(pred_df, scene, iou=iou, id_col=id_col)
    return pred_df.merge(a[["pred_id", "gt_id", "purity", "match_frac"]], left_on=id_col,
                         right_on="pred_id", how="left").drop(columns="pred_id")
