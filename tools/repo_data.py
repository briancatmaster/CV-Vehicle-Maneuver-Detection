"""Repo ground truth and helpers shared by train_relinker_v2.py, eval_repo_clips.py and test_relinker_v2.py.

What lives here:
  SCENES                 the two repo clips (better_tests/tracksid3duplicate.csv = parking,
                         better_tests/airport_tracks.csv = airport) with their true fps and frame size
  gt_tracklet_to_vehicle ground truth tracklet -> vehicle, built from the hand labels in relinker_script.py
                         (with the 282_2 / 282_3 label fix, see below)
  labels_clean           pair labels used to train relinker_v2 ('succ_drop' = successor-only)
  fold_of / clusters     fixed vehicle-grouped 5-fold split (no vehicle is in two folds)
  pairs_for              candidate pairs + features for a scene (relinker_v2.build_pairs)
  vehicle_metrics        vehicle-level metrics incl. detection-weighted IDF1 over labeled tracklets

Nothing here writes files. relinker_script.py is only read, never modified.
"""
import contextlib
import io
import os
import zlib

import numpy as np
import pandas as pd

import relinker_v2 as rl

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
MODELS_DIR = os.path.join(TOOLS_DIR, "models")

# relinker_script.py is a research script, not an importable module: everything after
# '#MODEL TRAINING SECTION' trains models and draws plots. We exec only the part above that marker
# (from the repo root, because it opens the CSVs with relative paths) to get the two track tables and
# the hand labels (parking_vehicles, airport_chains, airport_isolated, parking_lot_not_same).
_src = open(os.path.join(REPO_ROOT, "relinker_script.py")).read()
NS = {}
_cwd = os.getcwd()
os.chdir(REPO_ROOT)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_src[:_src.index("#MODEL TRAINING SECTION")], "relinker_script.py (head)", "exec"), NS)
finally:
    os.chdir(_cwd)

# True video properties (ffprobe of the source videos). The poster code approximated fps as
# frame range / video length; relinker_v2 uses the real values.
SCENES = {
    "parking": {"raw": NS["df1"], "fps": 60000 / 1001, "W": 1920, "H": 1080},
    "airport": {"raw": NS["df2"], "fps": 24000 / 1001, "W": 1280, "H": 720},
}

# Same random forest settings as the frozen models (the poster's RF, n_jobs only affects speed).
RF_BEST = dict(n_estimators=200, max_depth=5, min_samples_split=5, min_samples_leaf=2,
               class_weight="balanced", random_state=42)


def make_rf(**kw):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_jobs=2, **{**RF_BEST, **kw})


# ------------------------------------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------------------------------------
def split(scene):
    """The scene's rows split into tracklets exactly as relinker_script.py does (gap > 6 frames)."""
    return NS["split_into_tracklets"](SCENES[scene]["raw"].copy())


def airport_tracklet_chains():
    with contextlib.redirect_stdout(io.StringIO()):
        return NS["map_chains_to_tracklets"](NS["airport_chains"], split("airport"))


# Label fix (2026-09-27): parking tracklets 282_2 (frames 7286-7311) and 282_3 (7351-7390) are a static
# car parked behind the minivan (checked on the video frames), not the White SUV they are grouped with in
# relinker_script.py. With labelfix='drop' (default, used for the shipped models) they are removed from
# the ground truth (unknown); 'legacy' keeps the repo's original labels.
STATIC_282 = ("282_2", "282_3")


def gt_tracklet_to_vehicle(scene, labelfix="drop"):
    """Ground truth: tracklet id -> vehicle label (str). Tracklets that are not labeled are absent.
    parking: every tracklet of each id in a parking_vehicles group.
    airport: tracklets of the labeled chains (map_chains_to_tracklets) + every tracklet of each
    isolated id (one vehicle per isolated id)."""
    ds = split(scene)
    t2v = {}
    if scene == "parking":
        for v, ids in enumerate(NS["parking_vehicles"]):
            for t in ds[ds["track_id"].isin(ids)]["tracklet_id"].unique():
                t2v[t] = f"P{v}"
        if labelfix == "drop":
            for t in STATIC_282:
                t2v.pop(t, None)
        elif labelfix != "legacy":
            raise ValueError(f"labelfix must be 'drop' or 'legacy', got {labelfix!r}")
    else:
        for v, c in enumerate(airport_tracklet_chains()):
            for t in c:
                t2v[t] = f"A{v}"
        for i in sorted(NS["airport_isolated"]):
            for t in ds[ds["track_id"] == i]["tracklet_id"].unique():
                t2v[t] = f"Ai{i}"
    return t2v


def spans(scene):
    g = split(scene).groupby("tracklet_id")["frame"]
    return g.min().to_dict(), g.max().to_dict()


def successor_set(scene, t2v):
    """(A, B) pairs where B is A's immediate successor in its vehicle's time-ordered tracklets:
    the same-vehicle tracklet with the smallest first frame among those starting after A ends."""
    ff, lf = spans(scene)
    by_v = {}
    for t, v in t2v.items():
        by_v.setdefault(v, []).append(t)
    S = set()
    for ts in by_v.values():
        for a in ts:
            later = [b for b in ts if ff[b] > lf[a]]
            if later:
                S.add((a, min(later, key=lambda x: (ff[x], lf[x], x))))
    return S


# ------------------------------------------------------------------------------------------
# Pair labels, clusters and folds
# ------------------------------------------------------------------------------------------
def labels_clean(scene, P, kind="succ_drop"):
    """Label every candidate pair A -> B in P.
    kind='any':       1 if same GT vehicle, 0 if both labeled and different, NaN if either unknown.
    kind='succ_neg':  1 only for immediate successors; other same-vehicle pairs -> 0.
    kind='succ_drop': 1 only for immediate successors; other same-vehicle pairs -> NaN (not trained on).
    Parking pairs whose ids are in parking_lot_not_same are known negatives even when unlabeled."""
    t2v = gt_tracklet_to_vehicle(scene)
    v1 = P["id1"].map(t2v)
    v2 = P["id2"].map(t2v)
    y = np.where(v1.isna() | v2.isna(), np.nan, (v1 == v2).astype(float))
    if scene == "parking":
        nsp = {(str(a), str(b)) for a, b in NS["parking_lot_not_same"]}
        m = np.array([(str(a), str(b)) in nsp for a, b in zip(P["base1"], P["base2"])], bool)
        y = np.where(m, 0.0, y)
    if kind != "any":
        S = successor_set(scene, t2v)
        succ = np.array([(a, b) in S for a, b in zip(P["id1"], P["id2"])], bool)
        y = y.copy()
        y[(y == 1) & ~succ] = 0.0 if kind == "succ_neg" else np.nan
    return y


_FOLDS = None


def vehicle_folds(k=5):
    """Fixed '<scene>:<vehicle>' -> fold map: per scene, vehicles sorted by #tracklets (desc) and dealt
    round-robin. Built from the legacy labels so the split stayed the same across label variants."""
    global _FOLDS
    if _FOLDS is None:
        _FOLDS = {}
        for sc in ["parking", "airport"]:
            sizes = pd.Series(gt_tracklet_to_vehicle(sc, labelfix="legacy")).value_counts()
            order = sorted(sizes.index, key=lambda v: (-sizes[v], v))
            for i, v in enumerate(order):
                _FOLDS[f"{sc}:{v}"] = i % k
    return _FOLDS


def clusters(scene, P):
    """Cluster of each pair = GT vehicle of id1 (else id2, else 'u:<id1>'), prefixed by the scene."""
    t2v = gt_tracklet_to_vehicle(scene)
    c = P["id1"].map(t2v).fillna(P["id2"].map(t2v)).fillna("u:" + P["id1"].astype(str))
    return (scene + ":" + c.astype(str)).to_numpy()


def fold_of(cluster_arr, k=5):
    F = vehicle_folds(k)
    return np.array([F.get(c, zlib.crc32(c.encode()) % k) for c in cluster_arr])


# ------------------------------------------------------------------------------------------
# Pair tables
# ------------------------------------------------------------------------------------------
def pairs_for(scene, cfg):
    """Candidate pairs + features for a scene with relinker_v2.build_pairs (cfg overrides DEFAULT_CONFIG).
    Rows are sorted by (id1, id2), the order the models were trained in."""
    S = SCENES[scene]
    df = rl.normalize_tracks(S["raw"])
    dsplit, T, P = rl.build_pairs(df, S["fps"], S["W"], S["H"], cfg)
    P["scene"] = scene
    P = P.sort_values(["id1", "id2"]).reset_index(drop=True)
    return dsplit, T, P


def feature_matrix(P, feats):
    return np.nan_to_num(P[feats].to_numpy(float), nan=0.0, posinf=1e6, neginf=-1e6)


# ------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------
def ap(y, s):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, int)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


_NDETS = {}


def ndets(scene):
    """tracklet -> number of detections (rows); the weights for detection-level IDF1."""
    if scene not in _NDETS:
        _NDETS[scene] = split(scene).groupby("tracklet_id").size().to_dict()
    return _NDETS[scene]


def vehicle_metrics(t2pred, t2gt, w=None, boot=0, seed=0):
    """t2pred: tracklet -> predicted vehicle (every tracklet); t2gt: labeled tracklet -> GT vehicle.
    Computed over labeled tracklets only.
      n_gt / n_pred = #GT vehicles / #predicted vehicles among labeled tracklets; count_err = n_pred - n_gt
      frag      = mean over GT vehicles of #predicted vehicles its tracklets fall in (1 = perfect)
      impure    = #predicted vehicles holding labeled tracklets of >1 GT vehicle
      recovered = fraction of GT vehicles whose tracklets all share one predicted vehicle holding no
                  other GT vehicle
      idf1      = (needs w: tracklet -> #detections) detection-level identity F1: optimal one-to-one
                  GT-vehicle <-> predicted-vehicle matching, IDTP / #detections
    boot > 0 adds 95% bootstrap CIs resampling GT vehicles (idf1_ci, frag_ci, recovered_ci, count_err_ci)."""
    lab = list(t2gt)
    gt = pd.Series({t: t2gt[t] for t in lab})
    pr = pd.Series({t: t2pred[t] for t in lab})
    G, Pm = {}, {}
    for t in lab:
        G.setdefault(gt[t], []).append(t)
        Pm.setdefault(pr[t], set()).add(gt[t])
    per = {}
    for v, ts in G.items():
        ps = {pr[t] for t in ts}
        contaminated = any(len(Pm[p]) > 1 for p in ps)
        cnt = sum(1.0 / len(Pm[p]) for p in ps)       # share of predicted vehicles attributable to v
        per[v] = (len(ps), int(len(ps) == 1 and not contaminated), int(contaminated), cnt)
    idf1 = None
    if w is not None:
        from scipy.optimize import linear_sum_assignment
        gv = pd.factorize(gt.values)[0]
        pv = pd.factorize(pr.astype(str).values)[0]
        C = np.zeros((gv.max() + 1, pv.max() + 1))
        np.add.at(C, (gv, pv), np.array([w[t] for t in lab], float))
        r, c = linear_sum_assignment(-C)
        idtp_v = np.zeros(C.shape[0])
        idtp_v[r] = C[r, c]
        n_v = C.sum(1)
        idf1 = float(idtp_v.sum() / n_v.sum())
        for i, v in enumerate(pd.factorize(gt.values)[1]):
            per[v] = per[v] + (idtp_v[i], n_v[i])
    arr = np.array(list(per.values()), float)
    n_gt, n_pred = len(G), len(Pm)
    impure = sum(1 for p in Pm if len(Pm[p]) > 1)
    res = {"n_gt": n_gt, "n_pred": n_pred, "count_err": n_pred - n_gt, "frag": round(arr[:, 0].mean(), 3),
           "impure": impure, "recovered": round(arr[:, 1].mean(), 3)}
    if idf1 is not None:
        res["idf1"] = round(idf1, 3)
    if boot:
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(boot):
            s = arr[rng.integers(0, len(arr), len(arr))]
            bs.append([s[:, 0].mean(), s[:, 1].mean(), s[:, 3].sum() - len(s),
                       (s[:, 4].sum() / s[:, 5].sum()) if s.shape[1] > 4 else np.nan])
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        if idf1 is not None:
            res["idf1_ci"] = [round(lo[3], 3), round(hi[3], 3)]
        res["frag_ci"] = [round(lo[0], 2), round(hi[0], 2)]
        res["recovered_ci"] = [round(lo[1], 2), round(hi[1], 2)]
        res["count_err_ci"] = [round(lo[2], 1), round(hi[2], 1)]
    return res
