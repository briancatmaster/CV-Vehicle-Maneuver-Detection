"""Vehicle-level evaluation of relinker_v2 on the repo's two labeled clips (parking, airport).

For each clip it scores, against the hand-labeled ground truth (repo_data.gt_tracklet_to_vehicle,
with the 282_2 / 282_3 label fix):
  tracklets (no relink)    every gap>6 tracklet is its own vehicle
  raw DeepSORT ids         the tracker's own track ids
  relinker_v2 cross-scene  model trained on the OTHER clip only (parking model -> airport,
                           airport model -> parking): the honest, out-of-sample number
  relinker_v2 'both'       model trained on both clips, i.e. IN-SAMPLE here (optimistic; shown only
                           because 'both' is the default model for new footage)

Metric: detection-weighted identity F1 (IDF1) over the labeled tracklets (parking 127, airport 67),
using the tracker's own boxes, so it measures ID linking only, not detection. 95% CIs resample GT
vehicles (1000 draws); 'vs raw' is a paired bootstrap of the IDF1 difference to raw DeepSORT ids
(2000 draws, same resampled vehicles for both). Only 10 parking and 21 airport vehicles: CIs are wide.

    python tools/eval_repo_clips.py                    # frozen models in tools/models
    python tools/eval_repo_clips.py --models some/dir  # e.g. models from train_relinker_v2.py --out

Expected with the frozen v2.0 models: parking raw 0.885 -> cross-scene 0.960, airport raw 0.736 ->
cross-scene 0.795; in-sample 'both' 0.975 / 0.945. (The study's airport 'vs raw' CI, [-0.017, 0.135],
came from a longer shared random stream; this script seeds each clip separately and gets [-0.018, 0.131].)
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import relinker_v2 as rl
import repo_data as rd

OTHER = {"parking": "airport", "airport": "parking"}


def per_vehicle_idtp(t2p, t2gt, w):
    """IDTP and #detections per GT vehicle (sorted by name) under the optimal one-to-one matching."""
    from scipy.optimize import linear_sum_assignment
    lab = sorted(t2gt)
    vs = sorted(set(t2gt.values()))
    gi = {v: i for i, v in enumerate(vs)}
    pv = pd.factorize(pd.Series([str(t2p[t]) for t in lab]))[0]
    C = np.zeros((len(vs), pv.max() + 1))
    for t, j in zip(lab, pv):
        C[gi[t2gt[t]], j] += w[t]
    r, c = linear_sum_assignment(-C)
    idtp = np.zeros(len(vs))
    idtp[r] = C[r, c]
    return idtp, C.sum(1)


def paired_diff(a, b, n=2000, seed=0):
    """IDF1(a) - IDF1(b) and its 95% CI, resampling GT vehicles (same draw for both methods)."""
    (ia, na), (ib, nb) = a, b
    d0 = ia.sum() / na.sum() - ib.sum() / nb.sum()
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        k = rng.integers(0, len(ia), len(ia))
        ds.append(ia[k].sum() / na[k].sum() - ib[k].sum() / nb[k].sum())
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return round(d0, 3), [round(lo, 3), round(hi, 3)]


def evaluate(models_dir, boot=1000):
    import joblib
    bundles = {n: joblib.load(os.path.join(models_dir, f"relinker_v2_{n}.joblib"))
               for n in ["both", "parking", "airport"]}
    rows = []
    for sc in ["parking", "airport"]:
        S = rd.SCENES[sc]
        t2gt = rd.gt_tracklet_to_vehicle(sc)
        w = rd.ndets(sc)
        _, _, _, _, T = rl.relink(S["raw"], S["fps"], S["W"], S["H"], bundle=bundles["both"], return_all=True)
        methods = [("tracklets (no relink)", "-", {t: t for t in T["tid"]}),
                   ("raw DeepSORT ids", "-", dict(zip(T["tid"], T["base"])))]
        for name, kind in [(OTHER[sc], "cross-scene"), ("both", "IN-SAMPLE")]:
            _, t2v, _ = rl.relink(S["raw"], S["fps"], S["W"], S["H"], bundle=bundles[name])
            methods.append((f"relinker_v2 '{name}' model", kind, t2v))
        raw_idtp = per_vehicle_idtp(methods[1][2], t2gt, w)
        for name, kind, t2p in methods:
            m = rd.vehicle_metrics(t2p, t2gt, w=w, boot=boot)
            row = {"scene": sc, "method": name, "kind": kind, **m}
            if name.startswith("relinker"):
                row["vs_raw"], row["vs_raw_ci"] = paired_diff(per_vehicle_idtp(t2p, t2gt, w), raw_idtp)
            rows.append(row)
    return rows


def print_table(rows):
    print(f"{'scene':8s} {'method':28s} {'kind':12s} {'IDF1':>6s} {'95% CI':15s} {'vs raw [95% CI]':24s} "
          f"{'pred/true veh':>13s} {'count_err':>9s} {'impure':>6s}")
    for r in rows:
        ci = f"[{r['idf1_ci'][0]:.3f}, {r['idf1_ci'][1]:.3f}]" if "idf1_ci" in r else ""
        vs = (f"{r['vs_raw']:+.3f} [{r['vs_raw_ci'][0]:+.3f}, {r['vs_raw_ci'][1]:+.3f}]" if "vs_raw" in r else "")
        print(f"{r['scene']:8s} {r['method']:28s} {r['kind']:12s} {r['idf1']:6.3f} {ci:15s} {vs:24s} "
              f"{r['n_pred']:>6d}/{r['n_gt']:<6d} {r['count_err']:9d} {r['impure']:6d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vehicle-level IDF1 of relinker_v2 on the repo clips")
    ap.add_argument("--models", default=rd.MODELS_DIR, help="directory with relinker_v2_{both,parking,airport}.joblib")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws for the IDF1 CIs (0 = none)")
    ap.add_argument("--json", help="also write the rows to this JSON file")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    rows = evaluate(a.models, a.boot)
    print(f"models: {os.path.abspath(a.models)}")
    print_table(rows)
    print("\ncross-scene = trained on the other clip only (out of sample). IN-SAMPLE = the 'both' model was "
          "trained on this clip, so its numbers are optimistic.\nThe config and threshold were chosen on these "
          "same two clips, so even the cross-scene numbers carry some selection optimism.")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1, default=float)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    sys.exit(main())
