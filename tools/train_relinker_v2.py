"""Train the relinker_v2 model bundles from the repo's two labeled clips.

  relinker_v2_both.joblib     trained on parking + airport (the default model for new footage)
  relinker_v2_parking.joblib  parking only (apply to airport for honest cross-scene numbers)
  relinker_v2_airport.joblib  airport only (apply to parking for honest cross-scene numbers)

Training data: every candidate pair of both clips (relinker_v2.build_pairs with DEFAULT_CONFIG),
labeled with successor-only labels ('succ_drop': A -> B is positive only when B is the next tracklet of
the same vehicle; other same-vehicle pairs are left out; see repo_data.labels_clean). A small
random-forest grid is scored by vehicle-grouped 5-fold CV (average precision); the poster's RF
(depth 5, leaf 2) is kept unless a grid point beats it by more than 0.01 AP. The config (gate, features,
threshold, assembly, merge) travels inside the bundle, so relink() needs nothing else.

The frozen models in tools/models/ were made by this procedure. A run never overwrites them unless you
ask for it: pass an output directory, or --overwrite to write into tools/models/.

    python tools/train_relinker_v2.py --out /some/dir            # train, then compare with tools/models
    python tools/train_relinker_v2.py --overwrite                # replace tools/models/*.joblib
    python tools/train_relinker_v2.py --out d --config '{"threshold": 0.4}'   # DEFAULT_CONFIG overrides

After training into another directory it prints the largest predict_proba difference against the
tools/models bundle of the same name on every candidate pair of both clips (0.0 = identical models).
"""
import argparse
import datetime
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import relinker_v2 as rl
import repo_data as rd

MODEL_NAMES = ["both", "parking", "airport"]


def build_training_table(cfg, label_kind):
    parts = []
    for sc in ["parking", "airport"]:
        _, _, P = rd.pairs_for(sc, cfg)
        P["y"] = rd.labels_clean(sc, P, label_kind)
        P["cluster"] = rd.clusters(sc, P)
        P["fold"] = rd.fold_of(P["cluster"].to_numpy())
        parts.append(P)
    D = pd.concat(parts, ignore_index=True)
    rows = D["y"].notna().to_numpy()
    if cfg["same_id_rule"]:                 # rule mode: same-id continuations never reach the model
        rows &= D["same_id"].to_numpy() == 0
    return D[rows].reset_index(drop=True)


def grid_search(Dt, X, y):
    """Vehicle-grouped 5-fold CV average precision for a small grid around the poster's RF."""
    grid = [dict(max_depth=d, min_samples_leaf=l) for d in (4, 5, 8) for l in (2, 5)]
    fold = Dt["fold"].to_numpy()
    park = (Dt["scene"] == "parking").to_numpy()
    res = []
    for g in grid:
        oof = np.zeros(len(Dt))
        for k in range(5):
            tr, te = fold != k, fold == k
            oof[te] = rd.make_rf(**g).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        res.append({**g, "AP": round(rd.ap(y, oof), 4), "AP_parking": round(rd.ap(y[park], oof[park]), 4),
                    "AP_airport": round(rd.ap(y[~park], oof[~park]), 4)})
        print("  ", res[-1], flush=True)
    best = max(res, key=lambda r: r["AP"])
    base = next(r for r in res if r["max_depth"] == 5 and r["min_samples_leaf"] == 2)
    chosen = best if best["AP"] > base["AP"] + 0.01 else base
    return res, {"max_depth": chosen["max_depth"], "min_samples_leaf": chosen["min_samples_leaf"]}


def compare_with(out_dir, ref_dir):
    """Max |predict_proba difference| between bundles in out_dir and ref_dir, over every candidate pair
    of both clips (built with each reference bundle's own config)."""
    import joblib
    print(f"\ncomparison with {ref_dir}:")
    all_same = True
    for name in MODEL_NAMES:
        new_p = os.path.join(out_dir, f"relinker_v2_{name}.joblib")
        ref_p = os.path.join(ref_dir, f"relinker_v2_{name}.joblib")
        if not os.path.exists(ref_p):
            print(f"  {name:8s} no reference bundle")
            continue
        new, ref = joblib.load(new_p), joblib.load(ref_p)
        diffs, n = [], 0
        for sc in ["parking", "airport"]:
            _, _, P = rd.pairs_for(sc, {**rl.DEFAULT_CONFIG, **ref["config"]})
            X = rd.feature_matrix(P, ref["feature_cols"])
            diffs.append(np.abs(new["model"].predict_proba(X)[:, 1] - ref["model"].predict_proba(X)[:, 1]).max())
            n += len(P)
        same_cfg = new["config"] == ref["config"] and list(new["feature_cols"]) == list(ref["feature_cols"])
        all_same &= same_cfg and max(diffs) == 0.0
        print(f"  {name:8s} pairs {n}  max |dp| {max(diffs):.3g}  same config+features {same_cfg}  "
              f"n_train {new['n_train']} vs {ref['n_train']}")
    print("  -> identical to the reference models" if all_same else "  -> DIFFERENT from the reference models")
    return all_same


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the relinker_v2 model bundles on the repo clips")
    ap.add_argument("--out", help="output directory for the three .joblib bundles (required unless --overwrite)")
    ap.add_argument("--overwrite", action="store_true",
                    help="write into tools/models/, replacing the frozen v2.0 models")
    ap.add_argument("--config", default="{}", help="JSON overrides of relinker_v2.DEFAULT_CONFIG "
                    "(plus 'train_labels': any | succ_neg | succ_drop, default succ_drop)")
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison with tools/models")
    a = ap.parse_args(argv)
    if a.overwrite == bool(a.out):
        ap.error("give exactly one of --out DIR or --overwrite")
    out_dir = rd.MODELS_DIR if a.overwrite else os.path.abspath(a.out)
    if not a.overwrite and os.path.realpath(out_dir) == os.path.realpath(rd.MODELS_DIR):
        ap.error("--out points at tools/models; use --overwrite if you really mean to replace the frozen models")

    import joblib
    import sklearn
    warnings.filterwarnings("ignore")
    t0 = time.time()
    cfg = {**rl.DEFAULT_CONFIG, **json.loads(a.config)}
    label_kind = cfg.pop("train_labels", "succ_drop")
    feats = rl.FEATURES[cfg["feature_set"]]

    Dt = build_training_table(cfg, label_kind)
    X = rd.feature_matrix(Dt, feats)
    y = Dt["y"].astype(int).to_numpy()
    print(f"training rows {len(Dt)} (pos {y.sum()}) | parking {int((Dt.scene == 'parking').sum())} "
          f"airport {int((Dt.scene == 'airport').sum())}", flush=True)
    grid, params = grid_search(Dt, X, y)
    print("chosen", params, flush=True)

    os.makedirs(out_dir, exist_ok=True)
    info = {}
    for name in MODEL_NAMES:
        m = np.ones(len(Dt), bool) if name == "both" else (Dt["scene"] == name).to_numpy()
        mdl = rd.make_rf(**params).fit(X[m], y[m])
        mdl.set_params(n_jobs=1)            # relink() scores a few thousand pairs; threads do not pay off
        bundle = {"model": mdl, "feature_cols": feats, "config": cfg, "train_labels": label_kind,
                  "trained_on": ["parking", "airport"] if name == "both" else [name],
                  "n_train": int(m.sum()), "n_pos": int(y[m].sum()), "rf_params": {**rd.RF_BEST, **params},
                  "sklearn_version": sklearn.__version__, "relinker_version": rl.__version__,
                  "created": datetime.datetime.now().isoformat(timespec="seconds")}
        path = os.path.join(out_dir, f"relinker_v2_{name}.joblib")
        joblib.dump(bundle, path, compress=3)
        info[name] = {"file": os.path.basename(path), "n_train": int(m.sum()), "n_pos": int(y[m].sum())}
        print(f"wrote {path}  (n_train {m.sum()}, pos {y[m].sum()})", flush=True)
    with open(os.path.join(out_dir, "train_relinker_v2.json"), "w") as fh:
        json.dump({"config": cfg, "train_labels": label_kind, "grid": grid, "chosen": params, "models": info},
                  fh, indent=1)
    print(f"[{time.time() - t0:.0f}s] done")

    if not a.overwrite and not a.no_compare:
        compare_with(out_dir, rd.MODELS_DIR)


if __name__ == "__main__":
    sys.exit(main())
