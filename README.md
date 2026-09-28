# CV Vehicle Maneuver Detection

YOLO11 + DeepSORT track vehicles in traffic video, and a relinker repairs fragmented track IDs so
per-vehicle behaviour (arrivals, departures, stops, dwell time) can be measured from the tracks.

Poster: [`Research_Poster.pdf`](Research_Poster.pdf), *A Lightweight Spatio-Temporal Framework for
Repairing Fragmented Vehicle Tracks* (Brian Guo, Dr. Xinyu Liu, UMich MRADS 2026). The code and data as
they were for the poster are tagged **`poster-mrads-2026`** (`git switch -c poster-version poster-mrads-2026`).

## What is here

| Path | What it does |
|---|---|
| `async_yolo_test.py` | Video → YOLO → DeepSORT (or ByteTrack) → tracks CSV + annotated video |
| `realtime_yolo_test.py` | Live YOLO detection viewer (no tracking) |
| `relinker_script.py` | Poster-era relinker: random forest on pair features, hand labels for the two clips |
| `launch_reviewer.py` | Browser tool to accept/reject the relinker's suggestions (`relink_report.csv`) |
| `launch_annotator.py`, `annotator_distribution/` | Browser tool for labelling which IDs are the same vehicle |
| `tools/relinker_v2.py` | Relinker v2: works on any tracker's output, normalised for frame size and fps |
| `tools/score_tracks.py` | One command: relink a tracks file and compare behaviour before/after |
| `tools/mot_eval.py`, `tools/behavior.py`, `tools/zones.py` | ID metrics vs the repo labels, behaviour events/dwell, zone occupancy |
| `tools/train_relinker_v2.py`, `tools/eval_repo_clips.py`, `tools/test_relinker_v2.py` | Retrain, re-score and test relinker v2 from a plain checkout |
| `better_tests/` | Track CSVs: `tracksid3duplicate.csv` = parking lot (00009_Trim), `airport_tracks.csv` = airport drop-off |

Videos are not in git (`assets/` and `*.mp4` are ignored); put them in `assets/`.

## Setup

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn==1.8.0` is pinned because the model bundles in `tools/models/` were saved with it.

## 1. Track a video (`async_yolo_test.py`)

```bash
python async_yolo_test.py                                  # DeepSORT (poster parameters) on assets/Airport_DropOff_Footage_STOCK.mp4
python async_yolo_test.py --video assets/00009_Trim.mp4 --csv parking_tracks.csv --out-video parking_out.mp4
python async_yolo_test.py --tracker bytetrack              # ultralytics ByteTrack, default bytetrack.yaml
python tools/export_embedder_onnx.py                       # once, for --embedder onnx (pip install onnx onnxruntime)
python async_yolo_test.py --embedder onnx                  # same DeepSORT tracks, ~4x faster tracker step on CPU
```

| Flag | Default | |
|---|---|---|
| `--video` | airport stock clip | input video |
| `--model` | `yolo11n.pt` | YOLO weights |
| `--csv` | `tracks.csv` | output tracks CSV |
| `--out-video` | `output.mp4` | annotated video; `none` skips it |
| `--conf` | **0.1** | detector confidence threshold (the poster runs used 0.25) |
| `--tracker` | `deepsort` | `deepsort` or `bytetrack` |
| `--embedder` | `torch` | DeepSORT appearance model runtime: `torch` or `onnx` |

CSV columns: `frame,track_id,x1,y1,x2,y2,velocity_x,velocity_y,conf,class`. One row per track that was
matched to a detection in that frame; the box is the tracker's Kalman box, velocity is the Kalman centre
velocity in px/frame, and `conf`/`class` belong to the detection matched to that track. CSVs written by
earlier versions of the script repeat one detection's `conf`/`class` on every row of a frame, so don't
rely on those two columns in older files.

Notes:
- `--embedder onnx` sets `OPENBLAS_NUM_THREADS=1` itself, before numpy is imported. In torch mode you can
  get part of the same speed-up by running with `OPENBLAS_NUM_THREADS=1`.
- ByteTrack's default config only starts a new track from a detection with conf ≥ 0.6, and ultralytics
  fixes its frame rate at 30 (track buffer = 30 frames).

## 2. Relink and score a tracks file

```bash
# any tracker CSV / headerless CSV / MOTChallenge txt; frame numbers = original video frames
python tools/score_tracks.py my_tracks.csv --fps 30000/1001 --width 1920 --height 1080
python tools/score_tracks.py my_tracks.txt --video my_clip.mp4 --out-dir score_out/my_clip

# the repo's two labelled clips: also IDF1 + event F1 vs the repo labels, raw vs relinked
python tools/score_tracks.py better_tests/tracksid3duplicate.csv --scene parking
python tools/score_tracks.py better_tests/airport_tracks.csv --scene airport \
    --model tools/models/relinker_v2_parking.joblib        # model not trained on this clip
```

Outputs (default `score_out/<input name>/`): `relinked.csv` (input rows + `tracklet_id` + `vehicle_id`),
`vehicles.csv` (one row per relinked vehicle: first/last seen, stops, dwell, raw IDs it merged) and
`summary.json`. `--model`, `--threshold` and `--merge` are passed to the relinker.

The relinker on its own:

```bash
python tools/relinker_v2.py tracks.csv --fps 29.97 --width 1920 --height 1080 --out-prefix out/run1
python tools/relinker_v2.py tracks.txt --video clip.mp4 --out-prefix out/run1
```

Always pass `--out-prefix`; without it the outputs are written next to the input file.

### What the relinker has been shown to do

- **Repo clips** (`python tools/eval_repo_clips.py`, vehicle-level IDF1 over the labelled tracklets):
  parking raw DeepSORT IDs 0.885 → 0.960, airport 0.736 → 0.795, each with the model trained on the
  *other* clip. Only 10 + 21 vehicles are labelled, so the CIs are wide (the airport gain's CI crosses 0),
  and the config was chosen on these same clips. The default `relinker_v2_both` model is in-sample on both.
- **Held-out test** (8 unseen UA-DETRAC cameras, pre-registered, frozen models): on deliberately
  fragmented ByteTrack output it helped every sequence (+0.060 IDF1, 8/8); on clean ByteTrack and
  BoT-SORT output it was neutral; on DeepSORT output with this repo's settings it **hurt** (−0.024, 6/8),
  because the default `sameid` merge joins pieces that share an ID and DeepSORT sometimes hands one ID
  from one car to another. Turning the merge off gave +0.024 in a post-hoc check. Until a checked
  same-ID join lands (v2.1), consider `--merge none` for DeepSORT output.
- Behaviour numbers are only as good as the identities: on the parking clip raw DeepSORT IDs report 21
  arrivals for 1 real arrival. The labels are partial, so a wrong merge into an unlabelled car does not
  lower IDF1. On new footage there is no ground truth; the summary shows what changed, not whether it is right.

### Reproduce / retrain

```bash
python tools/test_relinker_v2.py                  # smoke + regression tests, ~20 s, no pytest needed
python tools/eval_repo_clips.py                   # the repo-clip numbers above
python tools/train_relinker_v2.py --out some/dir  # retrain; matches tools/models exactly
python tools/train_relinker_v2.py --overwrite     # replace tools/models/*.joblib
```

Ground truth is the hand labels in `relinker_script.py`, with one fix: parking tracklets 282_2/282_3
are a parked car and are removed from the White SUV group.

## 3. Review and label (poster-era tools)

```bash
python launch_reviewer.py relink_report.csv --video path/to/video.mp4
python launch_annotator.py better_tests/tracksid3duplicate.csv --video path/to/video.mp4
```

See `annotator_distribution/README.md` for the stand-alone annotator package.
