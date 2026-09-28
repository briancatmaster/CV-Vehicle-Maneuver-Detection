"""Run YOLO + a multi-object tracker over a video and write one CSV row per tracked vehicle per frame.

CSV columns: frame,track_id,x1,y1,x2,y2,velocity_x,velocity_y,conf,class
  frame      1-based frame number
  x1..y2     the tracker's (Kalman-filtered) box, in pixels
  velocity_* Kalman state centre velocity in px/frame
  conf/class the detection that was matched to THIS track in THIS frame
Rows are only written for tracks that were matched to a detection in that frame.

Examples (from the repo root):
  python async_yolo_test.py                                   # DeepSORT with the poster parameters, airport video
  python async_yolo_test.py --conf 0.25                       # ... and the poster-era detector threshold
  python async_yolo_test.py --video assets/00009_Trim.mp4 --tracker bytetrack --csv parking_tracks.csv
  python async_yolo_test.py --embedder onnx                   # same DeepSORT tracks, much faster on CPU
                                                              # (run tools/export_embedder_onnx.py once first)
"""
import os
import sys

# --embedder onnx: numpy's OpenBLAS must be single-threaded or DeepSORT's small Kalman/cosine-distance
# matrix ops oversubscribe the CPU (measured 3.5x slower per frame on an 8-thread laptop).
# This only works if it is set BEFORE numpy is imported, so it is done here, first thing.
# (You can also get the same speed-up in torch mode by running with OPENBLAS_NUM_THREADS=1 yourself.)
if "--embedder=onnx" in sys.argv or any(a == "--embedder" and b == "onnx" for a, b in zip(sys.argv, sys.argv[1:])):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from ultralytics import YOLO
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import csv
from deep_sort_realtime.deepsort_tracker import DeepSort

HERE = os.path.dirname(os.path.abspath(__file__))  # repo root

VEHICLE_CLASSES = [2, 3, 5, 6, 7] #car, motorcycle, bus, train, truck
VIDEO_PATH = os.path.join(HERE, "assets", "Airport_DropOff_Footage_STOCK.mp4")
MODEL_PATH = "yolo11n.pt"
ONNX_EMBEDDER_PATH = os.path.join(HERE, "tools", "models", "deepsort_mobilenetv2_embedder.onnx")

parser = argparse.ArgumentParser(description="YOLO + DeepSORT/ByteTrack vehicle tracker -> tracks CSV + annotated video")
parser.add_argument("--video", default=VIDEO_PATH, help="input video (default: the airport stock clip)")
parser.add_argument("--model", default=MODEL_PATH, help="YOLO weights (default: yolo11n.pt)")
parser.add_argument("--csv", default="tracks.csv", help="output tracks CSV (default: tracks.csv)")
parser.add_argument("--out-video", default="output.mp4",
                    help="annotated output video (default: output.mp4); pass '' or none to skip writing it")
parser.add_argument("--conf", type=float, default=0.1,
                    help="detector confidence threshold (default 0.1: missed detections limit everything downstream; "
                         "the poster-era runs used ultralytics' default 0.25)")
parser.add_argument("--tracker", choices=["deepsort", "bytetrack"], default="deepsort",
                    help="deepsort = the poster setup (default); bytetrack = ultralytics' built-in ByteTrack, default bytetrack.yaml")
parser.add_argument("--embedder", choices=["torch", "onnx"], default="torch",
                    help="DeepSORT appearance model runtime: torch (as shipped) or onnx (same MobileNetV2 weights "
                         "through ONNX Runtime; needs onnxruntime and tools/export_embedder_onnx.py run once)")
args = parser.parse_args()

# Prefer the repo's own copy of the weights when we are run from somewhere else
if not os.path.exists(args.model) and os.path.exists(os.path.join(HERE, args.model)):
    args.model = os.path.join(HERE, args.model)

model = YOLO(args.model)

if args.tracker == "deepsort":
    tracker = DeepSort(
        max_age=125,
        n_init=6,
        nms_max_overlap=0.6,
        max_cosine_distance=0.125,
        nn_budget=150,
        max_iou_distance=0.6,
        #embedder="mobilenet",
        #half=True,
        #bgr=True
    )

    '''
    tracker = DeepSort(
        max_age=20,
        n_init=2,
        nms_max_overlap=0.3,
        max_cosine_distance=0.8,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )
    '''

    '''
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        embedder="mobilenet"
    )
    '''

    if args.embedder == "onnx":
        # Same MobileNetV2 weights and the library's own preprocessing, only the forward pass runs in
        # ONNX Runtime instead of torch. On the test laptop this gave identical tracks at ~8.8 ms/crop
        # instead of ~146 ms/crop.
        try:
            import onnxruntime as ort
        except ImportError:
            sys.exit("--embedder onnx needs onnxruntime: pip install onnxruntime")
        if not os.path.exists(ONNX_EMBEDDER_PATH):
            sys.exit(f"{ONNX_EMBEDDER_PATH} not found. Create it once with: python tools/export_embedder_onnx.py")
        so = ort.SessionOptions()
        so.intra_op_num_threads = min(4, os.cpu_count() or 1)  # 4 was faster than 8 on the test laptop
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # don't busy-wait between calls
        ort_session = ort.InferenceSession(ONNX_EMBEDDER_PATH, so, providers=["CPUExecutionProvider"])
        torch_embedder = tracker.embedder

        def onnx_predict(crops):
            if len(crops) == 0:
                return []
            x = torch.cat([torch_embedder.preprocess(c) for c in crops]).numpy()
            return list(ort_session.run(None, {"x": x})[0])

        tracker.embedder.predict = onnx_predict
else:
    # ByteTrack exactly as model.track(tracker="bytetrack.yaml") would build it, but called directly so we
    # can read each track's matched detection (conf/class) and its Kalman velocity.
    # ultralytics' matching step needs the 'lap' package (and would try to pip-install it on its own if missing)
    try:
        import lap
    except ImportError:
        sys.exit("--tracker bytetrack needs the 'lap' package: pip install lap")
    from ultralytics.utils import IterableSimpleNamespace, yaml_load
    from ultralytics.utils.checks import check_yaml
    from ultralytics.trackers.byte_tracker import BYTETracker
    tracker_cfg = IterableSimpleNamespace(**yaml_load(check_yaml("bytetrack.yaml")))
    tracker = BYTETracker(args=tracker_cfg, frame_rate=30)  # ultralytics itself always passes frame_rate=30
    # Note: with the default yaml a NEW track needs a detection with conf >= 0.6 (new_track_thresh); detections
    # between 0.1 and 0.5 are only used to keep existing tracks alive. This "clean" default fragments very little
    # (it beat the repo DeepSORT on unseen UA-DETRAC cameras) but can miss vehicles that are never detected confidently.

csv_file = open(args.csv, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "velocity_x", "velocity_y", "conf", "class"])

frame_id = 0
results = model.predict(source=args.video, stream=True, conf=args.conf)

cap = cv2.VideoCapture(args.video)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
cap.release()

out = None
if args.out_video and args.out_video.lower() != "none":
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_video, fourcc, fps_in, (width, height))
start_time = time.time()
tracker_time = 0.0

for r in results:
    frame_start_time = time.time()
    frame = r.orig_img.copy()
    frame_id += 1
    rows = []  # (track_id, x1, y1, x2, y2, velocity_x, velocity_y, conf, class) for this frame

    if args.tracker == "deepsort":
        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            obj_class  = int(box.cls)

            if obj_class in VEHICLE_CLASSES:
                ltwh = [x1, y1, x2 - x1, y2 - y1]
                detections.append((ltwh, conf, obj_class))
        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            track_id = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())

            #Kalman Filter Activities
            kalman_state = t.mean
            velocity_x = kalman_state[4]
            velocity_y = kalman_state[5]

            # conf/class of the detection matched to this track this frame
            # (time_since_update == 0 means there was one, but guard against None anyway)
            det_conf = t.get_det_conf()
            det_class = t.get_det_class()
            conf = float(det_conf) if det_conf is not None else ""
            obj_class = int(det_class) if det_class is not None else ""
            rows.append((track_id, x1, y1, x2, y2, velocity_x, velocity_y, conf, obj_class))
    else:
        boxes = r.boxes.cpu().numpy()
        keep = [int(c) in VEHICLE_CLASSES for c in boxes.cls]
        tracker.update(boxes[keep], frame)
        # After update(), tracked_stracks holds exactly the tracks matched to a detection this frame
        # (unmatched ones move to lost_stracks), so this is the same rule as DeepSORT's time_since_update == 0.
        for t in tracker.tracked_stracks:
            if not t.is_activated:
                continue
            x1, y1, x2, y2 = map(int, t.xyxy)
            # ByteTrack's Kalman state is also (cx, cy, aspect, h, vx, vy, va, vh), so [4], [5] are px/frame
            velocity_x = t.mean[4]
            velocity_y = t.mean[5]
            rows.append((int(t.track_id), x1, y1, x2, y2, velocity_x, velocity_y, float(t.score), int(t.cls)))
    tracker_time += time.time() - frame_start_time

    for track_id, x1, y1, x2, y2, velocity_x, velocity_y, conf, obj_class in rows:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        csv_writer.writerow([
            frame_id,
            track_id,
            x1, y1, x2, y2,
            velocity_x,
            velocity_y,
            conf,
            obj_class
        ])
    end = time.time()
    fps_runtime = 1 / max(end - frame_start_time, 1e-6)  # tracking + drawing only (detection happens in the generator)
    #print(f"FPS: {fps_runtime:.2f}")
    if out is not None:
        cv2.putText(frame, f"{fps_runtime:.2f} FPS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        out.write(frame)


csv_file.close()
if out is not None:
    out.release()
cv2.destroyAllWindows()

#Calculating total run time & average FPS (detection + tracking + writing)
total_time = time.time() - start_time
average_fps = frame_id / total_time
print("Total Run Time is " + str(total_time) + " seconds")
print("Average FPS is " + str(average_fps) + " frames-per-second")
print(f"Tracker ({args.tracker}" + (f", {args.embedder} embedder" if args.tracker == "deepsort" else "") +
      f") averaged {1000 * tracker_time / max(frame_id, 1):.1f} ms/frame")
