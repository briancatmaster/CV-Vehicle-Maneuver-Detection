from ultralytics import YOLO
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import csv
from deep_sort_realtime.deepsort_tracker import DeepSort

VEHICLE_CLASSES = [2, 3, 5, 6, 7] #car, motorcycle, bus, train, truck
VIDEO_PATH = "assets/00009_Trim.mp4"
MODEL_PATH = "yolo11n.pt"
csv_file = open("tracks.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "class"])

model = YOLO(MODEL_PATH)

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

frame_id = 0
results = model.predict(source=VIDEO_PATH, stream=True)

cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
cap.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps_in, (width, height))

for r in results:
    start = time.time()
    frame = r.orig_img.copy()
    frame_id += 1
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        csv_writer.writerow([
            frame_id,
            track_id,
            x1, y1, x2, y2,
            conf,
            obj_class
        ])
    end = time.time()
    fps_runtime = 1 / (end - start)
    #print(f"FPS: {fps_runtime:.2f}")
    cv2.putText(frame, f"{fps_runtime:.2f} FPS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    out.write(frame)


csv_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()