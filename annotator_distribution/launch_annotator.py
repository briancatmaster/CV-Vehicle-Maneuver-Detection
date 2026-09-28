import sys
import os
import json
import glob
import shutil
import subprocess
import webbrowser
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from functools import partial

"""
Vehicle Annotation Tool

Zero-config usage (recommended for annotators):
  Place one .mp4 and one .csv in a "data/" folder next to this script, then:
  python launch_annotator.py

Power-user usage:
  python launch_annotator.py <tracking_csv> --video <video.mp4> [--video-length <seconds>]
"""


def auto_detect_files(data_dir):
    """Find a single .mp4 and .csv in data_dir. Returns (csv_path, video_path, n_videos, n_csvs)."""
    if not os.path.isdir(data_dir):
        return None, None, 0, 0
    videos = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    video = videos[0] if len(videos) >= 1 else None
    csv = csvs[0] if len(csvs) >= 1 else None
    return csv, video, len(videos), len(csvs)


def get_video_duration(video_path):
    """Try ffprobe to get duration in seconds. Returns float or None."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ], stderr=subprocess.DEVNULL, timeout=10)
        return float(out.decode().strip())
    except Exception:
        return None


class VideoHandler(SimpleHTTPRequestHandler):
    """Serves video with Range request support for seeking."""

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")

        if range_header:
            byte_range = range_header.strip().split("=")[1]
            parts = byte_range.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def log_message(self, format, *args):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            pass


def start_video_server(video_path):
    video_dir = os.path.dirname(os.path.abspath(video_path))
    handler = partial(VideoHandler, directory=video_dir)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    video_filename = os.path.basename(video_path)
    video_url = f"http://127.0.0.1:{port}/{video_filename}"
    return video_url, server


def build_html(csv_text, video_url, video_length):
    escaped_csv = json.dumps(csv_text)
    video_url_js = json.dumps(video_url)
    video_length_js = json.dumps(video_length) if video_length else "null"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vehicle Annotation Tool</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0c0c14; overflow: hidden; }}
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: #0c0c14; }}
  ::-webkit-scrollbar-thumb {{ background: #2a2a3e; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #4a4a6e; }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
</head>
<body>
<div id="root"></div>
<script>
const RAW_CSV = {escaped_csv};
const VIDEO_URL = {video_url_js};
const VIDEO_LENGTH_INITIAL = {video_length_js};

const e = React.createElement;
const useState = React.useState;
const useEffect = React.useEffect;
const useMemo = React.useMemo;
const useCallback = React.useCallback;
const useRef = React.useRef;

/* ── Theme ── */
const font = "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace";
const bg = "#0c0c14";
const surface = "#12121e";
const surfaceHover = "#1a1a2e";
const border = "#2a2a3e";
const textPrimary = "#e2e2f0";
const textSecondary = "#8888a0";

const GROUP_COLORS = [
  "#4ade80","#60a5fa","#f472b6","#facc15","#a78bfa",
  "#fb923c","#2dd4bf","#e879f9","#f87171","#38bdf8",
  "#a3e635","#fbbf24","#c084fc","#22d3ee","#fb7185"
];

/* ── Parse tracking CSV ── */
function parseTrackingCSV(text) {{
  const lines = text.trim().split("\\n");
  const headers = lines[0].split(",").map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {{
    const vals = lines[i].split(",");
    const obj = {{}};
    headers.forEach((h, j) => {{ obj[h] = vals[j] ? vals[j].trim() : ""; }});
    obj.frame = parseInt(obj.frame);
    obj.track_id = parseInt(obj.track_id);
    obj.x1 = parseFloat(obj.x1);
    obj.y1 = parseFloat(obj.y1);
    obj.x2 = parseFloat(obj.x2);
    obj.y2 = parseFloat(obj.y2);
    if (!isNaN(obj.frame)) rows.push(obj);
  }}
  return rows;
}}

/* ── Index by frame ── */
function buildFrameIndex(rows) {{
  const idx = {{}};
  let maxFrame = 0;
  const trackIds = new Set();
  for (const r of rows) {{
    if (!idx[r.frame]) idx[r.frame] = [];
    idx[r.frame].push(r);
    if (r.frame > maxFrame) maxFrame = r.frame;
    trackIds.add(r.track_id);
  }}
  return {{ idx, maxFrame, trackIds: Array.from(trackIds).sort((a,b) => a - b) }};
}}

/* ── Storage key ── */
const storageKey = "annotator_" + btoa(RAW_CSV.slice(0, 80)).replace(/[^a-z0-9]/gi, "").slice(0, 16);

function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(storageKey)); }} catch(e) {{ return null; }}
}}

function saveState(state) {{
  localStorage.setItem(storageKey, JSON.stringify(state));
}}

/* ── App ── */
function App() {{
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const animRef = useRef(null);

  const [trackingData] = useState(() => parseTrackingCSV(RAW_CSV));
  const [frameData] = useState(() => buildFrameIndex(trackingData));
  const maxFrame = frameData.maxFrame;

  /* Video length: from server, or stored, or user-prompted */
  const [videoLength, setVideoLength] = useState(() => {{
    if (VIDEO_LENGTH_INITIAL) return VIDEO_LENGTH_INITIAL;
    const stored = localStorage.getItem(storageKey + "_vidlen");
    return stored ? parseFloat(stored) : null;
  }});
  const [lengthInput, setLengthInput] = useState("");
  const fps = videoLength ? maxFrame / videoLength : 30;

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedTrackId, setSelectedTrackId] = useState(null);
  const [activeGroupIdx, setActiveGroupIdx] = useState(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const showOverlayRef = useRef(true);

  /* Annotation state */
  const savedState = useMemo(() => loadState(), []);
  const [vehicleGroups, setVehicleGroups] = useState(savedState?.vehicleGroups || []);
  const [isolated, setIsolated] = useState(savedState?.isolated || []);
  const [notSamePairs, setNotSamePairs] = useState(savedState?.notSamePairs || []);
  const [notSameDraft, setNotSameDraft] = useState(null);

  /* Persist state */
  useEffect(() => {{
    saveState({{ vehicleGroups, isolated, notSamePairs }});
  }}, [vehicleGroups, isolated, notSamePairs]);

  /* Keep ref in sync */
  useEffect(() => {{ showOverlayRef.current = showOverlay; }}, [showOverlay]);

  /* Track ID to group color mapping */
  const trackIdToGroup = useMemo(() => {{
    const map = {{}};
    vehicleGroups.forEach((g, i) => {{
      g.track_ids.forEach(tid => {{ map[tid] = i; }});
    }});
    return map;
  }}, [vehicleGroups]);

  const isolatedSet = useMemo(() => new Set(isolated), [isolated]);

  /* ── Video <-> Frame sync ── */
  const timeToFrame = useCallback((t) => Math.round(t * fps), [fps]);
  const frameToTime = useCallback((f) => f / fps, [fps]);

  function fmtTime(sec) {{
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }}

  /* ── Canvas drawing loop ── */
  const drawFrame = useCallback(() => {{
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const container = containerRef.current;
    if (!canvas || !video || !container) return;

    const containerRect = container.getBoundingClientRect();
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    canvas.style.left = (rect.left - containerRect.left) + "px";
    canvas.style.top = (rect.top - containerRect.top) + "px";

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frame = timeToFrame(video.currentTime);
    setCurrentFrame(frame);

    if (!showOverlayRef.current) return;

    /* Find closest frame with data */
    let boxes = frameData.idx[frame];
    if (!boxes) {{
      for (let d = 1; d <= 3; d++) {{
        boxes = frameData.idx[frame - d] || frameData.idx[frame + d];
        if (boxes) break;
      }}
    }}
    if (!boxes) return;

    /* Scale factors: video natural size -> displayed size */
    const scaleX = rect.width / video.videoWidth;
    const scaleY = rect.height / video.videoHeight;

    for (const box of boxes) {{
      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const w = (box.x2 - box.x1) * scaleX;
      const h = (box.y2 - box.y1) * scaleY;

      const gIdx = trackIdToGroup[box.track_id];
      const isIsolated = isolatedSet.has(box.track_id);
      const isSelected = box.track_id === selectedTrackId;

      let color;
      if (isSelected) {{
        color = "#ffffff";
      }} else if (gIdx !== undefined) {{
        color = GROUP_COLORS[gIdx % GROUP_COLORS.length];
      }} else if (isIsolated) {{
        color = "#555555";
      }} else {{
        color = "#818cf8";
      }}

      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x, y, w, h);

      /* Label */
      const label = "" + box.track_id;
      ctx.font = "bold 12px " + font;
      const tm = ctx.measureText(label);
      const lw = tm.width + 8;
      const lh = 18;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x, y - lh, lw, lh);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#000";
      ctx.fillText(label, x + 4, y - 4);
    }}
  }}, [frameData, timeToFrame, trackIdToGroup, isolatedSet, selectedTrackId]);

  useEffect(() => {{
    let running = true;
    function loop() {{
      if (!running) return;
      drawFrame();
      animRef.current = requestAnimationFrame(loop);
    }}
    loop();
    return () => {{ running = false; cancelAnimationFrame(animRef.current); }};
  }}, [drawFrame]);

  /* ── Canvas click: select track ── */
  const handleCanvasClick = useCallback((ev) => {{
    if (!showOverlayRef.current) return;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = ev.clientX - rect.left;
    const clickY = ev.clientY - rect.top;

    const scaleX = rect.width / video.videoWidth;
    const scaleY = rect.height / video.videoHeight;

    const frame = timeToFrame(video.currentTime);
    let boxes = frameData.idx[frame];
    if (!boxes) {{
      for (let d = 1; d <= 3; d++) {{
        boxes = frameData.idx[frame - d] || frameData.idx[frame + d];
        if (boxes) break;
      }}
    }}
    if (!boxes) return;

    let best = null;
    let bestArea = Infinity;
    for (const box of boxes) {{
      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const w = (box.x2 - box.x1) * scaleX;
      const h = (box.y2 - box.y1) * scaleY;
      if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {{
        const area = w * h;
        if (area < bestArea) {{ best = box; bestArea = area; }}
      }}
    }}
    if (best) {{
      setSelectedTrackId(prev => prev === best.track_id ? null : best.track_id);
    }}
  }}, [frameData, timeToFrame]);

  /* ── Keyboard shortcuts ── */
  useEffect(() => {{
    const handler = (ev) => {{
      if (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA") return;
      const vid = videoRef.current;
      if (!vid) return;

      switch(ev.key) {{
        case " ":
          ev.preventDefault();
          if (vid.paused) vid.play(); else vid.pause();
          setIsPlaying(!vid.paused);
          break;
        case "ArrowLeft":
          ev.preventDefault();
          vid.currentTime = Math.max(0, vid.currentTime - (ev.shiftKey ? 5 : 1/fps));
          break;
        case "ArrowRight":
          ev.preventDefault();
          vid.currentTime = Math.min(vid.duration, vid.currentTime + (ev.shiftKey ? 5 : 1/fps));
          break;
        case "g":
        case "G":
          setVehicleGroups(prev => [...prev, {{ track_ids: [], label: "Vehicle " + (prev.length + 1) }}]);
          setActiveGroupIdx(vehicleGroups.length);
          break;
        case "i":
        case "I":
          if (selectedTrackId !== null && !isolatedSet.has(selectedTrackId)) {{
            setIsolated(prev => [...prev, selectedTrackId]);
          }}
          break;
        case "o":
        case "O":
          setShowOverlay(prev => !prev);
          break;
        case "Escape":
          setSelectedTrackId(null);
          break;
      }}
    }};
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }}, [fps, selectedTrackId, isolatedSet, vehicleGroups]);

  /* ── Actions ── */
  const addToGroup = useCallback((groupIdx, trackId) => {{
    if (trackId === null) return;
    setVehicleGroups(prev => {{
      const next = prev.map((g, i) => {{
        const filtered = g.track_ids.filter(t => t !== trackId);
        if (i === groupIdx && !g.track_ids.includes(trackId)) {{
          return {{ ...g, track_ids: [...filtered, trackId].sort((a,b) => a-b) }};
        }}
        return {{ ...g, track_ids: filtered }};
      }});
      return next;
    }});
    setIsolated(prev => prev.filter(t => t !== trackId));
  }}, []);

  const removeFromGroup = useCallback((groupIdx, trackId) => {{
    setVehicleGroups(prev => prev.map((g, i) =>
      i === groupIdx ? {{ ...g, track_ids: g.track_ids.filter(t => t !== trackId) }} : g
    ));
  }}, []);

  const deleteGroup = useCallback((groupIdx) => {{
    setVehicleGroups(prev => prev.filter((_, i) => i !== groupIdx));
    if (activeGroupIdx === groupIdx) setActiveGroupIdx(null);
    else if (activeGroupIdx > groupIdx) setActiveGroupIdx(activeGroupIdx - 1);
  }}, [activeGroupIdx]);

  const removeIsolated = useCallback((trackId) => {{
    setIsolated(prev => prev.filter(t => t !== trackId));
  }}, []);

  const addNotSamePair = useCallback(() => {{
    if (notSameDraft && notSameDraft.id1 !== undefined && selectedTrackId !== null && notSameDraft.id1 !== selectedTrackId) {{
      const pair = [Math.min(notSameDraft.id1, selectedTrackId), Math.max(notSameDraft.id1, selectedTrackId)];
      const exists = notSamePairs.some(p => p[0] === pair[0] && p[1] === pair[1]);
      if (!exists) {{
        setNotSamePairs(prev => [...prev, pair]);
      }}
      setNotSameDraft(null);
    }}
  }}, [notSameDraft, selectedTrackId, notSamePairs]);

  const removeNotSamePair = useCallback((idx) => {{
    setNotSamePairs(prev => prev.filter((_, i) => i !== idx));
  }}, []);

  /* ── Export ── */
  const exportJSON = useCallback(() => {{
    const output = {{
      video_length_seconds: videoLength,
      vehicle_groups: vehicleGroups.map((g, i) => ({{ id: i, track_ids: g.track_ids, label: g.label }})),
      isolated: isolated.sort((a,b) => a - b),
      not_same_pairs: notSamePairs
    }};
    const blob = new Blob([JSON.stringify(output, null, 2)], {{ type: "application/json" }});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "annotations.json";
    a.click();
  }}, [vehicleGroups, isolated, notSamePairs, videoLength]);

  /* ── Import ── */
  const importJSON = useCallback(() => {{
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (ev) => {{
      const file = ev.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {{
        try {{
          const data = JSON.parse(e.target.result);
          if (data.vehicle_groups) setVehicleGroups(data.vehicle_groups.map(g => ({{ track_ids: g.track_ids || [], label: g.label || "" }})));
          if (data.isolated) setIsolated(data.isolated);
          if (data.not_same_pairs) setNotSamePairs(data.not_same_pairs);
        }} catch(err) {{ alert("Invalid JSON: " + err.message); }}
      }};
      reader.readAsText(file);
    }};
    input.click();
  }}, []);

  /* ── Visible track IDs at current frame ── */
  const visibleTracks = useMemo(() => {{
    const boxes = frameData.idx[currentFrame];
    if (!boxes) return [];
    return [...new Set(boxes.map(b => b.track_id))].sort((a,b) => a - b);
  }}, [frameData, currentFrame]);

  /* ── Stats ── */
  const stats = useMemo(() => {{
    const grouped = new Set();
    vehicleGroups.forEach(g => g.track_ids.forEach(t => grouped.add(t)));
    return {{
      totalTracks: frameData.trackIds.length,
      grouped: grouped.size,
      isolated: isolated.length,
      unlabeled: frameData.trackIds.length - grouped.size - isolated.length
    }};
  }}, [frameData, vehicleGroups, isolated]);

  /* ── RENDER ── */

  /* Header */
  const header = e("div", {{
    style: {{ padding: "12px 20px", borderBottom: "1px solid " + border, display: "flex",
              alignItems: "center", justifyContent: "space-between", background: surface }}
  }},
    e("div", {{ style: {{ display: "flex", alignItems: "center", gap: 12 }} }},
      e("div", {{ style: {{ width: 8, height: 8, borderRadius: "50%", background: "#818cf8",
                            boxShadow: "0 0 8px #818cf866" }} }}),
      e("span", {{ style: {{ fontSize: 15, fontWeight: 700, letterSpacing: 1, fontFamily: font, color: textPrimary }} }}, "VEHICLE ANNOTATOR")
    ),
    e("div", {{ style: {{ display: "flex", gap: 16, fontSize: 12, fontFamily: font }} }},
      e("span", {{ style: {{ color: "#4ade80" }} }}, stats.grouped + " grouped"),
      e("span", {{ style: {{ color: "#888" }} }}, stats.isolated + " isolated"),
      e("span", {{ style: {{ color: textSecondary }} }}, stats.unlabeled + " unlabeled"),
      e("span", {{ style: {{ color: textSecondary }} }}, "/ " + stats.totalTracks + " total")
    )
  );

  /* Video + Canvas */
  const videoPanel = e("div", {{
    ref: containerRef,
    style: {{ position: "relative", background: "#000", flex: 1, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}
  }},
    e("video", {{
      ref: videoRef,
      src: VIDEO_URL,
      controls: true,
      preload: "auto",
      onPlay: () => setIsPlaying(true),
      onPause: () => setIsPlaying(false),
      style: {{ maxWidth: "100%", maxHeight: "100%" }}
    }}),
    e("canvas", {{
      ref: canvasRef,
      onClick: handleCanvasClick,
      style: {{ position: "absolute", top: 0, left: 0,
                pointerEvents: showOverlay ? "auto" : "none",
                cursor: showOverlay ? "crosshair" : "default" }}
    }})
  );

  /* Frame info bar */
  const frameBar = e("div", {{
    style: {{ padding: "8px 16px", background: surface, borderTop: "1px solid " + border, borderBottom: "1px solid " + border,
              display: "flex", gap: 16, alignItems: "center", fontSize: 12, fontFamily: font, color: textSecondary }}
  }},
    e("span", null, "Frame: ", e("span", {{ style: {{ color: textPrimary, fontWeight: 600 }} }}, currentFrame)),
    e("span", null, "Time: ", e("span", {{ style: {{ color: textPrimary }} }}, fmtTime(frameToTime(currentFrame)))),
    e("span", null, "Visible: ", e("span", {{ style: {{ color: textPrimary }} }}, visibleTracks.length)),
    selectedTrackId !== null ? e("span", null, "Selected: ", e("span", {{ style: {{ color: "#fff", fontWeight: 700 }} }}, "#" + selectedTrackId)) : null,
    e("div", {{ style: {{ flex: 1 }} }}),
    e("button", {{
      onClick: () => setShowOverlay(prev => !prev),
      style: {{ background: showOverlay ? "#818cf8" : "transparent", color: showOverlay ? "#000" : textSecondary,
                border: "1px solid " + (showOverlay ? "#818cf8" : border), borderRadius: 4, padding: "3px 10px",
                fontFamily: font, fontSize: 10, fontWeight: 600, cursor: "pointer" }}
    }}, showOverlay ? "OVERLAY ON (O)" : "OVERLAY OFF (O)")
  );

  /* ── Right sidebar ── */

  /* Visible tracks - bigger chips, more prominent */
  const visibleTracksList = e("div", {{ style: {{ padding: "12px 14px", borderBottom: "1px solid " + border }} }},
    e("div", {{ style: {{ fontSize: 11, color: textSecondary, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10, fontFamily: font, fontWeight: 600 }} }},
      "Tracks in Frame (" + visibleTracks.length + ")"),
    e("div", {{ style: {{ display: "flex", flexWrap: "wrap", gap: 6 }} }},
      ...visibleTracks.map(tid => {{
        const gIdx = trackIdToGroup[tid];
        const isIso = isolatedSet.has(tid);
        const isSel = tid === selectedTrackId;
        let chipColor = gIdx !== undefined ? GROUP_COLORS[gIdx % GROUP_COLORS.length] : isIso ? "#555" : "#818cf8";
        return e("button", {{
          key: tid,
          onClick: () => setSelectedTrackId(prev => prev === tid ? null : tid),
          style: {{
            background: isSel ? chipColor : bg,
            color: isSel ? "#000" : chipColor,
            border: "2px solid " + chipColor,
            borderRadius: 6, padding: "6px 12px", fontFamily: font, fontSize: 13,
            fontWeight: 700, cursor: "pointer",
            boxShadow: isSel ? "0 0 8px " + chipColor + "66" : "none"
          }}
        }}, "" + tid);
      }})
    ),
    visibleTracks.length === 0 ? e("div", {{ style: {{ fontSize: 11, color: textSecondary, fontStyle: "italic" }} }}, "No tracks at this frame") : null
  );

  /* Vehicle groups */
  const groupsPanel = e("div", {{ style: {{ padding: "12px 14px", borderBottom: "1px solid " + border }} }},
    e("div", {{ style: {{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }} }},
      e("div", {{ style: {{ fontSize: 11, color: textSecondary, textTransform: "uppercase", letterSpacing: 1, fontFamily: font, fontWeight: 600 }} }}, "Vehicle Groups"),
      e("button", {{
        onClick: () => {{
          setVehicleGroups(prev => [...prev, {{ track_ids: [], label: "Vehicle " + (prev.length + 1) }}]);
          setActiveGroupIdx(vehicleGroups.length);
        }},
        style: {{ background: "#818cf8", color: "#000", border: "none", borderRadius: 6, padding: "6px 14px",
                  fontFamily: font, fontSize: 11, fontWeight: 700, cursor: "pointer" }}
      }}, "+ NEW GROUP (G)")
    ),
    ...vehicleGroups.map((group, gIdx) => {{
      const color = GROUP_COLORS[gIdx % GROUP_COLORS.length];
      const isActive = activeGroupIdx === gIdx;
      return e("div", {{
        key: gIdx,
        onClick: () => setActiveGroupIdx(isActive ? null : gIdx),
        style: {{
          background: isActive ? surfaceHover : "transparent",
          border: "2px solid " + (isActive ? color : border),
          borderRadius: 8, padding: "10px 12px", marginBottom: 8, cursor: "pointer"
        }}
      }},
        e("div", {{ style: {{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }} }},
          e("div", {{ style: {{ display: "flex", alignItems: "center", gap: 8 }} }},
            e("div", {{ style: {{ width: 12, height: 12, borderRadius: 3, background: color }} }}),
            e("input", {{
              value: group.label,
              onClick: (ev) => ev.stopPropagation(),
              onChange: (ev) => {{
                const val = ev.target.value;
                setVehicleGroups(prev => prev.map((g, i) => i === gIdx ? {{ ...g, label: val }} : g));
              }},
              style: {{ background: "transparent", border: "none", color: textPrimary, fontFamily: font,
                        fontSize: 13, fontWeight: 600, outline: "none", width: 150 }}
            }})
          ),
          e("div", {{ style: {{ display: "flex", gap: 6 }} }},
            isActive && selectedTrackId !== null ? e("button", {{
              onClick: (ev) => {{ ev.stopPropagation(); addToGroup(gIdx, selectedTrackId); }},
              style: {{ background: color, color: "#000", border: "none", borderRadius: 4, padding: "4px 10px",
                        fontFamily: font, fontSize: 11, fontWeight: 700, cursor: "pointer" }}
            }}, "+ #" + selectedTrackId) : null,
            e("button", {{
              onClick: (ev) => {{ ev.stopPropagation(); deleteGroup(gIdx); }},
              style: {{ background: "transparent", color: "#f87171", border: "1px solid #f87171", borderRadius: 4,
                        padding: "3px 8px", fontFamily: font, fontSize: 10, cursor: "pointer" }}
            }}, "DEL")
          )
        ),
        e("div", {{ style: {{ display: "flex", flexWrap: "wrap", gap: 4 }} }},
          ...group.track_ids.map(tid =>
            e("span", {{
              key: tid,
              onClick: (ev) => {{ ev.stopPropagation(); setSelectedTrackId(tid); }},
              style: {{ background: color + "22", color: color, border: "1px solid " + color + "55",
                        borderRadius: 4, padding: "3px 8px", fontSize: 12, fontFamily: font, cursor: "pointer",
                        fontWeight: 600, display: "flex", alignItems: "center", gap: 4 }}
            }},
              tid,
              e("span", {{
                onClick: (ev) => {{ ev.stopPropagation(); removeFromGroup(gIdx, tid); }},
                style: {{ cursor: "pointer", opacity: 0.5, fontSize: 14 }}
              }}, "\\u00d7")
            )
          ),
          group.track_ids.length === 0 ? e("span", {{ style: {{ fontSize: 11, color: textSecondary, fontStyle: "italic" }} }},
            "Select a track above, then click +") : null
        )
      );
    }}),
    vehicleGroups.length === 0 ? e("div", {{ style: {{ padding: "16px 0", textAlign: "center" }} }},
      e("div", {{ style: {{ fontSize: 12, color: textSecondary, marginBottom: 4 }} }}, "No groups yet"),
      e("div", {{ style: {{ fontSize: 11, color: textSecondary, fontStyle: "italic" }} }}, "Press G or click + NEW GROUP to start")
    ) : null
  );

  /* ── Advanced section (collapsed by default) ── */
  const advancedSection = e("div", {{ style: {{ borderBottom: "1px solid " + border }} }},
    e("div", {{
      onClick: () => setShowAdvanced(prev => !prev),
      style: {{ padding: "10px 14px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center" }}
    }},
      e("span", {{ style: {{ fontSize: 10, color: textSecondary, textTransform: "uppercase", letterSpacing: 1, fontFamily: font }} }}, "Advanced"),
      e("span", {{ style: {{ color: textSecondary, fontSize: 10, fontFamily: font }} }}, showAdvanced ? "\\u25B2" : "\\u25BC")
    ),
    showAdvanced ? e("div", {{ style: {{ padding: "0 14px 12px" }} }},
      /* Isolated */
      e("div", {{ style: {{ marginBottom: 12 }} }},
        e("div", {{ style: {{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }} }},
          e("div", {{ style: {{ fontSize: 10, color: textSecondary, fontFamily: font }} }}, "ISOLATED TRACKS"),
          selectedTrackId !== null && !isolatedSet.has(selectedTrackId) && trackIdToGroup[selectedTrackId] === undefined ?
            e("button", {{
              onClick: () => setIsolated(prev => [...prev, selectedTrackId]),
              style: {{ background: "#555", color: "#fff", border: "none", borderRadius: 4, padding: "3px 10px",
                        fontFamily: font, fontSize: 10, fontWeight: 700, cursor: "pointer" }}
            }}, "+ #" + selectedTrackId + " (I)") : null
        ),
        e("div", {{ style: {{ display: "flex", flexWrap: "wrap", gap: 4 }} }},
          ...isolated.map(tid =>
            e("span", {{
              key: tid,
              style: {{ background: "#55555522", color: "#888", border: "1px solid #55555544",
                        borderRadius: 3, padding: "2px 7px", fontSize: 11, fontFamily: font,
                        display: "flex", alignItems: "center", gap: 3 }}
            }},
              tid,
              e("span", {{ onClick: () => removeIsolated(tid), style: {{ cursor: "pointer", opacity: 0.5 }} }}, "\\u00d7")
            )
          ),
          isolated.length === 0 ? e("span", {{ style: {{ fontSize: 10, color: textSecondary, fontStyle: "italic" }} }}, "none") : null
        )
      ),
      /* Not-same pairs */
      e("div", null,
        e("div", {{ style: {{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }} }},
          e("div", {{ style: {{ fontSize: 10, color: textSecondary, fontFamily: font }} }}, "NOT-SAME PAIRS"),
          notSameDraft === null ?
            e("button", {{
              onClick: () => {{ if (selectedTrackId !== null) setNotSameDraft({{ id1: selectedTrackId }}); }},
              disabled: selectedTrackId === null,
              style: {{ background: selectedTrackId !== null ? "#dc2626" : "#333", color: "#fff", border: "none", borderRadius: 4, padding: "3px 10px",
                        fontFamily: font, fontSize: 10, fontWeight: 700, cursor: selectedTrackId !== null ? "pointer" : "default", opacity: selectedTrackId !== null ? 1 : 0.4 }}
            }}, "START PAIR") :
            e("div", {{ style: {{ display: "flex", gap: 4, alignItems: "center" }} }},
              e("span", {{ style: {{ fontSize: 10, color: "#f87171", fontFamily: font }} }}, "#" + notSameDraft.id1 + " \\u2260 ?"),
              selectedTrackId !== null && selectedTrackId !== notSameDraft.id1 ?
                e("button", {{
                  onClick: addNotSamePair,
                  style: {{ background: "#dc2626", color: "#fff", border: "none", borderRadius: 4, padding: "3px 8px",
                            fontFamily: font, fontSize: 10, fontWeight: 700, cursor: "pointer" }}
                }}, "\\u2260 #" + selectedTrackId) : null,
              e("button", {{
                onClick: () => setNotSameDraft(null),
                style: {{ background: "transparent", color: textSecondary, border: "1px solid " + border, borderRadius: 4, padding: "3px 8px",
                          fontFamily: font, fontSize: 10, cursor: "pointer" }}
              }}, "CANCEL")
            )
        ),
        e("div", {{ style: {{ display: "flex", flexWrap: "wrap", gap: 4 }} }},
          ...notSamePairs.map((pair, idx) =>
            e("span", {{
              key: idx,
              style: {{ background: "#dc262622", color: "#f87171", border: "1px solid #dc262644",
                        borderRadius: 3, padding: "2px 7px", fontSize: 11, fontFamily: font,
                        display: "flex", alignItems: "center", gap: 3 }}
            }},
              pair[0] + " \\u2260 " + pair[1],
              e("span", {{ onClick: () => removeNotSamePair(idx), style: {{ cursor: "pointer", opacity: 0.5 }} }}, "\\u00d7")
            )
          ),
          notSamePairs.length === 0 ? e("span", {{ style: {{ fontSize: 10, color: textSecondary, fontStyle: "italic" }} }}, "none") : null
        )
      )
    ) : null
  );

  /* Export / Import */
  const exportPanel = e("div", {{ style: {{ padding: "14px", marginTop: "auto" }} }},
    e("div", {{ style: {{ display: "flex", gap: 8 }} }},
      e("button", {{
        onClick: exportJSON,
        style: {{ flex: 1, padding: "10px 0", background: "#818cf8", color: "#000", border: "none",
                  borderRadius: 6, fontFamily: font, fontSize: 12, fontWeight: 700, cursor: "pointer", letterSpacing: 0.5 }}
      }}, "\\u2193 EXPORT JSON"),
      e("button", {{
        onClick: importJSON,
        style: {{ flex: 1, padding: "10px 0", background: bg, color: textSecondary, border: "1px solid " + border,
                  borderRadius: 6, fontFamily: font, fontSize: 12, cursor: "pointer", letterSpacing: 0.5 }}
      }}, "\\u2191 IMPORT")
    )
  );

  /* Video-length prompt modal (shown until videoLength is set) */
  const lengthModal = !videoLength ? e("div", {{
    style: {{ position: "fixed", inset: 0, zIndex: 200, background: "rgba(0,0,0,0.85)",
              display: "flex", alignItems: "center", justifyContent: "center" }}
  }},
    e("div", {{ style: {{ background: surface, border: "1px solid " + border, borderRadius: 10,
                          padding: 28, maxWidth: 440, fontFamily: font }} }},
      e("div", {{ style: {{ fontSize: 18, fontWeight: 700, color: textPrimary, marginBottom: 10 }} }},
        "How long is the video?"),
      e("div", {{ style: {{ fontSize: 12, color: textSecondary, lineHeight: 1.5, marginBottom: 16 }} }},
        "We couldn't detect the video length automatically. Please enter the duration in seconds (e.g. 123 for a 2:03 video). This is only asked once."),
      e("div", {{ style: {{ display: "flex", gap: 8 }} }},
        e("input", {{
          type: "number",
          placeholder: "seconds",
          value: lengthInput,
          autoFocus: true,
          onChange: (ev) => setLengthInput(ev.target.value),
          onKeyDown: (ev) => {{
            if (ev.key === "Enter") {{
              const v = parseFloat(lengthInput);
              if (v > 0) {{
                setVideoLength(v);
                localStorage.setItem(storageKey + "_vidlen", String(v));
              }}
            }}
          }},
          style: {{ flex: 1, background: bg, border: "1px solid " + border, color: textPrimary,
                    fontFamily: font, fontSize: 14, padding: "10px 14px", borderRadius: 6, outline: "none" }}
        }}),
        e("button", {{
          onClick: () => {{
            const v = parseFloat(lengthInput);
            if (v > 0) {{
              setVideoLength(v);
              localStorage.setItem(storageKey + "_vidlen", String(v));
            }}
          }},
          style: {{ background: "#818cf8", color: "#000", border: "none", borderRadius: 6,
                    padding: "10px 18px", fontFamily: font, fontSize: 13, fontWeight: 700, cursor: "pointer" }}
        }}, "OK")
      )
    )
  ) : null;

  /* Layout */
  return e("div", {{ style: {{ fontFamily: font, background: bg, color: textPrimary, height: "100vh", display: "flex", flexDirection: "column" }} }},
    lengthModal,
    header,
    e("div", {{ style: {{ display: "flex", flex: 1, overflow: "hidden" }} }},
      /* Left: video + frame bar */
      e("div", {{ style: {{ flex: 1, display: "flex", flexDirection: "column", borderRight: "1px solid " + border }} }},
        videoPanel,
        frameBar
      ),
      /* Right: sidebar */
      e("div", {{ style: {{ width: 340, background: surface, display: "flex", flexDirection: "column", overflow: "auto" }} }},
        visibleTracksList,
        groupsPanel,
        advancedSection,
        exportPanel
      )
    )
  );
}}

ReactDOM.render(e(App), document.getElementById("root"));
</script>
</body>
</html>"""


def main():
    csv_path = None
    video_path = None
    video_length = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--video" and i + 1 < len(args):
            video_path = args[i + 1]
            i += 2
        elif args[i] == "--video-length" and i + 1 < len(args):
            video_length = float(args[i + 1])
            i += 2
        elif csv_path is None:
            csv_path = args[i]
            i += 1
        else:
            i += 1

    # Auto-detect from data/ folder if no CSV/video given
    if not csv_path and not video_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        auto_csv, auto_video, n_videos, n_csvs = auto_detect_files(data_dir)
        if n_videos == 0 or n_csvs == 0:
            print("=" * 60)
            print("No video/CSV found. To get started, put one .mp4 and one .csv")
            print(f"in the 'data/' folder next to this script:")
            print(f"  {data_dir}")
            print()
            print("Or run with explicit paths:")
            print(f"  python {os.path.basename(sys.argv[0])} <tracking_csv> --video <video.mp4>")
            print("=" * 60)
            sys.exit(1)
        if n_videos > 1 or n_csvs > 1:
            print(f"Found {n_videos} .mp4 and {n_csvs} .csv in data/. Using:")
            print(f"  video: {os.path.basename(auto_video)}")
            print(f"  csv:   {os.path.basename(auto_csv)}")
            print(f"(Only one of each expected. Remove extras from data/ to change.)")
        csv_path = auto_csv
        video_path = auto_video

    if not csv_path or not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    if not video_path or not os.path.exists(video_path):
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    # Auto-detect video length via ffprobe if not specified
    if not video_length:
        video_length = get_video_duration(video_path)
        if video_length:
            print(f"Detected video length: {video_length:.2f} seconds (via ffprobe)")
        else:
            print("Could not detect video length automatically. You'll be prompted in the browser.")

    with open(csv_path, "r") as f:
        csv_text = f.read()

    row_count = len(csv_text.strip().split("\n")) - 1
    print(f"Loaded {csv_path} ({row_count} tracking rows)")

    video_url, server = start_video_server(video_path)
    print(f"Video server running at {video_url}")

    html = build_html(csv_text, video_url, video_length)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix="vehicle_annotator_", delete=False
    )
    tmp.write(html)
    tmp.close()

    print(f"Opening annotator in browser...")
    print(f"  File: {tmp.name}")
    print(f"  Keyboard: Space=play/pause, Arrows=step, G=new group, I=isolate, S/E=maneuver start/end")
    print(f"  Click bounding boxes to select tracks, then assign to groups.")
    print(f"  When done, click 'Export JSON' to save annotations.")

    wsl_interop = "/proc/sys/fs/binfmt_misc/WSLInterop"
    if os.path.exists(wsl_interop) or "microsoft" in os.uname().release.lower():
        try:
            import subprocess
            win_path = subprocess.check_output(["wslpath", "-w", tmp.name]).decode().strip()
            subprocess.Popen(["explorer.exe", win_path])
        except Exception as ex:
            print(f"WSL browser launch failed ({ex}), try opening manually:")
            print(f"  Windows path: \\\\wsl$\\Ubuntu{tmp.name}")
    else:
        webbrowser.open("file://" + tmp.name)

    print(f"\nVideo server must stay running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
