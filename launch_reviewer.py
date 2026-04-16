import sys
import os
import json
import webbrowser
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from functools import partial

"""
python launch_reviewer.py relink_report.csv --video better_tests/outputid3duplicatevid_h264.mp4
"""

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
            # Parse "bytes=START-END"
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
        # Quiet down the request logging
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

def build_html(csv_text, video_url=None):
    escaped = json.dumps(csv_text)
    video_url_js = json.dumps(video_url)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relink Reviewer</title>
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
const RAW_CSV = {escaped};
const VIDEO_URL = {video_url_js};
const e = React.createElement;
const useState = React.useState;
const useEffect = React.useEffect;
const useMemo = React.useMemo;
const useCallback = React.useCallback;
const useRef = React.useRef;

function parseTimestamp(ts) {{
  const parts = ts.split(":");
  return parseInt(parts[0], 10) * 60 + parseFloat(parts[1]);
}}

function parseCSV(text) {{
  const lines = text.trim().split("\\n");
  const headers = lines[0].split(",");
  return lines.slice(1).map((line, i) => {{
    const vals = line.split(",");
    const obj = {{ _idx: i }};
    headers.forEach((h, j) => {{ obj[h.trim()] = vals[j] ? vals[j].trim() : ""; }});
    obj.confidence = parseFloat(obj.confidence) || 0;
    obj.gap_seconds = parseFloat(obj.gap_seconds) || 0;
    return obj;
  }});
}}

const ACTION_COLORS = {{
  auto_accept: {{ bg: "#0a2e1a", border: "#16a34a", text: "#4ade80", label: "AUTO" }},
  review: {{ bg: "#2e2000", border: "#ca8a04", text: "#facc15", label: "REVIEW" }},
  low_confidence: {{ bg: "#2e0a0a", border: "#dc2626", text: "#f87171", label: "LOW" }},
}};

const VERDICT_COLORS = {{
  accepted: {{ bg: "#16a34a", text: "#fff" }},
  rejected: {{ bg: "#dc2626", text: "#fff" }},
}};

const font = "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace";
const bg = "#0c0c14";
const surface = "#12121e";
const surfaceHover = "#1a1a2e";
const border = "#2a2a3e";
const textPrimary = "#e2e2f0";
const textSecondary = "#8888a0";

function ConfBar({{ val }}) {{
  const pct = val * 100;
  const color = val >= 0.9 ? "#4ade80" : val >= 0.7 ? "#facc15" : "#f87171";
  return e("div", {{ style: {{ display: "flex", alignItems: "center", gap: 8 }} }},
    e("div", {{ style: {{ width: 60, height: 6, background: "#1a1a2e", borderRadius: 3, overflow: "hidden" }} }},
      e("div", {{ style: {{ width: pct + "%", height: "100%", background: color, borderRadius: 3, transition: "width 0.3s ease" }} }})
    ),
    e("span", {{ style: {{ fontVariantNumeric: "tabular-nums", fontSize: 13, color: color }} }}, val.toFixed(3))
  );
}}

const storageKey = "relink_" + btoa(RAW_CSV.slice(0,80)).replace(/[^a-z0-9]/gi,"").slice(0,16);

function App() {{
  const videoRef = useRef(null);
  const expandedVideoRef = useRef(null);
  const [videoExpanded, setVideoExpanded] = useState(false);
  const [rows, setRows] = useState([]);
  const [search, setSearch] = useState("");
  const [filterAction, setFilterAction] = useState("all");
  const [filterVerdict, setFilterVerdict] = useState("all");
  const [focusIdx, setFocusIdx] = useState(null);
  const [verdicts, setVerdicts] = useState(() => {{
    try {{ return JSON.parse(localStorage.getItem(storageKey) || "{{}}"); }} catch(e) {{ return {{}}; }}
  }});

  useEffect(() => {{
    setRows(parseCSV(RAW_CSV));
  }}, []);

  useEffect(() => {{
    localStorage.setItem(storageKey, JSON.stringify(verdicts));
  }}, [verdicts]);

  const filtered = useMemo(() => {{
    return rows.filter(r => {{
      if (filterAction !== "all" && r.action !== filterAction) return false;
      if (filterVerdict === "pending" && verdicts[r._idx]) return false;
      if (filterVerdict === "accepted" && verdicts[r._idx] !== "accepted") return false;
      if (filterVerdict === "rejected" && verdicts[r._idx] !== "rejected") return false;
      if (search) {{
        const s = search.toLowerCase().trim();
        const matchId = (tid) => {{
          const base = tid.split("_")[0].toLowerCase();
          const full = tid.toLowerCase();
          return base === s || full.startsWith(s + "_") || full === s;
        }};
        return matchId(r.id_lost) || matchId(r.id_gained);
      }}
      return true;
    }});
  }}, [rows, filterAction, filterVerdict, search, verdicts]);

  const setVerdict = useCallback((idx, v) => {{
    setVerdicts(prev => ({{ ...prev, [idx]: v }}));
  }}, []);

  const focusRow = focusIdx !== null ? rows[focusIdx] : null;

  const nextReview = useCallback(() => {{
    const start = focusIdx !== null ? focusIdx + 1 : 0;
    for (let i = start; i < rows.length; i++) {{
      if (!verdicts[rows[i]._idx] && rows[i].action === "review") {{
        setFocusIdx(i); return;
      }}
    }}
    for (let i = 0; i < start; i++) {{
      if (!verdicts[rows[i]._idx] && rows[i].action === "review") {{
        setFocusIdx(i); return;
      }}
    }}
    setFocusIdx(null);
  }}, [focusIdx, rows, verdicts]);

  const stats = useMemo(() => {{
    const total = rows.length;
    const accepted = Object.values(verdicts).filter(v => v === "accepted").length;
    const rejected = Object.values(verdicts).filter(v => v === "rejected").length;
    return {{ total, accepted, rejected, pending: total - accepted - rejected }};
  }}, [rows, verdicts]);

  const exportCSV = () => {{
    const header = "id_lost,last_seen,id_gained,first_seen,confidence,action,gap_seconds,verdict";
    const lines = rows.map(r =>
      r.id_lost+","+r.last_seen+","+r.id_gained+","+r.first_seen+","+r.confidence+","+r.action+","+r.gap_seconds+","+(verdicts[r._idx]||"pending")
    );
    const blob = new Blob([header+"\\n"+lines.join("\\n")], {{ type: "text/csv" }});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "relink_reviewed.csv";
    a.click();
  }};

  useEffect(() => {{
    const handler = (ev) => {{
      if (!focusRow) return;
      if (ev.target.tagName === "INPUT") return;
      if (ev.key === "a" || ev.key === "A") {{ setVerdict(focusRow._idx, "accepted"); nextReview(); }}
      else if (ev.key === "r" || ev.key === "R") {{ setVerdict(focusRow._idx, "rejected"); nextReview(); }}
      else if (ev.key === "n" || ev.key === "N") {{ nextReview(); }}
      else if (ev.key === "Escape") {{ if (videoExpanded) {{ setVideoExpanded(false); }} else {{ setFocusIdx(null); }} }}
    }};
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }}, [focusRow, nextReview, setVerdict, videoExpanded]);

  useEffect(() => {{
    if (!focusRow || !VIDEO_URL) return;
    const vid = videoRef.current;
    if (!vid) return;
    const startTime = Math.max(0, parseTimestamp(focusRow.last_seen) - 2);
    const endTime = parseTimestamp(focusRow.first_seen) + 2;
    vid.currentTime = startTime;
    setVideoExpanded(false);
    const onTimeUpdate = () => {{
      if (vid.currentTime >= endTime) {{ vid.pause(); }}
    }};
    vid.addEventListener("timeupdate", onTimeUpdate);
    return () => vid.removeEventListener("timeupdate", onTimeUpdate);
  }}, [focusIdx]);

  // HEADER
  const header = e("div", {{
    style: {{ padding: "20px 24px", borderBottom: "1px solid "+border, display: "flex",
              alignItems: "center", justifyContent: "space-between", background: surface }}
  }},
    e("div", {{ style: {{ display: "flex", alignItems: "center", gap: 12 }} }},
      e("div", {{ style: {{ width: 8, height: 8, borderRadius: "50%", background: "#4ade80",
                            boxShadow: "0 0 8px #4ade8066" }} }}),
      e("span", {{ style: {{ fontSize: 15, fontWeight: 700, letterSpacing: 1 }} }}, "RELINK REVIEWER"),
      e("span", {{ style: {{ fontSize: 11, color: textSecondary }} }}, "LOCAL")
    ),
    e("div", {{ style: {{ display: "flex", gap: 16, fontSize: 12 }} }},
      e("span", {{ style: {{ color: "#4ade80" }} }}, "\\u2713 " + stats.accepted),
      e("span", {{ style: {{ color: "#f87171" }} }}, "\\u2717 " + stats.rejected),
      e("span", {{ style: {{ color: textSecondary }} }}, "\\u25FB " + stats.pending),
      e("span", {{ style: {{ color: textSecondary }} }}, "/ " + stats.total)
    )
  );

  // TOOLBAR
  const toolbar = e("div", {{
    style: {{ padding: "12px 16px", display: "flex", gap: 8, alignItems: "center",
              borderBottom: "1px solid "+border, flexWrap: "wrap" }}
  }},
    e("input", {{
      type: "text", placeholder: "Search base ID...", value: search,
      onChange: ev => setSearch(ev.target.value),
      style: {{ background: bg, border: "1px solid "+border, borderRadius: 4, color: textPrimary,
                fontFamily: font, fontSize: 12, padding: "6px 10px", width: 140, outline: "none" }}
    }}),
    ...["all","auto_accept","review","low_confidence"].map(f =>
      e("button", {{
        key: f, onClick: () => setFilterAction(f),
        style: {{ background: filterAction===f ? surfaceHover : "transparent",
                  color: filterAction===f ? textPrimary : textSecondary,
                  border: "1px solid "+(filterAction===f ? "#4a4a6e" : border),
                  borderRadius: 4, padding: "4px 10px", fontFamily: font, fontSize: 11,
                  cursor: "pointer", textTransform: "uppercase" }}
      }}, f==="all"?"ALL":f==="auto_accept"?"AUTO":f==="review"?"REVIEW":"LOW")
    ),
    e("div", {{ style: {{ flex: 1 }} }}),
    ...["all","pending","accepted","rejected"].map(f =>
      e("button", {{
        key: f, onClick: () => setFilterVerdict(f),
        style: {{ background: filterVerdict===f ? surfaceHover : "transparent",
                  color: filterVerdict===f ? textPrimary : textSecondary,
                  border: "1px solid "+(filterVerdict===f ? "#4a4a6e" : border),
                  borderRadius: 4, padding: "4px 8px", fontFamily: font, fontSize: 10,
                  cursor: "pointer", textTransform: "uppercase" }}
      }}, f)
    )
  );

  // TABLE ROWS
  const tableRows = filtered.map((r, i) => {{
    const ac = ACTION_COLORS[r.action] || ACTION_COLORS.review;
    const v = verdicts[r._idx];
    const isFocused = focusIdx !== null && rows[focusIdx] && rows[focusIdx]._idx === r._idx;
    return e("tr", {{
      key: r._idx,
      onClick: () => setFocusIdx(rows.findIndex(x => x._idx === r._idx)),
      style: {{ cursor: "pointer", background: isFocused ? "#1e1e3a" : i%2===0 ? "transparent" : "#0e0e18",
                borderLeft: isFocused ? "3px solid #818cf8" : "3px solid transparent" }},
      onMouseEnter: ev => {{ if(!isFocused) ev.currentTarget.style.background = surfaceHover; }},
      onMouseLeave: ev => {{ if(!isFocused) ev.currentTarget.style.background = i%2===0?"transparent":"#0e0e18"; }}
    }},
      e("td", {{ style: {{ padding: "6px 10px" }} }},
        e("div", {{ style: {{ fontSize: 9, fontWeight: 700, color: ac.text, background: ac.bg,
                              border: "1px solid "+ac.border, borderRadius: 3, padding: "2px 6px",
                              textAlign: "center", width: 40 }} }}, ac.label)
      ),
      e("td", {{ style: {{ padding: "6px 10px", fontWeight: 600, whiteSpace: "nowrap" }} }}, r.id_lost),
      e("td", {{ style: {{ padding: "6px 6px", color: textSecondary, fontSize: 11, whiteSpace: "nowrap" }} }}, r.last_seen),
      e("td", {{ style: {{ padding: "6px 10px", fontWeight: 600, whiteSpace: "nowrap" }} }},
        e("span", {{ style: {{ color: textSecondary, marginRight: 4 }} }}, "\\u2192"), r.id_gained),
      e("td", {{ style: {{ padding: "6px 6px", color: textSecondary, fontSize: 11, whiteSpace: "nowrap" }} }}, r.first_seen),
      e("td", {{ style: {{ padding: "6px 10px" }} }}, e(ConfBar, {{ val: r.confidence }})),
      e("td", {{ style: {{ padding: "6px 10px", color: textSecondary, fontVariantNumeric: "tabular-nums" }} }}, r.gap_seconds+"s"),
      e("td", {{ style: {{ padding: "6px 10px" }} }},
        v ? e("span", {{ style: {{ fontSize: 10, fontWeight: 700, color: VERDICT_COLORS[v].text,
                                    background: VERDICT_COLORS[v].bg, borderRadius: 3, padding: "2px 8px" }} }},
              v==="accepted"?"\\u2713":"\\u2717")
          : e("span", {{ style: {{ color: "#444", fontSize: 11 }} }}, "\\u2014")
      )
    );
  }});

  // DETAIL PANEL
  let detailContent;
  if (focusRow) {{
    const ac = ACTION_COLORS[focusRow.action] || ACTION_COLORS.review;
    const confColor = focusRow.confidence >= 0.9 ? "#4ade80" : focusRow.confidence >= 0.7 ? "#facc15" : "#f87171";
    detailContent = e("div", {{ style: {{ padding: 20 }} }},
      e("div", {{ style: {{ display: "inline-block", fontSize: 10, fontWeight: 700, color: ac.text,
                            background: ac.bg, border: "1px solid "+ac.border, borderRadius: 4,
                            padding: "3px 10px", marginBottom: 20, textTransform: "uppercase", letterSpacing: 1 }} }},
        focusRow.action.replace("_"," ")),
      VIDEO_URL ? e("div", {{ style: {{ marginBottom: 16 }} }},
        e("div", {{ style: {{ position: "relative", cursor: "pointer" }}, onClick: () => setVideoExpanded(true) }},
          e("video", {{ ref: videoRef, src: VIDEO_URL, controls: true, preload: "metadata",
            style: {{ width: "100%", borderRadius: 6, background: "#000" }} }}),
          e("div", {{ style: {{ position: "absolute", top: 6, right: 6, background: "rgba(0,0,0,0.7)",
            color: "#fff", fontSize: 10, padding: "2px 6px", borderRadius: 3, pointerEvents: "none" }} }}, "\\u26F6 expand")
        ),
        e("div", {{ style: {{ fontSize: 9, color: textSecondary, marginTop: 4, textAlign: "center" }} }},
          "Seeking to " + focusRow.last_seen + " \\u2192 " + focusRow.first_seen + " (auto-pauses)")
      ) : null,
      e("div", {{ style: {{ background: bg, borderRadius: 8, padding: 20, marginBottom: 20 }} }},
        e("div", {{ style: {{ textAlign: "center", marginBottom: 8 }} }},
          e("div", {{ style: {{ fontSize: 22, fontWeight: 700, color: "#818cf8" }} }}, focusRow.id_lost),
          e("div", {{ style: {{ fontSize: 11, color: textSecondary, marginTop: 2 }} }}, "last seen "+focusRow.last_seen)
        ),
        e("div", {{ style: {{ textAlign: "center", padding: "8px 0", color: textSecondary }} }},
          e("div", {{ style: {{ fontSize: 18 }} }}, "\\u2193"),
          e("div", {{ style: {{ fontSize: 11, background: surfaceHover, display: "inline-block",
                                padding: "2px 10px", borderRadius: 10, marginTop: 2 }} }}, focusRow.gap_seconds+"s gap")
        ),
        e("div", {{ style: {{ textAlign: "center", marginTop: 8 }} }},
          e("div", {{ style: {{ fontSize: 22, fontWeight: 700, color: "#818cf8" }} }}, focusRow.id_gained),
          e("div", {{ style: {{ fontSize: 11, color: textSecondary, marginTop: 2 }} }}, "first seen "+focusRow.first_seen)
        )
      ),
      e("div", {{ style: {{ marginBottom: 20 }} }},
        e("div", {{ style: {{ fontSize: 10, color: textSecondary, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }} }}, "Confidence"),
        e("div", {{ style: {{ width: "100%", height: 10, background: bg, borderRadius: 5, overflow: "hidden" }} }},
          e("div", {{ style: {{ width: (focusRow.confidence*100)+"%", height: "100%", background: confColor, borderRadius: 5 }} }})
        ),
        e("div", {{ style: {{ fontSize: 28, fontWeight: 700, marginTop: 6, color: confColor, fontVariantNumeric: "tabular-nums" }} }},
          (focusRow.confidence*100).toFixed(1)+"%")
      ),
      e("div", {{ style: {{ display: "flex", gap: 8, marginBottom: 16 }} }},
        e("button", {{
          onClick: () => {{ setVerdict(focusRow._idx, "accepted"); nextReview(); }},
          style: {{ flex: 1, padding: "12px 0", background: verdicts[focusRow._idx]==="accepted"?"#16a34a":bg,
                    color: verdicts[focusRow._idx]==="accepted"?"#fff":"#4ade80", border: "1px solid #16a34a",
                    borderRadius: 6, fontFamily: font, fontSize: 13, fontWeight: 700, cursor: "pointer" }}
        }}, "\\u2713 ACCEPT (A)"),
        e("button", {{
          onClick: () => {{ setVerdict(focusRow._idx, "rejected"); nextReview(); }},
          style: {{ flex: 1, padding: "12px 0", background: verdicts[focusRow._idx]==="rejected"?"#dc2626":bg,
                    color: verdicts[focusRow._idx]==="rejected"?"#fff":"#f87171", border: "1px solid #dc2626",
                    borderRadius: 6, fontFamily: font, fontSize: 13, fontWeight: 700, cursor: "pointer" }}
        }}, "\\u2717 REJECT (R)")
      ),
      e("button", {{
        onClick: nextReview,
        style: {{ width: "100%", padding: "8px 0", background: "transparent", color: textSecondary,
                  border: "1px solid "+border, borderRadius: 6, fontFamily: font, fontSize: 11, cursor: "pointer" }}
      }}, "SKIP TO NEXT REVIEW (N)"),
      e("div", {{ style: {{ marginTop: 20, fontSize: 10, color: "#555", lineHeight: 1.6 }} }},
        "Keyboard: A = accept, R = reject, N = next review, Esc = deselect")
    );
  }} else {{
    detailContent = e("div", {{
      style: {{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center", padding: 30, color: textSecondary, textAlign: "center" }}
    }},
      e("div", {{ style: {{ fontSize: 32, marginBottom: 12, opacity: 0.3 }} }}, "\\u25CE"),
      e("div", {{ style: {{ fontSize: 12, marginBottom: 8 }} }}, "Click a row to inspect"),
      e("div", {{ style: {{ fontSize: 11 }} }},
        "Or press ",
        e("span", {{ onClick: nextReview, style: {{ color: "#818cf8", cursor: "pointer", textDecoration: "underline" }} }}, "N"),
        " to jump to first review item"
      )
    );
  }}

  const videoOverlay = videoExpanded && focusRow && VIDEO_URL ? e("div", {{
    onClick: (ev) => {{ if (ev.target === ev.currentTarget) setVideoExpanded(false); }},
    style: {{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,0.85)",
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}
  }},
    e("video", {{ ref: expandedVideoRef, src: VIDEO_URL, controls: true, autoPlay: false,
      style: {{ maxWidth: "90vw", maxHeight: "80vh", borderRadius: 8, background: "#000" }},
      onLoadedMetadata: () => {{
        if (expandedVideoRef.current && videoRef.current) {{
          expandedVideoRef.current.currentTime = videoRef.current.currentTime;
        }}
      }}
    }}),
    e("div", {{ style: {{ marginTop: 12, fontSize: 11, color: textSecondary }} }},
      focusRow.id_lost + " \\u2192 " + focusRow.id_gained + "  |  " + focusRow.last_seen + " \\u2192 " + focusRow.first_seen),
    e("div", {{ style: {{ marginTop: 8, fontSize: 10, color: "#555" }} }}, "Click outside or press Esc to close")
  ) : null;

  return e("div", {{ style: {{ fontFamily: font, background: bg, color: textPrimary, height: "100vh" }} }},
    videoOverlay,
    header,
    e("div", {{ style: {{ display: "flex", height: "calc(100vh - 65px)" }} }},
      e("div", {{ style: {{ flex: 1, display: "flex", flexDirection: "column", borderRight: "1px solid "+border, overflow: "hidden" }} }},
        toolbar,
        e("div", {{ style: {{ flex: 1, overflow: "auto" }} }},
          e("table", {{ style: {{ width: "100%", borderCollapse: "collapse", fontSize: 12 }} }},
            e("thead", null,
              e("tr", {{ style: {{ background: surface, position: "sticky", top: 0, zIndex: 1 }} }},
                ...["","ID Lost","@","ID Gained","@","Conf","Gap","Verdict"].map((h,i) =>
                  e("th", {{ key: i, style: {{ padding: "8px 10px", textAlign: "left", color: textSecondary,
                                                fontWeight: 600, fontSize: 10, textTransform: "uppercase",
                                                letterSpacing: 0.5, borderBottom: "1px solid "+border, whiteSpace: "nowrap" }} }}, h)
                )
              )
            ),
            e("tbody", null, ...tableRows)
          ),
          filtered.length === 0 ? e("div", {{ style: {{ textAlign: "center", padding: 40, color: textSecondary, fontSize: 13 }} }},
            "No pairs match current filters.") : null
        )
      ),
      e("div", {{ style: {{ width: 320, background: surface, display: "flex", flexDirection: "column", overflow: "auto" }} }},
        detailContent,
        e("div", {{ style: {{ padding: 16, borderTop: "1px solid "+border, marginTop: "auto" }} }},
          e("button", {{
            onClick: exportCSV,
            style: {{ width: "100%", padding: "10px 0", background: bg, color: textSecondary,
                      border: "1px solid "+border, borderRadius: 6, fontFamily: font, fontSize: 11,
                      cursor: "pointer", letterSpacing: 0.5 }}
          }}, "\\u2193 EXPORT WITH VERDICTS")
        )
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
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--video" and i + 1 < len(args):
            video_path = args[i + 1]
            i += 2
        elif csv_path is None:
            csv_path = args[i]
            i += 1
        else:
            i += 1
    csv_path = csv_path or "relink_report.csv"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print(f"Usage: python {sys.argv[0]} <csv> [--video <path_to_video.mp4>]")
        sys.exit(1)

    if video_path and not os.path.exists(video_path):
        print(f"Error: video file {video_path} not found")
        sys.exit(1)

    with open(csv_path, "r") as f:
        csv_text = f.read()

    row_count = len(csv_text.strip().split("\n")) - 1
    print(f"Loaded {csv_path} ({row_count} pairs)")

    video_url = None
    if video_path:
        video_url, server = start_video_server(video_path)
        print(f"Video server running at {video_url}")

    html = build_html(csv_text, video_url)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix="relink_reviewer_", delete=False
    )
    tmp.write(html)
    tmp.close()

    print(f"Opening reviewer in browser...")
    print(f"  File: {tmp.name}")
    print(f"  Keyboard: A=accept, R=reject, N=next review, Esc=deselect")
    print(f"  When done, click 'Export with Verdicts' to save results.")

    # WSL2: use explorer.exe to open in Windows browser
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

    if video_path:
        print(f"\nVideo server must stay running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()