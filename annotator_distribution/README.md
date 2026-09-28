# Vehicle Annotator

Thank you for helping label vehicles! This tool runs locally on your computer and takes about 5 minutes to set up the first time.

## Quick Start (3 steps)

### 1. Download this folder

Click the green **Code** button at the top of the GitHub page → **Download ZIP**. Unzip it somewhere easy to find (Desktop is fine).

### 2. Install Python (skip if you already have it)

- **Windows**: Download from [python.org/downloads](https://www.python.org/downloads/). During install, **check the box "Add Python to PATH"**.
- **macOS**: Open Terminal and run `python3 --version`. If it says "command not found", install it from [python.org/downloads](https://www.python.org/downloads/).
- **Linux**: You already have it.

### 3. Launch the tool

- **Windows**: Double-click `start.bat`
- **macOS / Linux**: Double-click `start.sh` (if that does nothing, open a terminal in this folder and run `./start.sh`)

Your browser will open with the annotator. If it doesn't, look for a link in the terminal window and click it.

---

## What am I doing?

A computer-vision system automatically tracked vehicles in a video and gave each one a numeric ID. But the system sometimes loses a vehicle and gives it a **new ID** when it reappears. **Your job** is to tell us which IDs actually belong to the same real-world vehicle.

### The workflow

1. **Play the video** and watch the numbered bounding boxes on each vehicle.
2. When you see a vehicle you want to label, click its box (or click its number in the sidebar) to select it.
3. Click **+ NEW GROUP** on the right.
4. With the box still selected, click the **+ #ID** button on your new group. That vehicle is now in the group.
5. Scrub forward. If the same real-world vehicle appears again with a different number, select that number and add it to the same group.
6. Repeat until you've grouped every vehicle you can match up.
7. Click **⬇ EXPORT JSON** at the bottom right — a file called `annotations.json` will download.
8. **Email that file back to me.**

That's it!

### Tips

- A vehicle only belongs to one group. If you put it in the wrong group, just add it to a different one — it'll move.
- You don't have to label every vehicle — just the ones you're confident about.
- If a vehicle only has one ID the whole video (never reappears with a new number), you can skip it. Or, if you want to be thorough, use the **Advanced** section's "Isolated Tracks" button to mark it as confirmed-single.
- Your work **auto-saves** in the browser. Closing and reopening won't lose progress.

### Keyboard shortcuts

| Key | Does |
|---|---|
| Space | Play / pause |
| Left / Right arrows | Step one frame |
| Shift + arrows | Jump 5 seconds |
| G | New group |
| I | Mark selected as isolated |
| O | Toggle bounding-box overlay |
| Esc | Deselect |

---

## Troubleshooting

**"Python is not installed"** — Install from [python.org](https://www.python.org/downloads/) and try again. On Windows make sure you checked "Add Python to PATH" during install.

**Browser doesn't open** — Look at the terminal window that opened. It'll print a link like `file:///tmp/...html`. Copy that and paste it into your browser.

**"No video/CSV found"** — Make sure the `data/` folder has one `.mp4` file and one `.csv` file. They should already be there — if you deleted them, re-download the ZIP.

**Video won't play** — Try Chrome or Firefox. Safari sometimes has issues with the video format.

**Something else broke** — Email me a screenshot and I'll help.
