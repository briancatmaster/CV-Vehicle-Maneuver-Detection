import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from collections import defaultdict
import joblib
import sys
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict

CSV_PATH = "better_tests/tracksid3duplicate.csv"
OTHER_CSV_PATH = "better_tests/airport_tracks.csv"
PARKING_VIDEO_LENGTH = 123
AIRPORT_VIDEO_LENGTH = 20

@dataclass
class Vehicle_Pair:
    time_diff: int = 0
    pix_dist: float = 0.0

    def confidence(self) -> float:
        return 1.0 #insert function later f(time_diff, pix_dist, velocity_error, trajectory_smoothness)

parking_chains = [
    [5, 126, 133, 138, 252, 273], # Black SUV
    [160, 249, 281, 282, 294, 296, 298, 302], # White SUV
    [56, 66, 68, 71, 85, 108, 111], # Unnamed Vehicle A
    [263, 287], # Stationary Vehicle
]

parking_lot_not_same = {(4, 238), (238, 4)}
parking_isolated = set()

airport_chains = [
    [1, 24, 1],
    [5, 21, 35, 21, 102],
    [4, 25, 16],
    [16, 42, 71, 127],
    [20, 70],
    [53, 93],
    [26, 41],
    [19, 28, 44, 45, 87, 98, 87, 98, 87, 129],
    [38, 79, 105, 108, 117, 142],
    [12, 15, 23, 30, 55, 66, 55, 73, 84, 110],
    [36, 57, 64],
    [11, 27, 36],
    [60, 83, 96, 141],
]
airport_isolated = {2, 3, 6, 7, 32, 78, 88, 125}

def map_chains_to_tracklets(chains, df_split):
    lookup = defaultdict(list)
    for tid, grp in df_split.groupby('tracklet_id'):
        orig_id = int(tid.split('_')[0])
        lookup[orig_id].append((tid, grp['frame'].min(), grp['frame'].max()))
    
    for orig_id in lookup:
        lookup[orig_id].sort(key=lambda x: x[1])  # sort by first_frame
    
    tracklet_chains = []
    for chain in chains:
        last_end_frame = -1
        mapped = []
        ok = True
        
        for orig_id in chain:
            if orig_id not in lookup:
                print(f"  WARNING: ID {orig_id} not found")
                ok = False
                break
            
            picked = None
            for tid, first_frame, last_frame in lookup[orig_id]:
                if first_frame > last_end_frame:
                    picked = tid
                    last_end_frame = last_frame
                    break
            
            if picked is None:
                print(f"  WARNING: No valid tracklet for ID {orig_id} after frame {last_end_frame}")
                ok = False
                break
            
            mapped.append(picked)
        
        if ok:
            tracklet_chains.append(mapped)
            print(f"  {chain} → {mapped}")
    
    return tracklet_chains

def is_same_vehicle(tid1, tid2, tracklet_chains, not_same_pairs=set(), isolated=set()):
    """
    tid1, tid2: tracklet ID strings like "16_0"
    tracklet_chains: list of lists of tracklet ID strings
    isolated: set of ORIGINAL integer IDs
    """
    tid1, tid2 = str(tid1), str(tid2)
    
    if (tid1, tid2) in not_same_pairs or (tid2, tid1) in not_same_pairs:
        return 0
    
    base1 = int(tid1.split('_')[0])
    base2 = int(tid2.split('_')[0])
    if base1 in isolated or base2 in isolated:
        return 0
    
    # Check if ANY chain contains BOTH — handles shared tracklets
    for chain in tracklet_chains:
        if tid1 in chain and tid2 in chain:
            return 1
    
    # Both in known chains but never together = different vehicles
    in_any_1 = any(tid1 in c for c in tracklet_chains)
    in_any_2 = any(tid2 in c for c in tracklet_chains)

    if in_any_1 and in_any_2:
        return 0

    if in_any_1 or in_any_2:
        return 0
    
    return None

df1 = pd.read_csv(CSV_PATH)
df2 = pd.read_csv(OTHER_CSV_PATH)

#Based on 100 frames * 123 seconds / 7390 frames
#MAX_FRAME_GAP = 1.6644 * last_frame / VIDEO_LENGTH

"""
def frame_to_timestamp(frame):
    ratio = frame / last_frame
    timestamp_secs = int(ratio * VIDEO_LENGTH)
    minutes = timestamp_secs // 60
    seconds = timestamp_secs % 60
    str_seconds = ""
    
    if (seconds < 10):
        str_seconds = "0" + str(seconds)
    else:
        str_seconds = str(seconds)

    return str(minutes) + ":" + str_seconds
"""
    
def split_into_tracklets(df_raw, max_gap=6):
    """
    If an ID disappears for more than max_gap frames and comes back,
    treat the reappearance as a new tracklet.
    
    Adds a 'tracklet_id' column: "1_0", "1_1", "24_0", etc.
    Original 'track_id' is preserved.
    """
    df_raw['cx'] = (df_raw['x1'] + df_raw['x2']) / 2
    df_raw['cy'] = (df_raw['y1'] + df_raw['y2']) / 2
    #last_frame = df_raw['frame'].max()

    df = df_raw.copy().sort_values(['track_id', 'frame']).reset_index(drop=True)
    frame_diff = df.groupby('track_id')['frame'].diff()
    new_segment = (frame_diff > max_gap).fillna(False).astype(int)
    sub_id = new_segment.groupby(df['track_id']).cumsum()
    df['tracklet_id'] = df['track_id'].astype(str) + "_" + sub_id.astype(str)
    
    return df

def get_avg_heading(track_dict, use_last=True, n_frames=5):
    vels = track_dict['velocities']
    if use_last:
        subset = vels[-n_frames:]
    else:
        subset = vels[:n_frames]
    
    avg_vx = np.mean([v[0] for v in subset])
    avg_vy = np.mean([v[1] for v in subset])
    return math.atan2(avg_vy, avg_vx)

#Helper function so we can use BOTH datasets
def generate_candidate_pairs(raw_df, video_length, max_gap=6):
    raw_df = split_into_tracklets(raw_df, max_gap=max_gap)
    tracks = {}
    #Groups the DataFrame by track_id. (g is the df containing only rows for the specific track)
    for tracklet_id, g in raw_df.groupby('tracklet_id'):
        tracks[tracklet_id] = {
            'frames': g['frame'].tolist(),
            'positions': list(zip(g['cx'], g['cy'])),
            'bboxes': list(zip(g['x1'], g['y1'], g['x2']-g['x1'], g['y2']-g['y1'])),
            'velocities': list(zip(g['velocity_x'], g['velocity_y'])),
            'first_velocity': (g['velocity_x'].iloc[0], g['velocity_y'].iloc[0]),
            'last_velocity': (g['velocity_x'].iloc[-1], g['velocity_y'].iloc[-1]),
            'first_frame': g['frame'].iloc[0],
            'last_frame': g['frame'].iloc[-1],
            'first_position': (g['cx'].iloc[0], g['cy'].iloc[0]),
            'last_position': (g['cx'].iloc[-1], g['cy'].iloc[-1])
        }

    potential_pairs = {}

    for track_A, dfA in tracks.items():
        for track_B, dfB in tracks.items():
            if track_A == track_B:
                continue
            else:
                if dfB['first_frame'] <= dfA['last_frame']:
                    continue
            
            current_pair = Vehicle_Pair()
            current_pair.time_diff = dfB['first_frame'] - dfA['last_frame']
            MAX_FRAME_GAP = 10 * (raw_df['frame'].max() - raw_df['frame'].min()) / video_length

            if current_pair.time_diff > MAX_FRAME_GAP:
                continue

            last_A_position = dfA['last_position']
            first_B_position = dfB['first_position']

            current_pair.pix_dist = math.sqrt(math.pow(last_A_position[0] - first_B_position[0], 2) + math.pow(last_A_position[1] - first_B_position[1], 2))
            MAX_PIXEL_DIST = 300
            #MAX_PIXEL_DIST = max(40 * current_pair.time_diff / (MAX_FRAME_GAP * 0.3), 40)

            #We can tune this MAX_PIXEL_DIST later
            if current_pair.pix_dist < MAX_PIXEL_DIST:
                potential_pairs[(track_A, track_B)] = current_pair
            #    print("We've identifed two ids that may be the same vehicle with time & space logic!")
            #    print(str(track_A) + " | last seen at " + str(dfA['last_frame']) + " frame; " + str(track_B) + " | first seen at " + str(dfB['first_frame']) + " frame")
            #    print(frame_to_timestamp(dfA['last_frame']) + "-->" + frame_to_timestamp(dfB['first_frame']) + "\n")
    
    ml_data = []
    for (i, j), info in potential_pairs.items():
        dfA = tracks[i]
        dfB = tracks[j]
        xI, yI = dfA["last_position"]
        xJ, yJ = dfB["first_position"]

        #Kalman
        vx = dfA['last_velocity'][0]
        vy = dfA['last_velocity'][1]
        velocityB = dfB['first_velocity']

        x_pred = xI + (info.time_diff * vx)
        y_pred = yI + (info.time_diff * vy)
        
        trajectory_error = np.sqrt((x_pred - xJ)**2 + (y_pred - yJ)**2)
        traj_sqrt = trajectory_error / math.sqrt(info.time_diff)
        velocity_error = np.sqrt((vx - velocityB[0])**2 + (vy - velocityB[1])**2)
        speed = info.pix_dist / info.time_diff

        #Box Area Ratio
        w_a, h_a = get_stable_box(dfA['bboxes'], use_last=True)
        w_b, h_b = get_stable_box(dfB['bboxes'], use_last=False)
        area_a = w_a * h_a
        area_b = w_b * h_b
        box_area_ratio = min(area_a, area_b) / max(area_a, area_b) if max(area_a, area_b) > 0 else 0

        #Aspect Ratio Similarity
        aspect_a = w_a / h_a if h_a > 0 else 0
        aspect_b = w_b / h_b if h_b > 0 else 0
        aspect_ratio_diff = abs(aspect_a - aspect_b)

        #Heading Angle Difference
        heading_a = get_avg_heading(dfA, use_last=True)
        heading_b = get_avg_heading(dfB, use_last=False)

        heading_diff = abs(heading_a - heading_b)
        if heading_diff > math.pi:
            heading_diff = 2 * math.pi - heading_diff

        #Box Area Change Rate
        bboxes_A = dfA['bboxes']
        last_bboxes_A = bboxes_A[-10:]
        if len(last_bboxes_A) > 1:
            areas_A = [b[2] * b[3] for b in last_bboxes_A]
            area_growth_A = (areas_A[-1] - areas_A[0]) / len(areas_A)
        else:
            area_growth_A = 0

        bboxes_B = dfB['bboxes']
        first_bboxes_B = bboxes_B[:10]
        if len(first_bboxes_B) > 1:
            areas_B = [b[2] * b[3] for b in first_bboxes_B]
            area_growth_B = (areas_B[-1] - areas_B[0]) / len(areas_B)
        else:
            area_growth_B = 0

        area_growth_diff = abs(area_growth_A - area_growth_B)

        ml_data.append({
                'id1': i,
                'id2': j,
                #'trajectory_error': trajectory_error,
                'traj_sqrt': traj_sqrt,
                'velocity_error': velocity_error,
                'speed': speed,
                'time_diff': info.time_diff,
                'pix_dist': info.pix_dist,
                'box_area_ratio': box_area_ratio,
                'aspect_ratio_diff': aspect_ratio_diff,
                'heading_diff': heading_diff,
                'area_growth_diff': area_growth_diff
            })
    return ml_data, tracks

def get_stable_box(bboxes, use_last=True, n_frames=5):    
    if use_last:
        subset = bboxes[-n_frames:]
    else:
        subset = bboxes[:n_frames]
    
    avg_w = np.mean([b[2] for b in subset])
    avg_h = np.mean([b[3] for b in subset])
    return avg_w, avg_h

def build_relinked_chains(df_predictions, raw_df, threshold=0.50, max_gap=6):
    """
    Take model predictions and build connected vehicle chains.
    Uses a greedy approach: for each tracklet, pick its best 
    forward link (highest confidence) to avoid conflicts.
    Then merges flickering chains (same base ID pair, no temporal overlap).
    """

    accepted = df_predictions[df_predictions['confidence'] >= threshold].copy()
    accepted = accepted.sort_values('confidence', ascending=False)
    used_as_source = set()
    used_as_target = set()
    clean_links = []

    for _, row in accepted.iterrows():
        src, tgt = row['id1'], row['id2']
        if src not in used_as_source and tgt not in used_as_target:
            clean_links.append((src, tgt, row['confidence']))
            used_as_source.add(src)
            used_as_target.add(tgt)

    G = nx.DiGraph()
    for src, tgt, conf in clean_links:
        G.add_edge(src, tgt, confidence=conf)

    chains = []
    for component in nx.weakly_connected_components(G):
        subgraph = G.subgraph(component)
        starts = [n for n in subgraph if subgraph.in_degree(n) == 0]

        for start in starts:
            chain = [start]
            current = start
            while True:
                successors = list(subgraph.successors(current))
                if not successors:
                    break
                current = successors[0]
                chain.append(current)

            if len(chain) > 1:
                chains.append(chain)

    # --- Merge flickering chains (same base ID pair, no temporal overlap) ---
    df_split = split_into_tracklets(raw_df, max_gap=max_gap)
    tracklet_frames = {}
    for tid, grp in df_split.groupby('tracklet_id'):
        tracklet_frames[tid] = (int(grp['frame'].min()), int(grp['frame'].max()))

    merged = True
    while merged:
        merged = False
        for i in range(len(chains)):
            if chains[i] is None:
                continue
            ids_i = {t.split('_')[0] for t in chains[i]}
            for j in range(i + 1, len(chains)):
                if chains[j] is None:
                    continue
                ids_j = {t.split('_')[0] for t in chains[j]}
                if len(ids_i & ids_j) < 2:
                    continue
                # Check for temporal overlap between different base IDs
                combined = chains[i] + chains[j]
                has_overlap = False
                for idx_a in range(len(combined)):
                    for idx_b in range(idx_a + 1, len(combined)):
                        a, b = combined[idx_a], combined[idx_b]
                        if a.split('_')[0] == b.split('_')[0]:
                            continue
                        if a in tracklet_frames and b in tracklet_frames:
                            a_s, a_e = tracklet_frames[a]
                            b_s, b_e = tracklet_frames[b]
                            if a_s <= b_e and b_s <= a_e:
                                has_overlap = True
                                break
                    if has_overlap:
                        break
                if not has_overlap:
                    combined_sorted = sorted(
                        set(combined),
                        key=lambda t: tracklet_frames.get(t, (0, 0))[0]
                    )
                    chains[i] = combined_sorted
                    ids_i = {t.split('_')[0] for t in chains[i]}
                    chains[j] = None
                    merged = True
        chains = [c for c in chains if c is not None]

    return chains, clean_links

def summarize_chains(chains, links, raw_df, video_length, max_gap=6):
    df = split_into_tracklets(raw_df, max_gap=max_gap).copy()

    fps = (raw_df['frame'].max() - raw_df['frame'].min()) / video_length

    # Build lookup: tracklet_id -> frame range
    tracklet_info = {}
    for tid, grp in df.groupby('tracklet_id'):
        tracklet_info[tid] = {
            'first_frame': int(grp['frame'].min()),
            'last_frame': int(grp['frame'].max()),
            'detections': int(len(grp)),
            'base_id': str(grp['track_id'].iloc[0]),
        }

    print(
        f"{'Chain':<10} {'Base ID':<10} {'Orig IDs':<18} {'Tracklets':<9} "
        f"{'First Seen':<12} {'Last Seen':<12} {'Duration':<10} "
        f"{'Edges':<6} {'AvgConf':<8} {'MinConf':<8} {'MaxGap'}"
    )
    print("-" * 120)

    for i, chain in enumerate(chains, start=1):
        chain_id = f"chain_{i:03d}"

        valid_tracklets = [t for t in chain if t in tracklet_info]
        if not valid_tracklets:
            continue

        first_frame = min(tracklet_info[t]['first_frame'] for t in valid_tracklets)
        last_frame = max(tracklet_info[t]['last_frame'] for t in valid_tracklets)

        first_sec = first_frame / fps
        last_sec = last_frame / fps
        duration = last_sec - first_sec

        first_ts = f"{int(first_sec // 60)}:{first_sec % 60:05.2f}"
        last_ts = f"{int(last_sec // 60)}:{last_sec % 60:05.2f}"

        base_id = valid_tracklets[0].split('_')[0]
        orig_ids = sorted({t.split('_')[0] for t in valid_tracklets})
        orig_ids_str = ",".join(orig_ids[:4])
        if len(orig_ids) > 4:
            orig_ids_str += "..."

        # Only edges that connect consecutive tracklets in this chain
        chain_edges = []
        for j in range(1, len(valid_tracklets)):
            prev_t = valid_tracklets[j - 1]
            curr_t = valid_tracklets[j]
            edge = next(
                (l for l in links if l['src'] == prev_t and l['tgt'] == curr_t),
                None
            )
            if edge is not None:
                gap = tracklet_info[curr_t]['first_frame'] - tracklet_info[prev_t]['last_frame']
                edge['time_diff'] = gap
                chain_edges.append(edge)

        confs = [e['confidence'] for e in chain_edges]
        gaps = [e['time_diff'] for e in chain_edges]

        avg_conf = np.mean(confs) if confs else 0
        min_conf = np.min(confs) if confs else 0
        max_gap = np.max(gaps) if gaps else 0

        print(
            f"{chain_id:<10} {base_id:<10} {orig_ids_str:<18} {len(valid_tracklets):<9} "
            f"{first_ts:<12} {last_ts:<12} {duration:>6.1f}s   "
            f"{len(chain_edges):<6} {avg_conf:<8.2f} {min_conf:<8.2f} {max_gap}"
        )

        for j, tid in enumerate(valid_tracklets):
            info = tracklet_info[tid]
            f1 = info['first_frame']
            f2 = info['last_frame']
            dets = info['detections']

            detail = ""
            if j > 0:
                prev_t = valid_tracklets[j - 1]
                edge = next(
                    (
                        l for l in links
                        if l['src'] == prev_t and l['tgt'] == tid
                    ),
                    None
                )
                if edge is not None:
                    detail = (
                        f"  <- conf: {edge['confidence']:.2f}, gap: {edge['time_diff']}"
                    )

            print(
                f"    {tid:<12} frames {f1}-{f2} ({dets} detections){detail}"
            )
        print()

feature_cols = ['traj_sqrt', 'velocity_error', 'speed', 'time_diff', 'pix_dist', 'box_area_ratio', 'area_growth_diff', 'heading_diff', 'aspect_ratio_diff']

"""
#MODEL TRAINING SECTION
df_parking_split = split_into_tracklets(df1)
df_airport_split = split_into_tracklets(df2)

parking_tc = map_chains_to_tracklets(parking_chains, df_parking_split)
airport_tc = map_chains_to_tracklets(airport_chains, df_airport_split)
parking_ns_t = {(f"{a}_0", f"{b}_0") for (a, b) in parking_lot_not_same}

ml_data_parking, tracks_parking = generate_candidate_pairs(df1, PARKING_VIDEO_LENGTH)
ml_data_airport, tracks_airport = generate_candidate_pairs(df2, AIRPORT_VIDEO_LENGTH)

df_parking = pd.DataFrame(ml_data_parking)
df_airport = pd.DataFrame(ml_data_airport)
print(f"\nParking pairs before labeling: {len(df_parking)}")
print(f"Airport pairs before labeling: {len(df_airport)}")

df_parking['y'] = df_parking.apply(
    lambda r: is_same_vehicle(r['id1'], r['id2'], parking_tc, parking_ns_t, parking_isolated), axis=1)
df_airport['y'] = df_airport.apply(
    lambda r: is_same_vehicle(r['id1'], r['id2'], airport_tc, set(), airport_isolated), axis=1)

print(f"\nParking None: {df_parking['y'].isna().sum()}")
print(f"Airport None: {df_airport['y'].isna().sum()}")

df_parking['source'] = 'parking_lot'
df_airport['source'] = 'airport'
df_combined = pd.concat([df_parking, df_airport], ignore_index=True)
df_train = df_combined[df_combined['y'].notna()].copy()
df_train['y'] = df_train['y'].astype(int)

#RETRAIN FOREST IF NEEDED
X = df_train[feature_cols]
y = df_train['y']

param_grid = {
    'n_estimators': [200, 300],          
    'max_depth': [5, 10, 15],          
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [
        'balanced',
        {0: 1, 1: 10},    # manually match the ratio
        {0: 1, 1: 15},    # slightly over-weight positives
    ] 
}

#print(f"Positive pairs: {(y == 1).sum()}")
#print(f"Negative pairs: {(y == 0).sum()}")
#print(f"Ratio: 1:{(y == 0).sum() / max((y == 1).sum(), 1):.1f}")

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, 
                           cv=5, scoring='f1', n_jobs=-1, verbose=1)

print("Starting Grid Search...")
grid_search.fit(X, y)

best_rf = grid_search.best_estimator_
y_probs_cv = cross_val_predict(best_rf, X, y, cv=5, method='predict_proba')[:, 1]
print("\n--- Tuned Random Forest Threshold Sweep (CV Out-of-Fold) ---")
print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 50)

thresholds = np.arange(0.30, 0.81, 0.05)

for thresh in thresholds:
    y_pred_thresh = (y_probs_cv >= thresh).astype(int)
    prec = precision_score(y, y_pred_thresh, zero_division=0)
    rec = recall_score(y, y_pred_thresh, zero_division=0)
    f1 = f1_score(y, y_pred_thresh, zero_division=0)
    print(f"{thresh:<10.2f} | {prec:<10.2f} | {rec:<10.2f} | {f1:<10.2f}")

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [feature_cols[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances for Vehicle Linking")
plt.bar(range(len(importances)), importances[indices], align="center", color='teal')
plt.xticks(range(len(importances)), sorted_features, rotation=45)
plt.ylabel("Relative Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("Plot saved successfully as 'feature_importance.png' in your current directory!")

joblib.dump({
    'model': best_rf,
    'threshold_parking': 0.40,
    'threshold_airport': 0.60,
    'feature_cols': feature_cols,
}, "rf_relink_new_best.pkl")
"""

bundle = joblib.load("rf_relink_new_best.pkl")
best_rf = bundle['model']
feature_cols = bundle['feature_cols']

scene_type = 'airport'  # or 'airport'
threshold = bundle[f'threshold_{scene_type}']

raw_df = pd.read_csv("better_tests/tracksid3duplicate.csv")
ml_data, tracks = generate_candidate_pairs(raw_df, 123)
df = pd.DataFrame(ml_data)

X = df[feature_cols]
df['confidence'] = best_rf.predict_proba(X)[:, 1]
df['accept'] = (df['confidence'] >= 0.50).astype(int)

accepted = df[df['accept'] == 1].sort_values('confidence', ascending=False)
print(f"Total pairs: {len(df)}")
print(f"Accepted: {df['accept'].sum()}")
chains, links = build_relinked_chains(df, raw_df, threshold=threshold, max_gap=6)

links_dicts = [
    {'src': l[0], 'tgt': l[1], 'confidence': l[2]}
    for l in links
]

summarize_chains(chains, links_dicts, raw_df, 123, max_gap=6)

#df_repaired = repair_csv(raw_df, chains, max_gap=6)
#df_repaired.to_csv("repaired_tracking.csv", index=False)