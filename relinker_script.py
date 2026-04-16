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
from sklearn.model_selection import LeaveOneOut

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

parking_vehicles = [
    {5, 126, 133, 138, 252, 273},                          # Black SUV
    {11, 160, 249, 281, 282, 294, 296, 298, 302},          # White SUV
    {56, 66, 68, 71, 85, 108, 111},                        # Vehicle A
    {4, 263, 287},                                           # Stationary
    {1}, {3}, {6}, {7}, {8}, {9},                           # Isolated
]

parking_lot_not_same = {(4, 238), (238, 4)}

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

def label_from_vehicle_groups(df_pairs, vehicle_groups, df_split, not_same=None):
    tracklet_to_vehicle = {}
    for v_idx, base_ids in enumerate(vehicle_groups):
        tids = df_split[df_split['track_id'].isin(base_ids)]['tracklet_id'].unique()
        for tid in tids:
            tracklet_to_vehicle[tid] = v_idx

    # Build not_same tracklet pairs if provided
    not_same_tracklets = set()
    if not_same:
        for id_a, id_b in not_same:
            tids_a = df_split[df_split['track_id']==id_a]['tracklet_id'].unique()
            tids_b = df_split[df_split['track_id']==id_b]['tracklet_id'].unique()
            for ta in tids_a:
                for tb in tids_b:
                    not_same_tracklets.add((ta, tb))
                    not_same_tracklets.add((tb, ta))

    labels = []
    for _, row in df_pairs.iterrows():
        id1, id2 = row['id1'], row['id2']
        if (id1, id2) in not_same_tracklets:
            labels.append(0)
        else:
            v1 = tracklet_to_vehicle.get(id1)
            v2 = tracklet_to_vehicle.get(id2)
            if v1 is None or v2 is None:
                labels.append(None)
            elif v1 == v2:
                labels.append(1)
            else:
                labels.append(0)

    df_pairs['y'] = labels
    return df_pairs

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

def generate_relink_report(df_predictions, raw_df, video_length, max_gap=6,
                           auto_accept_threshold=0.90,
                           review_threshold=0.70,
                           include_low_conf=False):
    #Generate a chronological analyst-facing CSV from RF pairwise predictions.
    #No graph logic — just ranked, triage-labeled pairs sorted by time.

    df_split = split_into_tracklets(raw_df, max_gap=max_gap)
    fps = (raw_df['frame'].max() - raw_df['frame'].min()) / video_length
 
    # Build tracklet info lookup
    tracklet_info = {}
    for tid, grp in df_split.groupby('tracklet_id'):
        tracklet_info[tid] = {
            'first_frame': int(grp['frame'].min()),
            'last_frame': int(grp['frame'].max()),
        }
 
    # Filter by minimum threshold
    min_threshold = review_threshold if not include_low_conf else 0.0
    df_filtered = df_predictions[df_predictions['confidence'] >= min_threshold].copy()
 
    rows = []
    for _, row in df_filtered.iterrows():
        id_lost = row['id1']
        id_gained = row['id2']
        conf = row['confidence']
 
        info_lost = tracklet_info.get(id_lost)
        info_gained = tracklet_info.get(id_gained)
        if info_lost is None or info_gained is None:
            continue
 
        frame_lost = info_lost['last_frame']
        frame_gained = info_gained['first_frame']
        gap_frames = frame_gained - frame_lost
        gap_seconds = round(gap_frames / fps, 2)
 
        time_lost_sec = frame_lost / fps
        time_gained_sec = frame_gained / fps
 
        def fmt_time(sec):
            return f"{int(sec // 60)}:{sec % 60:05.2f}"
 
        # Triage label
        if conf >= auto_accept_threshold:
            action = 'auto_accept'
        elif conf >= review_threshold:
            action = 'review'
        else:
            action = 'low_confidence'
 
        rows.append({
            '_sort_frame': frame_lost,
            'id_lost': id_lost,
            'last_seen': fmt_time(time_lost_sec),
            'id_gained': id_gained,
            'first_seen': fmt_time(time_gained_sec),
            'confidence': round(conf, 3),
            'action': action,
            'gap_seconds': gap_seconds,
        })
 
    report = pd.DataFrame(rows)
 
    # Keep only the best (highest confidence) match per id_lost
    report = report.sort_values('confidence', ascending=False)
    report = report.drop_duplicates(subset='id_lost', keep='first')
 
    # Sort chronologically
    report = report.sort_values('_sort_frame').drop(
        columns='_sort_frame'
    ).reset_index(drop=True)
 
    # Final column order
    cols = ['id_lost', 'last_seen', 'id_gained', 'first_seen', 'confidence', 'action', 'gap_seconds']
    report = report[cols]
 
    # Print summary
    total = len(report)
    auto = (report['action'] == 'auto_accept').sum()
    review = (report['action'] == 'review').sum()
    low = (report['action'] == 'low_confidence').sum()
    print(f"Relink Report: {total} suggested associations")
    print(f"  auto_accept (>={auto_accept_threshold}): {auto}")
    print(f"  review ({review_threshold}-{auto_accept_threshold}): {review}")
    if include_low_conf:
        print(f"  low_confidence (<{review_threshold}): {low}")
 
    return report

feature_cols = ['traj_sqrt', 'velocity_error', 'speed', 'time_diff', 'pix_dist', 'box_area_ratio', 'area_growth_diff', 'heading_diff', 'aspect_ratio_diff']

#MODEL TRAINING SECTION
df_parking_split = split_into_tracklets(df1)
df_airport_split = split_into_tracklets(df2)

airport_tc = map_chains_to_tracklets(airport_chains, df_airport_split)

ml_data_parking, tracks_parking = generate_candidate_pairs(df1, PARKING_VIDEO_LENGTH)
ml_data_airport, tracks_airport = generate_candidate_pairs(df2, AIRPORT_VIDEO_LENGTH)

df_parking = pd.DataFrame(ml_data_parking)
df_airport = pd.DataFrame(ml_data_airport)
print(f"\nParking pairs before labeling: {len(df_parking)}")
print(f"Airport pairs before labeling: {len(df_airport)}")

df_parking = label_from_vehicle_groups(df_parking, parking_vehicles, df_parking_split, not_same=parking_lot_not_same)

df_airport['y'] = df_airport.apply(
    lambda r: is_same_vehicle(r['id1'], r['id2'], airport_tc, set(), airport_isolated), axis=1)

print(f"\nParking None: {df_parking['y'].isna().sum()}")
print(f"Airport None: {df_airport['y'].isna().sum()}")

print("Parking pos:", df_parking[df_parking['y']==1].shape[0])
print("Parking neg:", df_parking[df_parking['y']==0].shape[0])
print("Airport pos:", df_airport[df_airport['y']==1].shape[0])
print("Airport neg:", df_airport[df_airport['y']==0].shape[0])

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

# Match poster font: Latin Modern Sans (lmodern + \sfdefault) ≈ cmss10 in matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['cmss10', 'DejaVu Sans'],
    'mathtext.fontset': 'cm',
})

readable_feature_names = {
    'pix_dist': 'Pixel Distance\n(Last to First Position)',
    'speed': 'Apparent Speed\n(Distance / Frame Gap)',
    'traj_sqrt': 'Trajectory Prediction\nError (Normalized)',
    'box_area_ratio': 'Bounding Box\nArea Ratio',
    'aspect_ratio_diff': 'Aspect Ratio\nDifference',
    'velocity_error': 'Kalman Velocity\nError',
    'time_diff': 'Frame Gap\n(Time Between Sightings)',
    'area_growth_diff': 'Bounding Box\nGrowth Rate Diff',
    'heading_diff': 'Heading Angle\nDifference',
}
sorted_readable = [readable_feature_names[f] for f in sorted_features]
sorted_importances = importances[indices]

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(sorted_importances)))
bars = ax.barh(range(len(sorted_importances)-1, -1, -1), sorted_importances,
               color=colors, edgecolor='white', linewidth=0.5, height=0.7)
for bar, val in zip(bars, sorted_importances):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center', ha='left', fontsize=12, fontweight='bold', color='#333333')
ax.set_yticks(range(len(sorted_readable)-1, -1, -1))
ax.set_yticklabels(sorted_readable, fontsize=12)
ax.set_xlabel('Relative Importance', fontsize=14, fontweight='bold')
ax.set_title('Random Forest Feature Importances\nfor Vehicle Re-Identification Linking',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlim(0, max(sorted_importances) * 1.22)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight', facecolor='white')
print("Plot saved successfully as 'feature_importance.png' in your current directory!")

joblib.dump({
    'model': best_rf,
    'threshold_parking': 0.40,
    'threshold_airport': 0.60,
    'feature_cols': feature_cols,
}, "relink_LOOCV.pkl")

"""
bundle = joblib.load("rf_relink_test.pkl")
best_rf = bundle['model']
feature_cols = bundle['feature_cols']

scene_type = 'airport'  # or 'airport'
threshold = bundle[f'threshold_{scene_type}']

# PARKING LOT DATA ONLY
raw_df = pd.read_csv("better_tests/tracksid3duplicate.csv")
ml_data, tracks = generate_candidate_pairs(raw_df, 123)
df = pd.DataFrame(ml_data)

X = df[feature_cols]
df['confidence'] = best_rf.predict_proba(X)[:, 1]
print(f"Total candidate pairs: {len(df)}")

report = generate_relink_report(
    df, raw_df,
    video_length=123,
    max_gap=6,
    auto_accept_threshold=0.90,
    review_threshold=0.70,
    include_low_conf=True,   # set False to hide <0.70
)

report.to_csv("relink_report.csv", index=False)
print(f"\nSaved to relink_report.csv")
print(report.head(20).to_string(index=False))

# Test all data using part before the random forest retrain
for source_name, df_source in [("Parking", df_parking), ("Airport", df_airport)]:
    df_labeled = df_source[df_source['y'].notna()].copy()
    X = df_labeled[feature_cols]
    y_true = df_labeled['y'].astype(int)
    y_probs = best_rf.predict_proba(X)[:, 1]

    print(f"\n=== {source_name} ({(y_true==1).sum()} pos, {(y_true==0).sum()} neg) ===")
    for thresh in [0.40, 0.50, 0.60, 0.70]:
        y_pred = (y_probs >= thresh).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        print(f"  θ={thresh:.2f}  P={p:.2f}  R={r:.2f}  F1={f:.2f}")
"""