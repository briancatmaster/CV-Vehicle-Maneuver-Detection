import csv
import pandas as pd
import numpy as np
import math
from dataclasses import dataclass

CSV_PATH = "better_tests/tracksid3duplicate.csv"
VIDEO_LENGTH = 123

@dataclass
class Vehicle_Pair:
    time_diff: int = 0
    pix_dist: float = 0.0

    def confidence(self) -> float:
        return 1.0 #insert function later f(time_diff, pix_dist, velocity_error, trajectory_smoothness)

df = pd.read_csv(CSV_PATH)
df['cx'] = (df['x1'] + df['x2']) / 2
df['cy'] = (df['y1'] + df['y2']) / 2
last_frame = df['frame'].max()

#Based on 100 frames * 123 seconds / 7390 frames
MAX_FRAME_GAP = 1.6644 * last_frame / VIDEO_LENGTH

#Creating our dicionary
tracks = {}

#Groups the DataFrame by track_id. (g is the df containing only rows for the specific track)
for track_id, g in df.groupby('track_id'):
    g = g.sort_values('frame')
    
    tracks[track_id] = {
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

#Black SUV
#t = tracks[138]
#print(t['first_frame'], t['last_frame'], t['first_position'], t['last_position'])
#print(t['last_velocity'])

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

potential_pairs = {}

#FIRST LENIENT PASS
print("----- First Pass -----")

for track_A, dfA in tracks.items():
    for track_B, dfB in tracks.items():
        if track_A == track_B:
            continue
        else:
            if dfB['first_frame'] <= dfA['last_frame']:
                continue
        
        current_pair = Vehicle_Pair()
        current_pair.time_diff = dfB['first_frame'] - dfA['last_frame']

        """
        We can tune this MAX_FRAME_GAP later
        if current_pair.time_diff > MAX_FRAME_GAP:
            continue
        """

        last_A_position = dfA['last_position']
        first_B_position = dfB['first_position']

        current_pair.pix_dist = math.sqrt(math.pow(last_A_position[0] - first_B_position[0], 2) + math.pow(last_A_position[1] - first_B_position[1], 2))
        MAX_PIXEL_DIST = max(40 * current_pair.time_diff / (MAX_FRAME_GAP * 0.3), 40)

        #We can tune this MAX_PIXEL_DIST later
        #if current_pair.pix_dist < MAX_PIXEL_DIST:
        #    print("We've identifed two ids that may be the same vehicle with time & space logic!")
        #    print(str(track_A) + " | last seen at " + str(dfA['last_frame']) + " frame; " + str(track_B) + " | first seen at " + str(dfB['first_frame']) + " frame")
        #    print(frame_to_timestamp(dfA['last_frame']) + "-->" + frame_to_timestamp(dfB['first_frame']) + "\n")
        potential_pairs[(track_A, track_B)] = current_pair

print(list(potential_pairs.keys()))

#SECOND LENIENT PASS
print("----- Second Pass -----")
#The goal of the 2nd pass is to use our continous variables to have confidence ratios for the pairs
#Maybe drop those that are below a certain threshold!
for (i, j), info in potential_pairs.items():
    dfA = tracks[i]
    dfB = tracks[j]

    xI, yI = dfA["last_position"]
    xJ, yJ = dfB["first_position"]

    #1. Kalman Filter
    vx = dfA['last_velocity'][0]
    vy = dfA['last_velocity'][1]

    #2. Constant Velocity Measure
    #Get last 5 frames
    recent_positions = dfA['positions'][-5:]
    x_positions = [p[0] for p in recent_positions]
    y_positions = [p[1] for p in recent_positions]
    #vx = (x_positions[-1] - x_positions[0]) / (len(recent_positions) - 1)
    #vy = (y_positions[-1] - y_positions[0]) / (len(recent_positions) - 1)
    
    x_pred = xI + (info.time_diff * vx)
    y_pred = yI + (info.time_diff * vy)

    trajectory_error = np.sqrt(
        (x_pred - xJ)**2 +
        (y_pred - yJ)**2
    )

    traj_sqrt = trajectory_error / math.sqrt(info.time_diff)

    #3. Direct Velocity Comparison
    velocityB = dfB['first_velocity']

    velocity_error = np.sqrt(
        (vx - velocityB[0])**2 + 
        (vy - velocityB[1])**2
    )

    km_error = np.sqrt(
        (vx - velocityB[0])**2 + 
        (vy - velocityB[1])**2
    )

    #Weights with regression, grid search, and other analysis
    b = -0.3911211825579939
    w_traj_sqrt = -0.076074
    w_vel = 1.122969
    w_spd = 0.124254

    logit = (
        b
        + w_traj_sqrt * traj_sqrt
        + w_vel * velocity_error
        + w_spd * (info.pix_dist / info.time_diff)
    )

    prob_same = 1.0 / (1.0 + math.exp(-logit))
    THRESH = 0.3523
    accept = (prob_same >= THRESH)

    #Worse means better
    #score = velocity_error + consider frame_gap stuff
    print(i, j, info.time_diff, info.pix_dist, traj_sqrt, velocity_error, prob_same, accept)
#print(list(potential_pairs.keys()))

"""
In this specific example, we wish to relink a vehicle which suddenly changes ID to be another vehicle entirely (new id)

STRATEGY: 
1) Track velocity from the past few frames through looking through the list of positions for that vehicle in our dictionary
2) After the SUV's last detection, loop through all other tracks within close-by frames (TUNABLE parameter)
3) Predict A's position using its final position & velocity (velocity is in pixel movement per frame * frames = "pixel movement")
4) Find how far B is from the prediction (TUNABLE)
5) Then sort the candidates by distance to find the closest one if any 

When considering candidate B (make sure the time does not overlap - strictly seperate)
Check spatial conditions (x, y, z boxes)
Check that a blocking vehicle is both overlapping with A & B candidates

Step 1: Have a dictionary with a pair of ids that are likely to be capturing the same true vehicle
Step 2: Make a list to track the blocking vehicle (refer to paper)
Step 3: Calculate some kind of f-score for each candidate blocking vehicle

Potentially scrappped: Next step is to implement blocking vehicle detection for each pair (starting with the temporal overlap check)

Imitation / Adaptive Learning
Consistent Annotator
"""