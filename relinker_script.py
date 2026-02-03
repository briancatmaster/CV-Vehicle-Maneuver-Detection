import csv
import pandas as pd
import math

CSV_PATH = "better_tests/tracksid3.csv"
VIDEO_LENGTH = 123

df = pd.read_csv(CSV_PATH)
df['cx'] = (df['x1'] + df['x2']) / 2
df['cy'] = (df['y1'] + df['y2']) / 2
last_frame = df['frame'].max()

#Creating our dicionary
tracks = {}

#Groups the DataFrame by track_id. (g is the df containing only rows for the specific track)
for track_id, g in df.groupby('track_id'):
    g = g.sort_values('frame')
    
    tracks[track_id] = {
        'frames': g['frame'].tolist(),
        'positions': list(zip(g['cx'], g['cy'])),
        'bboxes': list(zip(g['x1'], g['y1'], g['x2']-g['x1'], g['y2']-g['y1'])),
        'first_frame': g['frame'].iloc[0],
        'last_frame': g['frame'].iloc[-1],
        'first_position': (g['cx'].iloc[0], g['cy'].iloc[0]),
        'last_position': (g['cx'].iloc[-1], g['cy'].iloc[-1]),
    }

#Black SUV
#t = tracks[138]
#print(t['first_frame'], t['last_frame'], t['first_position'], t['last_position'])

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

potential_pairs = []

for track_A, dfA in tracks.items():
    for track_B, dfB in tracks.items():
        if track_A == track_B:
            continue
        else:
            if dfB['first_frame'] <= dfA['last_frame']:
                continue
        
        time_difference = dfB['first_frame'] - dfA['last_frame']

        #We can tune this MAX_FRAME_GAP later
        if time_difference > 100:
            continue
        
        last_A_position = dfA['last_position']
        first_B_position = dfB['first_position']

        pixel_distance = math.sqrt(math.pow(last_A_position[0] - first_B_position[0], 2) + math.pow(last_A_position[1] - first_B_position[1], 2))
        MAX_PIXEL_DIST = max(40 * time_difference / 30, 40)

        #We can tune this MAX_PIXEL_DIST later
        if pixel_distance < MAX_PIXEL_DIST:
            print("We've identifed two ids that may be the same vehicle with time & space logic!")
            print(str(track_A) + " | last seen at " + str(dfA['last_frame']) + " frame; " + str(track_B) + " | last seen at " + str(dfB['first_frame']) + " frame")
            print(frame_to_timestamp(dfA['last_frame']) + "-->" + frame_to_timestamp(dfB['first_frame']) + "\n")
            potential_pairs.append((track_A, track_B))

print(potential_pairs)

#Next step is to implement blocking vehicle detection for each pair (starting with the temporal overlap check)

    
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

Imitation / Adaptive Learning
Consistent Annotator
"""