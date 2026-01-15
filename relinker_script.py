import csv
import pandas as pd

CSV_PATH = "better_tests/tracksid3.csv"

df = pd.read_csv(CSV_PATH)
df['cx'] = (df['x1'] + df['x2']) / 2
df['cy'] = (df['y1'] + df['y2']) / 2

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
t = tracks[138]
print(t['first_frame'], t['last_frame'], t['first_position'], t['last_position'])
    
"""
In this specific example, we wish to relink a vehicle which suddenly changes ID to be another vehicle entirely (new id)

STRATEGY: 
1) Track velocity from the past few frames through looking through the list of positions for that vehicle in our dictionary
2) After the SUV's last detection, loop through all other tracks within close-by frames (TUNABLE parameter)
3) Predict A's position using its final position & velocity (velocity is in pixel movement per frame * frames = "pixel movement")
4) Find how far B is from the prediction (TUNABLE)
5) Then sort the candidates by distance to find the closest one if any 
"""