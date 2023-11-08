import numpy as np
import os
import pickle
if os.path.exists('pocket_score.pkl'):
    with open('pocket_score.pkl', 'rb') as f:
        pocket_score = pickle.load(f)
else:
    pocket_score={}

with open('PocketMatch_score.txt', 'r') as f:
    for line in f.read().splitlines():
        pocket = line[:32].strip()
        score = float(line[75:75+8])
        if pocket in pocket_score:
            pocket_score[pocket].append(score)
        else:
            pocket_score[pocket]=[score]

file = open('pocket_score.pkl', 'wb')
pickle.dump(pocket_score, file)
file.close()