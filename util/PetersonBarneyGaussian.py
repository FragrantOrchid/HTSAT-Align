import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import torch
import parselmouth
import matplotlib.pyplot as plt
from functools import lru_cache
vowels_num = 10
np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)
# PetersonBarney
file_path = '../data/verified_pb.data'
pb = []
with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split()
        if len(columns) >= 8:
            row = [
                int(columns[2]),
                int(columns[4].rstrip('.')),
                int(columns[5].rstrip('.')),
                int(columns[6].rstrip('.')),
                int(columns[7].rstrip('.'))
            ]
            pb.append(row)
            
pb = np.array(pb)
Y = pb[:,0]
X = pb[:,1:]

clf = GaussianNB()
clf.fit(X, Y)
# result = clf.predict_proba(X)

def getMatrix(filename):
    x = []
    sound = parselmouth.Sound(filename)
    pitch = sound.to_pitch()
    formants = sound.to_formant_burg()
    time_points = np.arange(1.0/320, 1.0, 1.0/160)
    for time_point in time_points:
        x.append([
            pitch.get_value_at_time(time_point),
            formants.get_value_at_time(1, time_point),
            formants.get_value_at_time(2, time_point),
            formants.get_value_at_time(3, time_point),
        ])
    x = np.array(x)
    mask = np.isnan(x).any(axis=1)
    x_vowel = x[~mask]
    result = np.zeros((len(time_points),vowels_num))
    if len(x_vowel) == 0:
        return result
    y = clf.predict_proba(x_vowel)
    result[~mask] = y
    return result

"""
filename = "/users/u220110626/SpeedCommandV2/left/32ad5b65_nohash_0.wav"
matrix = getMatrix(filename=filename)
# print(matrix)

print(matrix)
plt.figure(figsize=(8, 6))
plt.imshow(matrix, cmap='viridis', origin='lower')
plt.colorbar(label='Value')
plt.title('2D Heatmap of Array')
plt.xlabel('X Index')
plt.ylabel('Y Index')
plt.savefig("vowels.png")
"""


    

    
        

