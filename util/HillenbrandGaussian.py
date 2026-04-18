import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import torch
import parselmouth
import matplotlib.pyplot as plt
from functools import lru_cache
import pandas as pd
import csv
vowels_num = 12
np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)

filename = "../data/hillenbrand-vowel-formatted.csv"
map = {
    "ae":1,
    "ah":2,
    "aw":3,
    "eh":4,
    "ei":5,
    "er":6,
    "ih":7,
    "iy":8,
    "oa":9,
    "oo":10,
    "uh":11,
    "uw":12
}
result = []
with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        result.append([
            map[row[1][-2:]],   # class
            int(row[3]),        # f0
            int(row[4]),        # f1
            int(row[5]),        # f2
            int(row[6]),        # f3
            int(row[7])         # f4
        ])
        
result = np.array(result)
# print(len(result))
Y = result[:,0]
X = result[:,1:]

clf = GaussianNB()
clf.fit(X, Y)

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
            formants.get_value_at_time(4, time_point),
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
