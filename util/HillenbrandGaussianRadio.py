import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import torch
import parselmouth
import matplotlib.pyplot as plt
from functools import lru_cache
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
vowels_num = 12
np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)

filename = "/home/u220110626/HLHTSAT/data/hillenbrand-vowel-formatted.csv"
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
data = []
with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        data.append([
            map[row[1][-2:]],   # class
            int(row[4]),        # f1
            int(row[5]),
            int(row[6])         # f3
        ])
        
data = np.array(data)

data = data[~np.any(data == 0, axis = 1)]

data_radio = []

for row in data:
    data_radio.append([
        row[0],
        row[1]/row[2],
        row[1]/row[3],
        row[2]/row[3]
    ])
data_radio = np.array(data_radio)
print(data_radio)

data = data_radio

Y = data[:,0]
X = data[:,1:]


gmm = GaussianMixture(n_components=12)
gmm.fit(X)

labels = gmm.predict(X)
ari = adjusted_rand_score(Y, labels)
nmi = normalized_mutual_info_score(Y, labels)
print(ari)
print(nmi)
 

"""

Y = result[:,0]
X = result[:,1:]

clf = GaussianNB()
clf.fit(X, Y)
"""
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
            # formants.get_value_at_time(3, time_point),
            # formants.get_value_at_time(4, time_point),
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
