import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import joblib
import os
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
import parselmouth
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)
filename = "/home/u220110626/HLHTSAT/data/hillenbrand-vowel-formatted.csv"
map = {
    "ae":0,
    "ah":1,
    "aw":2,
    "eh":3,
    "ei":4,
    "er":5,
    "ih":6,
    "iy":7,
    "oa":8,
    "oo":9,
    "uh":10,
    "uw":11
}
data = []
useside = True
if useside == False:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            data.append([
                map[row[1][-2:]],   # class
                int(row[3]),        # f0
                int(row[4]),        # f1
                int(row[5]), # ,        # f2
                int(row[6])        # f3
            ])
elif useside == True:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            data.append([
                map[row[1][-2:]],   # class
                int(row[3]),        # f0
                int(row[4]),        # f1
                int(row[5]), # ,        # f2
                int(row[6])        # f3
            ])
            data.append([
                map[row[1][-2:]],
                int(row[3]),
                int(row[8]),
                int(row[9]),
                int(row[10])
            ])
            data.append([
                map[row[1][-2:]],
                int(row[3]),
                int(row[11]),
                int(row[12]),
                int(row[13])
            ])
            data.append([
                map[row[1][-2:]],
                int(row[3]),
                int(row[14]),
                int(row[15]),
                int(row[16])
            ])
    
data = np.array(data)
data = data[~np.any(data == 0, axis = 1)] # 去零

Y = data[:,0]
X = data[:,1:]
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

gmm = GaussianMixture(n_components=12)
gmm.fit(X_pca)

labels = gmm.predict(X_pca)
ari = adjusted_rand_score(Y, labels)
nmi = normalized_mutual_info_score(Y, labels)
print(ari)
print(nmi)

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
    result = np.zeros((len(time_points),12))
    if len(x_vowel) == 0:
        return result
    y = pca.transform(x_vowel)
    y = gmm.predict_proba(y)
    result[~mask] = y
    return result