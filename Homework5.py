#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:40:33 2019

@author: Tenkichi-MAC
"""
import warnings
warnings.filterwarnings("ignore")

# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn import datasets
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import linalg
from sklearn.cluster import KMeans

###########################################################################
# Problem 1
###########################################################################

M = np.genfromtxt('./usps.test', missing_values=0, skip_header=0, delimiter=',', dtype=float)
ytst = M[:, 0]
Xtst = M[:, 1:]
Xtst_centered = Xtst - np.mean(Xtst, axis=0)

M = np.genfromtxt('./usps.valid', missing_values=0, skip_header=0, delimiter=',', dtype=float)
yval = M[:, 0]
Xval = M[:, 1:]
Xval_centered = Xval - np.mean(Xval, axis=0)

M = np.genfromtxt('./usps.train', missing_values=0, skip_header=0, delimiter=',' , dtype=float)
ytrn = M[:, 0]
Xtrn = M[:, 1:]
Xtrn_centered = Xtrn -  np.mean(Xtrn, axis=0)

U, s, V = linalg.svd(Xtrn_centered)
E = np.square(s)
eigendigits = dict()
for index in range(0, 16):
    eigendigits[index] = V[index,].reshape(16, 16)
#    eigendigits[index] = plt.matshow(V[index,].reshape(16, 16))
#    plt.show()

cum_sum = np.cumsum(E);
cum_percent = cum_sum / cum_sum[-1]

k = dict()
X_proj = dict()
for index in [70, 80, 90, 100]:
    k[index] = np.shape(np.where(cum_percent <= index/100))[1]
    x_proj = list()
    for x in Xtrn:
        x_proj.append(np.matmul(V[:k[index], :], x).tolist())
    X_proj[index] = np.array(x_proj)

#X_proj[100] = Xtrn
#k[100] = len(cum_percent)
models = dict()
err_val = dict()
err_tst = dict()
for index in [70, 80, 90, 100]:
    for a in [0.0001, 0.001, 0.01, 0.1]:
        algo = linear_model.SGDClassifier(loss="hinge", penalty="l2", alpha=a)
        models[(index, a)] = algo.fit(X_proj[index], ytrn)
        
        xval_proj = list()
        for x in Xval:
            xval_proj.append(np.matmul(V[:k[index], :], x).tolist())
        xval_proj = np.array(xval_proj)
        predictions = models[(index, a)].predict(xval_proj)
        err_val[(index, a)] = 1 - metrics.accuracy_score(yval, predictions)

        
        xtst_proj = list()
        for x in Xtst:
            xtst_proj.append(np.matmul(V[:k[index], :], x).tolist())
        xtst_proj = np.array(xtst_proj)
        predictions = models[(index, a)].predict(xtst_proj)
        err_tst[(index, a)] = 1 - metrics.accuracy_score(ytst, predictions)


###########################################################################
# Problem 2
###########################################################################
# Geneate dataset
x, y = datasets.make_circles(n_samples = 1500 , factor =.5, noise =.05)

def spectral_clustering(x, k=2, gamma=1):
    condensed_distances = pdist(x)
    distances = squareform(condensed_distances)
    A = np.exp(-gamma * distances)
    D = np.diagflat(np.apply_along_axis(np.sum, 0, A))
    L = D - A
    w, v = linalg.eigh(L, eigvals = (0, k-1))
#    E = np.square(s)
    algo = KMeans(k)
    model = algo.fit(v)
    predictions = model.predict(v)
    return predictions

my_predictions = spectral_clustering(x, 2, 20)
sk_predictions = KMeans(2).fit(x).predict(x)

#LABEL_COLOR_MAP = {0 : 'r', 1 : 'k'}
#label_color = [LABEL_COLOR_MAP[l] for l in my_predictions]
#fig1 = plt.figure()
#fig1 = plt.scatter(x[:, 0], x[:, 1], c=label_color)
#
#label_color = [LABEL_COLOR_MAP[l] for l in sk_predictions]
#fig2 = plt.figure()
#fig2 = plt.scatter(x[:, 0], x[:, 1], c=label_color)

import cv2
img = cv2.imread('seg.jpg', 0)
#cv2.imshow('image', img)
#fig3 = plt.figure()
#plt.imshow(img)

plane_predictions = spectral_clustering(img.flatten().reshape(len(img.flatten()), 1), 2, gamma=0.1)
#fig4 = plt.figure()
#fig4 = plt.imshow(plane_predictions.reshape(81, 121))






    


