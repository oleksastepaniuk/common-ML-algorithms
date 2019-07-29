# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:37:24 2018

@author: Oleksa
Note: code implemented during the Lab 4 of the Data Mining Course
"""

import numpy as np
from scipy import linalg



def pca(data):
    '''Calculates and returns principal components'''
    
    cov = np.cov(data.T)
    eigVals, eigVectors = linalg.eig(cov)
    order = np.flip(np.argsort(eigVals), 0)

    eigVals    = eigVals[order] 
    eigVectors = eigVectors[:,order] * -1

    k = data.shape[1]
    projectionMatrix = eigVectors[: ,0:k]
    pcaByHandData    = data.dot(projectionMatrix)
    mean_norm        = np.mean(pcaByHandData, 0)
    pcaByHandData    = pcaByHandData - mean_norm
    
    return pcaByHandData