# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 03:39:48 2018

@author: Oleksa
"""
import numpy as np

def prepare_logit_nn(train, test):
    '''Prepares data for the logistic regression and neural network algorithms'''
    
    # Y values
    Y_train = np.array(train.iloc[:,0]).reshape(train.shape[0],1)
    Y_test  = np.array(test.iloc[:,0]).reshape(test.shape[0],1)

    # X values
    X_train = np.array(train.iloc[:,1:])
    X_test  = np.array(test.iloc[:,1:])
    
    # reshape data for the algorithm
    Y_train = Y_train.reshape(1, Y_train.shape[0])
    Y_test  = Y_test.reshape(1, Y_test.shape[0])
    
    X_train = X_train.reshape(X_train.shape[1], X_train.shape[0])
    X_test  = X_test.reshape(X_test.shape[1], X_test.shape[0])
    

    return Y_train, X_train, Y_test, X_test


def prepare_tree(train, test):
    '''Prepares data for the decision tree algorithm'''
    
    train_tree = np.array(train)
    test_tree  = np.array(test)
    
    return train_tree, test_tree