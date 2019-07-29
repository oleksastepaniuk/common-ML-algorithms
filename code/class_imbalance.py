# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 03:35:30 2018

@author: Oleksa
"""

import numpy as np
import pandas as pd

def test_indices(data_length, start, end):
    '''Calculate indices of all rows in the test set given start and end indicators'''
    
    start_ind = int(data_length * start)
    end_ind   = int(data_length * end)
    
    test_indices = np.arange(start_ind, end_ind+1, 1)
    
    return test_indices

# # test function
# for i in range(9): print(test_indices(100, start[i], end[i]))



def create_sets(pos_length, neg_length, start, end, positive, negative):
    '''Creates train and test subsets using the indices from the test_indices function'''
    
    pos_indices = test_indices(pos_length, start, end)
    pos_test    = positive[positive['index'].isin(pos_indices)]
    pos_train   = positive[~positive['index'].isin(pos_indices)]
    
    neg_indices = test_indices(neg_length, start, end)
    neg_test    = negative[negative['index'].isin(neg_indices)]
    neg_train   = negative[~negative['index'].isin(neg_indices)]
    
    # remove index column it is no longer needed
    pos_test  = pos_test.drop(['index'], axis=1)
    pos_train = pos_train.drop(['index'], axis=1)
    neg_test  = neg_test.drop(['index'], axis=1)
    neg_train = neg_train.drop(['index'], axis=1)
    
    return pos_test, pos_train, neg_test, neg_train



def undersample(pos_test, pos_train, neg_test, neg_train):
    '''Performs undersampling and returns unified train and test sets'''
    
    np.random.seed(10) # set a seed so that the results are consistent
    
    num_elements  = pos_train.shape[0]
    neg_train_new = neg_train.sample(n=num_elements)
    
    train = pd.concat([pos_train, neg_train_new])
    test  = pd.concat([pos_test,  neg_test])
    
    return train, test


def oversample(pos_test, pos_train, neg_test, neg_train):
    '''Performs oversampling and returns unified train and test sets'''
    
    np.random.seed(10) # set a seed so that the results are consistent
    
    num_elements  = neg_train.shape[0]
    pos_train_new = pos_train.sample(n=num_elements, replace=True)
    
    train = pd.concat([pos_train_new, neg_train])
    test  = pd.concat([pos_test,  neg_test])
    
    return train, test


def intel_sample(pos_train):
    '''Creates new positive train examples and returns unified train and test sets'''
    
    # dataframe that will store new results
    pos_new =  pd.DataFrame(columns=['class_positive', 'sex_I', 'sex_M', 'length', 'diameter', 'height',
                                    'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight'])
    
    # create middle point between each pair of pos_train observations
    for i in range(pos_train.shape[0]):
        
        p_1 = pos_train.iloc[i,:]
    
        for j in range(i+1, pos_train.shape[0]):
            
            p_2 = pos_train.iloc[j,:]
            
            point_new = pd.DataFrame({'class_positive': [1],
                                      'sex_I': [np.random.choice([p_1['sex_I'], p_2['sex_I']])],
                                      'sex_M': [np.random.choice([p_1['sex_M'], p_2['sex_M']])],
                                      'length':          [(p_1['length']+p_2['length'])/2],
                                      'diameter':        [(p_1['diameter']+p_2['diameter'])/2],
                                      'height':          [(p_1['height']+p_2['height'])/2],
                                      'whole_weight':    [(p_1['whole_weight']+p_2['whole_weight'])/2],
                                      'shucked_weight':  [(p_1['shucked_weight']+p_2['shucked_weight'])/2],
                                      'viscera_weight':  [(p_1['viscera_weight']+p_2['viscera_weight'])/2],
                                      'shell_weight':    [(p_1['shell_weight']+p_2['shell_weight'])/2]
                                      })
            
            # add new point to the set of new points
            pos_new = pd.concat([pos_new, point_new])
            
    pos_train_new = pd.concat([pos_train,  pos_new])
    
    return pos_train_new