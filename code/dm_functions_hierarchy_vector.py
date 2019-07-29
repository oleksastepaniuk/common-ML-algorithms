# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 06:29:18 2018

@author: Oleksa
"""
import pandas as pd
import numpy as np # delete in the end
import os
from tqdm import tqdm

# set working directory
os.chdir('./Desktop/Corsework_32341885')


def cluster_dist_group(x_1, y_1, x_2, y_2):
    '''Average linkage clustering distance between two groups of points'''
    
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    
    y_1 = np.array(y_1)
    y_2 = np.array(y_2)    
    
    len_1 = len(x_1)
    len_2 = len(x_2)

    x_1 = np.array([x_1,]*len_2)
    x_2 = np.array([x_2,]*len_1).transpose()
    
    y_1 = np.array([y_1,]*len_2)
    y_2 = np.array([y_2,]*len_1).transpose()

    result = np.mean(np.sqrt((x_1-x_2)**2 + (y_1-y_2)**2))
    
    return result
            


# small 5 points datset
df_dummy = np.transpose(np.array([[1, 2.5, 2, 2, 4, 4],[2, 4.5, 2, 2, 1.5, 2.5]]))
df = pd.DataFrame(df_dummy)
del(df_dummy)

# dataset of ponts from Coursework
df = pd.read_csv('./data/coordinates_pandas.csv')
df = df[['east_stand', 'north_stand']]
df = df[0:100]




########### PART 1. Prepare datasets used in iterations

####### Create a dataset that stores coordinates of points ang their group

df_groups         = pd.DataFrame(df).copy()
df_groups.columns = ['x', 'y']
df_groups         = df_groups.assign(group  = list(df_groups.index))
df_groups         = df_groups.assign(points = list(df_groups.index))

# variables used to guide iterations
groups            = max(df_groups['group'])  # index of the last group
start_groups      = groups                   # index of the last group at the beginning
n_points          = groups + 1               # number of points in the dataset
iteration         = 0                        # iteration index

# store x, y and points as lists of lists
x = []
y = []
points = []

for i in range(groups+1):
    x += [[df_groups['x'].iloc[i]]]
    y += [[df_groups['y'].iloc[i]]]
    points += [[df_groups['points'].iloc[i]]]
   
df_groups['x'] =  x
df_groups['y'] =  y
df_groups['points'] = points

del(x, y, points, i)


#######  Create dataset with a distance between each group
group_1  = []
x_1      = []
y_1      = []  
group_2  = []
x_2      = []
y_2      = []

n_groups = groups + 1
counter  = n_groups
start    = 1

for i in range(counter-1):
    n_groups -= 1
    group_1  += [i] * n_groups
    x_1      += [df_groups['x'].iloc[i]] * n_groups
    y_1      += [df_groups['y'].iloc[i]] * n_groups
    
    for j in range(start, counter):
        group_2 += [j]
        x_2     += [df_groups['x'].iloc[j]]
        y_2     += [df_groups['y'].iloc[j]]
    start += 1
del(n_groups, counter, start, i, j)

df_distance = pd.DataFrame({'group_1' : group_1,
                            'x_1'     : x_1,
                            'y_1'     : y_1,
                            'group_2' : group_2,
                            'x_2'     : x_2,
                            'y_2'     : y_2,
                            'distance': None,
                           })
del(group_1, x_1, y_1, group_2, x_2, y_2)


#######  Create a dataframe to store progress
df_progress =  pd.DataFrame(columns=['iteration', 'group', 'points', 'group_number',
                                     'share_inside'])


    
    
########### PART 2. Iterations
    

# Repeat following code until all points are in one group

while len(df_groups)!=1:

    
    # calculate first round of distances
    distance = df_distance['distance'].copy()
    
    
    for i in tqdm(range(len(distance))):
    
        if distance[i] is None:
            distance[i] = cluster_dist_group(df_distance['x_1'].iloc[i],
                                             df_distance['y_1'].iloc[i],
                                             df_distance['x_2'].iloc[i],
                                             df_distance['y_2'].iloc[i])
            if distance[i] == 0: break  # stop if there are two identical points
            
    df_distance['distance'] = distance
    del(distance, i)

    # what is the minimum distance
    min_dist = min(x for x in df_distance['distance'] if x is not None)

    # what first row has the minimum distance
    min_row = df_distance.index[df_distance['distance'] == min_dist].tolist()[0]

    # number of the new group
    new_group_number  = groups+1
    groups           += 1

    # coordinates of the new group
    new_group_x  = df_distance['x_1'].iloc[min_row] + df_distance['x_2'].iloc[min_row] 
    new_group_y  = df_distance['y_1'].iloc[min_row] + df_distance['y_2'].iloc[min_row]

    # groups that will be united
    old_group_1 = df_distance['group_1'].iloc[min_row]
    old_group_2 = df_distance['group_2'].iloc[min_row]

    # points in the new group
    new_group_points  = df_groups[df_groups['group'] == old_group_1]['points'].values[0] + \
                        df_groups[df_groups['group'] == old_group_2]['points'].values[0]

    # delete old groups from the df_distance
    df_distance = df_distance[(df_distance['group_1'] != old_group_1) & (df_distance['group_1'] != old_group_2)]
    df_distance = df_distance[(df_distance['group_2'] != old_group_1) & (df_distance['group_2'] != old_group_2)]

    # delete old groups from the df_groups
    df_groups = df_groups[(df_groups['group'] != old_group_1) & (df_groups['group'] != old_group_2)]



    # add new group to df_distance
    n_groups = len(df_groups)

    df_distance_upd = pd.DataFrame({'group_1' : [new_group_number] * n_groups,
                                    'x_1'     : [new_group_x]  * n_groups,
                                    'y_1'     : [new_group_y]  * n_groups,
                                    'group_2' : df_groups['group'].copy(),
                                    'x_2'     : df_groups['x'].copy(),
                                    'y_2'     : df_groups['y'].copy(),
                                    'distance': None,
                                    })
    del(n_groups)

    df_distance = pd.concat([df_distance_upd, df_distance]) # new group should be on the top to induce its growth
    df_distance = df_distance.reset_index(drop=True) # reset row names
    del(df_distance_upd)


    # add new group to df_groups
    df_groups_upd = pd.DataFrame({'x'     : [new_group_x],
                                  'y'     : [new_group_y],
                                  'group' : new_group_number,
                                  'points': [new_group_points],
                                  })

    df_groups = pd.concat([df_groups, df_groups_upd])
    df_groups = df_groups.reset_index(drop=True) # reset row names
    del(df_groups_upd)


    # add information about iteration to the df_progress
    iteration += 1

    df_progress_update = df_groups[df_groups['group']>start_groups][['group', 'points']]
    df_progress_update = df_progress_update.assign(iteration  = iteration)
    df_progress_update = df_progress_update.assign(group_number  = len(df_groups))
    df_progress_update = df_progress_update.assign(n_points   = None)

    points_in_group = []
    for i in range(len(df_progress_update)):
        points_in_group += [len(df_progress_update['points'].iloc[i])]
    df_progress_update['n_points'] = points_in_group
    del(points_in_group, i)

    df_progress_update = df_progress_update.assign(share_inside = sum(df_progress_update['n_points'])/n_points*100)

    df_progress = pd.concat([df_progress, df_progress_update], sort=False)
    df_progress = df_progress.reset_index(drop=True) # reset row names
    del(df_progress_update)

    
    









