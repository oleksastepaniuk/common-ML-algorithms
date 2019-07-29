# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:10:16 2018

@author: Oleksa
Note: Vectorised modification of the Kmeans algorithm. Logic is similar to the code provided in Lab 4,
however implementation is quite different
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


#################
def dist(p1, p2, matrix=False):
    '''Returns Euclidian distance between two points, or point and a matrix'''
    
    if matrix==True:
        distance = np.sqrt(np.sum((p1 - p2)**2, axis=1))
        distance = distance.reshape(len(distance), 1) # make sure that distance is a vector
    else:
        distance = np.sqrt(np.sum((p1 - p2)**2))
        
    return distance


################
def min_dist_pos(point, matrix):
    '''Returns smallest distance between a point and any point of the matrix'''
    
    # vector of distances
    matrix_dist = dist(point, matrix, matrix=True)
    
    min_pos = np.argmin(matrix_dist)

    return min_pos
   
     
################
def sum_dist(m1, m2):
    '''Returns sum of distances between corresponding elements of two matrices'''
    return sum(dist(m1, m2, matrix=True))[0]



## Check functions
#p1 = np.array([5, 6])
#p2 = np.array([5, 10])
#matrix   = np.array([[5,6],[7,8],[9,10]])
#matrix_2 = np.array([[6,6],[7,10],[9,10]])
#
#dist(p1, p2)
#min_dist_pos(p1, matrix)
#min_dist_pos(p2, matrix)
#sum_dist(matrix, matrix_2)


################ PREPARE DATA
def initial_centroids(data, k):
    '''Calculates coordinates of initial random centroids'''

    # number of columns
    columns = data.shape[1]

    # Initial centroids
    np.random.seed(17)                # set seed to make results reproducible
    C = np.random.random((k,columns))

    # scale the random numbers
    # in our case not really needed - data is normalised 
    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)
    C = min_values + C*(max_values - min_values)
    
    return C



################ RUN ALGORITHM
def find_kmeans_clusters(data, k, C, clusters=None, dist_centroids=float("inf"), threshold = 0.05, print_progress=False):

    if dist_centroids > threshold:
    
        ######### find which point belongs to which cluster
        # loop goes not through each point but through each centroid
    
        # create matrix where each column has distance of each point to one centroid
        distance_clust = dist(data, C[0], matrix=True)
    
        for i in range(1, k):
            distance_to_next = dist(data, C[i], matrix=True)
            distance_clust   = np.hstack((distance_clust, distance_to_next))
    
        ######### determine which centroid is the closest one
        clusters         = np.zeros((len(data),1))
        clusters_inverse = np.where(clusters==0,1,0)    # we need this to update only points without cluster
    
        for i in range(k):
        
            clusters_check = np.zeros((len(data),1)) + (i+1)
        
            # this loop is skipped on the last iteration, so all points without class at this iteration
            # are assigned to the last class
            for j in range(i+1, k):
                check = (distance_clust[:,i] < distance_clust[:,j]).reshape(len(data),1)
                clusters_check *= check
            
            clusters_check *= clusters_inverse
            clusters += clusters_check
            clusters_inverse = np.where(clusters==0, 1, 0)

        C_old = C.copy()

        # Determine coordinates of new centroids
        for i in range(1, k+1):
        
            indices = np.where(clusters==i, True, False).reshape(1,len(data))[0]
            cluster_points = data[indices]
        
            C[i-1] = np.mean(cluster_points, axis=0)
        del(i, indices, cluster_points)

        dist_centroids = sum_dist(C, C_old)
    
        # Check progress
        if print_progress == True:
            print(round(dist_centroids,4))
        
        return find_kmeans_clusters(data, k, C, clusters, dist_centroids)
    
    else:
        
        # calculate inertia criteria - it will be used as 'goodness of fit'
        inertia = 0
        for i in range(len(data)):
            inertia += dist(data[i,:], C[int(clusters[i][0]-1)])
            
        return clusters, inertia


################ PLOT RESULT   
def plot_clusters(data, k, clusters):
    '''Plot the clusters using UTM coordinates'''
    
    group_names = []

    # divide points into group by cluster
    for i in range(1, k+1):
    
        indices = np.where(clusters==i, True, False).reshape(1,len(data))[0]
        group_names += ["group_" + str(i)]
        locals()[group_names[i-1]] = data[indices] 

    # plot clusters
    plt.figure(figsize=(10, 7))
    cent_count = 0

    for name in group_names:
        plt.plot(locals()[name][:,0], locals()[name][:,1], ".")







###############################################################################

def main():

    ################ LOAD DATA
    
    ## set working directory
    #os.chdir('./Documents/Term_1/Data_Mining/Coursework')

    # dataset of points from Coursework
    df = pd.read_csv('./data/coordinates_pandas.csv')

    df_array = df[['east_norm', 'north_norm']]
    df_array = np.array(df_array)

    k_list = np.arange(2,16,1)
    inertia_list = []
    cluster_list = []
    
    for k in k_list:

        C =  initial_centroids(df_array, k)
        clusters, inertia = find_kmeans_clusters(df_array, k, C)
        cluster_list += [clusters]
        inertia_list += [inertia]
        
    plt.plot(k_list, inertia_list, 'k')
    plt.plot(k_list, inertia_list, 'ro')
    
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion")
    plt.title("Elbow method")

    df_plot = df[['easting', 'northing']]
    df_plot = np.array(df_plot)
    plot_clusters(df_plot, 5, cluster_list[3])
    
    plt.xlabel("Easting coordinates")
    plt.ylabel("Northing coordinates")
    plt.title("Number of clusters: 5")

#    plt.savefig("kmeans_joensuu.pdf")
#    plt.close()
    


# if script is executed, then main() will be executed
if __name__ == '__main__':
	main()