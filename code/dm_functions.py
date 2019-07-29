# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 06:29:18 2018

@author: Oleksa
"""
import math

#####     
def earth_distance(point1, point2):
    '''Find distance in km between two points using Harvesine formula'''
    # This is my Python translation of original Javascript code
    # Source: https://www.movable-type.co.uk/scripts/latlong.html
    
    # avoide modifying original points
    point1 = point1[:]
    point2 = point2[:]
    
    radius_earth =  6.3781*10**6 # meters
    
    # convert degrees to radians
    for i in range(2):
        point1[i] = point1[i] * math.pi / 180
        point2[i] = point2[i] * math.pi / 180

    delta_lat  = point1[0] - point2[0]
    delta_long = point1[1] - point2[1]

    a = math.sin(delta_lat/2)**2 + math.cos(point1[0])*math.cos(point2[0])*math.sin(delta_long/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius_earth * c
    
    return d / 1000




########### Group of functions for manipulations with numerical vectors
    
#####  
def vector_min(vector):
    '''Find the smallest element of the vector'''
    min_v = float("inf")
    for i in range(len(vector)):
        
        # if i'th element is not numeric, skip it
        if type(vector[i])==str or type(vector[i])==None:
            continue
        else:
            if vector[i] < min_v:
                min_v = vector[i]
    
    # if by the end of the loop min = Inf, return nothing
    if min_v == float("inf"):
        print("Could not find min value")
        return None
    else:
        return min_v
    
#####      
def vector_max(vector):
    '''Find the largest element of the vector'''
    max = -float("inf")
    for i in range(len(vector)):
        
        # if i'th element is not numeric, skip it
        if type(vector[i])==str or type(vector[i])==None:
            continue
        else:
            if vector[i] > max:
                max = vector[i]
    
    # if by the end of the loop min = Inf, return nothing
    if max == -float("inf"):
        print("Could not find max value")
        return None
    else:
        return max
    
    

#####   
def vector_plus(input_vector, number, minus=False):
    '''Add number to each element of the vector. If minus is True substraction is performed.'''
    
    vector = input_vector[:]
    
    if minus==True:
        number *= -1
        
    length = len(vector)
    
    result = []
    for i in range(length):
        result += [vector[i] + number]
         
    return result


#####     
def vector_mult(input_vector, number, divide=False):
    '''Multiply each element of the vector by number. If divide is True division is performed.'''
    
    vector = input_vector[:]
    
    if divide==True:
        number = 1/number
        
    length = len(vector)
    
    result = []
    for i in range(length):
        result += [vector[i] * number]
        
    return result     


#####      
def vector_mult_vector(first, second, divide=False):
    '''Multiply two vectors elementwise. If divide is True division is performed.'''
    
    vector_1 = first[:]
    vector_2 = second[:]
        
    length = len(vector_1)
    
    if length != len(vector_2):
        print("Error: vectors have different length")
        return
        
    if divide==True:
        vector_2_inverse = []
        for i in range(length):
            vector_2_inverse += [vector_2[i]**(-1)]
        return vector_mult_vector(vector_1, vector_2_inverse)
    
    result = []
    for i in range(length):
        result += [vector_1[i] * vector_2[i]]
        
    return result  



#####    
def vector_mean(input_vector):
    '''Finds mean of a numerical vector'''
    
    vector = input_vector[:]
    
    return sum(vector) / len(vector)
        

        
#####    
def vector_stdv(input_vector):
    '''Finds standard deviation of a numerical vector'''
    
    vector   = input_vector[:]
    vector_2 = vector_mult_vector(vector, vector)
    
    mean     = vector_mean(vector)
    mean_2   = vector_mean(vector_2)
    
    variance = mean_2 - mean**2
    
    return variance**0.5   


#####  
def vector_standard(input_vector):
    '''Standardize a numerical vector'''
    
    vector   = input_vector[:]
    
    mu    = vector_mean(vector)
    sigma = vector_stdv(vector)
    
    result = vector_plus(vector, mu, minus=True)
    result = vector_mult(result, sigma, divide=True)
    
    return result      
        
    
#####  
def vector_normal(input_vector):
    '''Normalize a numerical vector'''
    
    vector   = input_vector[:]
    
    v_max   = vector_max(vector)
    v_min   = vector_min(vector)
    v_range = v_max - v_min
    
    result  = vector_plus(vector, v_min,   minus=True)
    result  = vector_mult(result, v_range, divide=True)
    
    return result 








    
    
    
    
    
    
    

