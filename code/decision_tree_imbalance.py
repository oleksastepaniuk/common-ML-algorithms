# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 04:42:14 2018

@author: Oleksa
Note: variation of the decision tree for the imbalanced data
Note 2: instead of relying solely on Gini index, we select partition that provides the best 'accuracy'
Note 3: where accuracy is the share of postivive class in the node content
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:36:46 2018

@author: Oleksa
"""

# Package imports
import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, content, left=None, right=None, level=0, id="c", rules=[], print_accuracy=True):
        '''Create a new Tree with content, lef and right '''
        self.content = np.array(content) # assure that content is in the numpy array format
        
        self.part_column = None
        self.part_midpoint = None
        self.terminal = False
        self.accuracy = None
        self.gini = None
        
        self.left = left
        self.right = right
        
        self.level = level
        self.id    = id
        self.rules = rules
        self.print_accuracy=print_accuracy
        
    def __find_midpoints(self, n_col):
        '''Finds midpoints of a continuous variable'''
        # sort column of interest
        column = self.content[self.content[:,n_col].argsort()][:, n_col]
        column = np.unique(column) 
        return column[0:-1] + (column[1:] - column[0:-1]) / 2
    
    def __gini_index_son(self, vector):
        '''Calculate Gini index for a given vector of 0 and 1 of the son (left or right nod)'''
        v_length = len(vector)
        v_ones   = np.sum(vector)
        
        return 1 - (v_ones/v_length)**2 - ((v_length-v_ones)/v_length)**2
    
    def __gini_index(self, midpoint, n_col, n_target):
        '''Calculate total Gini index'''
        v_left  = self.content[self.content[:, n_col] <= midpoint][:,n_target]
        v_right = self.content[self.content[:, n_col] >  midpoint][:,n_target]
        
        l_total = len(self.content)
        l_left  = len(v_left)
        
        gini = ((l_left/l_total)*self.__gini_index_son(v_left) + 
               ((l_total-l_left)/l_total)*self.__gini_index_son(v_right))

        return gini
    
    def __find_partition(self, n_col, n_target):
        '''Find optimal partition using given column'''
        midpoints = self.__find_midpoints(n_col)
        
        opt_gini = float("inf")
        opt_midpoint = None
        
        for point in midpoints:
            gini    = self.__gini_index(point, n_col, n_target)
            
            if gini < opt_gini:
                opt_gini = gini
                opt_midpoint = point
        
        v_left  = self.content[self.content[:, n_col] <= opt_midpoint]
        v_right = self.content[self.content[:, n_col] >  opt_midpoint]
                
        return opt_midpoint, opt_gini, v_left, v_right
    
    def find_accuracy(self):
        '''Finds the nod accuracy'''
        
        length = len(self.content)
        sum_elements = sum(self.content[:,0])
        
        if sum_elements/length >= 0.5:
            majority_class = 1
            accuracy = sum_elements/length
        else:
            majority_class = 0
            accuracy = 1 - sum_elements/length
            
        return majority_class, accuracy
    
    def partition(self, accuracy_threshold=0.9):
        
        # we need to make a copy to use it in recursive calls
        accuracy_thres = accuracy_threshold
        
        # Print general information about the class
        if self.print_accuracy==True:
            print("Tree node:", self.id)
            print("Level:", self.level)
            print("Number of elements:", len(self.content))
        
        # Calculate and print accuracy score
        majority_class, accuracy = self.find_accuracy()
        self.accuracy = accuracy
        
        if self.print_accuracy==True:
            print("Majority class:", majority_class)
            print("Accuracy:", round(accuracy, 4))
        
        # If accuracy is greater than threshold or we have more than 9 levels
        # Make this node terminal
        if accuracy>accuracy_thres or self.level>9:
            self.terminal = True
            
            if self.print_accuracy==True:
                print("Terminal node!")
                print("#######", "\n", "\n")
            return [[[len(self.content), sum(self.content[:,0]), majority_class, self.accuracy], self.rules]]
        
        
        # Find optimal partition
        gini = float("inf")
        part_accuracy = 0
        column = None
        midpoint = None
        left = None
        right = None
        
        for i in range(1, self.content.shape[1]):
            
            # skip variables thar have only one unique value
            unique_values = np.unique(self.content[:,i])
            
            if len(unique_values)<2:
                continue
            
            col_midpoint, col_gini, col_left, col_right = self.__find_partition(i, n_target=0)
            
            ##### BEGINNING OF THE NEW PART #######
            left_length    = len(col_left)
            left_sum       = sum(col_left[:,0])
            left_accuracy  = left_sum / left_length
            
            right_length   = len(col_right)
            right_sum      = sum(col_right[:,0])
            right_accuracy = right_sum / right_length    
            
            col_accuracy  = max([left_accuracy, right_accuracy])
            
            if part_accuracy < col_accuracy:
                part_accuracy = col_accuracy
                gini = col_gini
                column = i
                midpoint = col_midpoint
                left = col_left
                right = col_right
                
            ##### END OF THE NEW PART #######
                
        # Record the optimal partition
        self.gini = gini
        self.part_column = column
        self.part_midpoint = midpoint
        
        if self.print_accuracy==True:
            print("\n")
            print("Gini:", self.gini)
            print("Partition column:", self.part_column)
            print("Partition midpoint:", self.part_midpoint)
            
            print("#######", "\n", "\n")
        

        level = self.level + 1
        name  = self.id + "l"
        rule_next  = self.rules + [[self.part_column, self.part_midpoint, True]]
        self.left = DecisionTree(content=left, level=level, id=name, rules=rule_next, print_accuracy=self.print_accuracy)
            
        name  = self.id + "r"
        rule_next  = self.rules + [[self.part_column, self.part_midpoint, False]]
        self.right = DecisionTree(content=right, level=level, id=name, rules=rule_next, print_accuracy=self.print_accuracy)
            
        return self.left.partition(accuracy_thres) + self.right.partition(accuracy_thres)
    
    
    
def print_tree_metrics(tree_output, print_accuracy=True):
    '''Calculates accuracy, recall and precision of decision tree result'''
    
    true_positive = 0
    false_positive =0
    true_negative = 0
    false_negative = 0

    for i in range(len(tree_output)):

        if tree_output[i][0][2]==1:
            true_positive  += tree_output[i][0][1]
            false_positive += tree_output[i][0][0] - tree_output[i][0][1]
        else:
            true_negative  += tree_output[i][0][0] - tree_output[i][0][1]
            false_negative += tree_output[i][0][1]        
        

    accuracy  = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    recall    = true_positive / (true_positive + false_positive)
    precision = true_positive / (true_positive + false_negative)

    if print_accuracy==True:
        print("Accuracy:", round(accuracy*100, 2), "%")
        print("Recall:", round(recall*100, 2), "%")
        print("Precision:", round(precision*100, 2), "%")
        
    d = {"metrics": [accuracy, recall, precision]}
    return d
    
    

def lable_using_tree(data, tree_output):
    '''Label data using the rules generated by the tree'''
    
    # vector that will store predicted labels
    labels = []
    
    for j in range(data.shape[0]):

        for i in range(len(tree_output)):
    
            # label of this terminal node
            label = tree_output[i][0][2]

            # number of satisfied rules
            rule_ok = 0 
            
            for rule in tree_output[i][1]:
                if (data[j,rule[0]]<=rule[1])==rule[2]:
                    rule_ok += 1
                else:
                    break
            
            if rule_ok==len(tree_output[i][1]):
                labels += [label]
    return np.array(labels)



def eval_tree_test(data, prediction, print_accuracy=True):
    '''Calculates accuracy, recall and precision of tree prediction for the test data'''

    result = pd.DataFrame({"true": data[:,0],
                           "predicted": prediction})

    result['result'] = np.where((result['true']==1) & (result['predicted']==1), "true_positive", 0)
    result['result'] = np.where((result['true']==0) & (result['predicted']==0), "true_negative", result['result'])
    result['result'] = np.where((result['true']==0) & (result['predicted']==1), "false_positive", result['result'])
    result['result'] = np.where((result['true']==1) & (result['predicted']==0), "false_negative", result['result'])

    true_positive  = sum(np.where(result['result']=="true_positive",  1, 0))
    true_negative  = sum(np.where(result['result']=="true_negative",  1, 0))
    false_positive = sum(np.where(result['result']=="false_positive", 1, 0))
    false_negative = sum(np.where(result['result']=="false_negative", 1, 0))

    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    recall   = true_positive / (true_positive + false_positive)
    precision   = true_positive / (true_positive + false_negative)

    if print_accuracy==True:
        print("Accuracy:", round(accuracy*100, 2), "%")
        print("Recall:", round(recall*100, 2), "%")
        print("Precision:", round(precision*100, 2), "%")
        
    d = {"metrics": [accuracy, recall, precision]}
    return d    
    
    
    
