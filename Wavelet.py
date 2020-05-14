'''
@author: Philip Derbeko <philip.derbeko@gmail.com>
'''

import pandas as pd
import numpy as np
import pywt
import math
from itertools import chain
from Utils import *

# config
relative_error = True
drawTree = False
drawNoisePerLevel = False
compareModelError = False
wavelet = 'haar'

range_sensitivity = 1

class Model:
    """ Encapsulation of a model """
    def __init__(self, tree, waveco):
        self.tree = tree
        self.waveco = waveco

    def GetLeaves(self):
        if isinstance(self.tree, dict):
            leaves_index = sorted([x for x in self.tree.keys() if isinstance(x, int)], reverse=True)[0]
            return [x.value for x in list(self.tree[leaves_index].values())]
        else:
            return self.tree

    def GetWaveco(self):
        return self.waveco

class WNode:
    """ Encapsulation of a node with all attributes """
    def __init__(self, name, value):
        self.name = name 
        self.value = value
        self.prune_error = 0.0
        self.edge_value = 1.0
        self.random_noise = 0.0
        self.subtree_size = 0
        self.average_value = 0.0

def CreateTree(waveco, fill_all_levels=True):
    """ Given a wavelet coefficients creates and returns a tree
    """
    # if we dont reconstruct all the levels, then there is no real need to build the tree
    # so, just reconstruct the data and save it as leaves.
    if not fill_all_levels:
        return pywt.waverec(waveco, wavelet)

    id = 1
    # members map contains a hash of member nodes.
    # memmap key is the level that points to an array of nodes of the level    
    memmap = {} 
    memmap[0] = {}
    memmap[0][str(id)] = WNode(str(id), waveco[0][0])
    id += 1
    nmembers = 1
    level = 1
    startind = 1
    reco = waveco[0]
    for coind in range(startind, len(waveco)):
        memmap[level] = {}
        # reconstruct a level. 
        reco = pywt.idwt(reco, waveco[coind], wavelet)
        for i in range(0, nmembers):
            # left
            memmap[level][str(id)] = WNode(str(id), reco[i*2])
            # edge value is set only once
            memmap[level-1][str(int(id/2))].edge_value = waveco[coind][i]
            id += 1
            # right
            memmap[level][str(id)] = WNode(str(id), reco[i*2+1])
            id += 1
        nmembers *= 2 
        level += 1
        
    return memmap

def PruneModelAddNoise(model, sensitivity, epsilon):
    # get wavelet coefficients
    leaves = model.GetLeaves()
    pruned_coef = list(model.GetWaveco())
    num_of_leaves = len(leaves)
    # cut the last level of the wavelet
    # reco = pywt.idwt(waveco[leaves])

    model.waveco[-1] = np.zeros_like(model.waveco[-1])

    # add noise
    # we cut one level, so subtree size is 2
    sensitivity = sensitivity / 2.0
    temp_array = [x + np.random.laplace(sensitivity / epsilon, 1.0, 1) for x in model.waveco[-1]]
    model.waveco[-1] = np.array(temp_array).flatten()

    # Rebuild the tree 
    model.tree = CreateTree(model.waveco, False)    

def CreateModel(vec, wavelet):
    treemap = {} 
    waveco = pywt.wavedec(vec, wavelet)
    treemap = CreateTree(waveco, False)
    #CalculateSubTreeError(treemap)
    #CalculateRandomNoise(treemap)
    # what is the format of the histogram vector
    return Model(treemap, waveco)

def Wavelet(vec, sensitivity, eps):
    model = CreateModel(vec, wavelet)
    PruneModelAddNoise(model, sensitivity, eps)
    
    leaves = model.GetLeaves()
    if len(vec) % 2 == 1:
        return leaves[:-1]
    return leaves

# HERE - all the below are not sure needed 
###########################################################################
def CalculatePruneError(tree, average, lindex, node_name):
    """ Given a node and average, calculate the prune error of the subtree"""
    prune_error  = 0.0
    level_difference = 0
    left_bound = int(node_name)
    right_bound = int(node_name)
    for levind in sorted(tree.keys()):
        if levind <= lindex:
            continue
        level_difference += 1
        left_bound *= 2
        right_bound = right_bound * 2 + 1
    
    if level_difference == 0:
        return 0.0
    # now we know the difference between given node level and the last level.
    levind = sorted(tree.keys(), reverse=True)[0]
    level = tree[levind]
    for node in level.values():
        iname = int(node.name)
        if iname >= left_bound and iname <= right_bound: 
            prune_error += abs(node.value - average)

    return prune_error

def CalculateSubTreeError(tree):
    """ calculates a pruning error for each node """
    prevlevel = []
    for lindex in sorted(tree.keys(), reverse=True):
        level = tree[lindex]
        for node in level.values():
            if len(prevlevel) == 0:
                node.prune_error = 0.0
                node.subtree_size = 0
            else:
                left_child_name = str(int(node.name) * 2)
                right_child_name = str(int(node.name) * 2 + 1)
                left_child = prevlevel[left_child_name]
                right_child = prevlevel[right_child_name]
                  
                node.average_value = (left_child.value + right_child.value) / 2
                node.prune_error = CalculatePruneError(tree, node.average_value, lindex, node.name)
                node.subtree_size = left_child.subtree_size + right_child.subtree_size + 2
        prevlevel = level

def GetMaxDeviation(tree, average, lindex, node_name):
    """ Given a node and average, finds a values that diverges the most from the average in the subtree"""
    max_val = 0.0
    level_difference = 0
    left_bound = int(node_name)
    right_bound = int(node_name)
    for levind in sorted(tree.keys()):
        if levind <= lindex:
            continue
        level_difference += 1
        left_bound *= 2
        right_bound = right_bound * 2 + 1
    
    if level_difference == 0:
        return 0.0
    # now we know the difference between given node level and the last level.
    levind = sorted(tree.keys(), reverse=True)[0]
    level = tree[levind]
    for node in level.values():
        iname = int(node.name)
        if iname >= left_bound and iname <= right_bound: 
            if abs(node.value - average) > max_val: 
                max_val = node.value

    return max_val

def CalculateRandomNoise(tree):
    """ calculates a required random noise to preserve a privacy """
    for lindex in sorted(tree.keys(), reverse=True):
        level = tree[lindex]
        for node in level.values():
            #max_diff = GetMaxDeviation(tree, node.average_value, lindex, node.name)
            if node.subtree_size > 0:
                # sensitivity = float(max_diff) / node.subtree_size
                sensitivity = 1 / node.subtree_size
            else:
                sensitivity = range_sensitivity
            node.random_noise = np.random.laplace(sensitivity / epsilon, 1.0, 1)

def CalculateErrorByLevel(tree):
    """ Given a tree returns an array that contains random noise errors by level. 
    """
    errors = []
    for lindex in sorted(tree.keys()):
        level = tree[lindex]
        cur_error = 0.0
        for node in level.values():
            cur_error += node.random_noise
        errors.append(cur_error / len(level.values()))    
    return errors

def CalculatePruneErrorByLevel(tree):
    """ Given a tree returns an array that contains average pruning errors for each level. 
    """
    errors = []
    for lindex in sorted(tree.keys()):
        level = tree[lindex]
        cur_error = 0.0
        for node in level.values():
            cur_error += node.prune_error
        errors.append(cur_error / len(level.values()))    
    return errors


def GetColumnData(filename, column_number, shouldCleanData):
    data = pd.read_csv(filename, header=None, na_values='?',skipinitialspace=True)
    if shouldCleanData:
        clean_data = data.dropna()
    else:
        clean_data = data
    column_data = clean_data[column_number]
    return column_data