
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
import sys

class Node:
    def __init__(self, nodeData):
        self.dataset = nodeData
        self.left = None
        self.right = None
        # Values used during testing 
        self.splitVal = None
        self.splitAttr = None
        self.cls = None

class DecisionTree:
    def __init__(self, dataset):
        self.rootData = self._readCSV(dataset)
        self.noOfAttr = self.rootData.shape[1] - 1
        self.Tree = Node(self.rootData)        
        self.rootNode = Node(self.rootData)

        # Configure the hyperparameters
        self.skips = None
        self.maxDepth = None
        self.Config()

    def Config(self, skips = 0.3):
        # Hyperparameters
        self.skips = skips
        self.maxDepth = self.rootData.shape[1]

    def Train(self):
        self._buildTree(self.rootNode, 0)

    def _buildTree(self, node, currentDepth):
        if currentDepth < self.maxDepth:  
        
            # If the node has only one type of class objects, assign class and return        
            if len(np.unique(node.dataset[:, -1])) == 1 :
                node.cls = self._setClass(node)
                return
            else:
                leftData, rightData, relativeAttrToSplitOn = self._getBestSplit(node)
                if (leftData is not None) and (rightData is not None):
                    node.left  = Node(leftData)
                    node.right = Node(rightData) 
                    # A node's attr to split on only takes the relative index: 
                    # If root node has attrs to split on as 0 1 2 3 4 5 6, some node might get 
                    # 0 4 5 6 . If this node splits on '4', it is actually stored as '1'. 
                    node.splitAttr = relativeAttrToSplitOn      
                    self._buildTree(node.left,  currentDepth +1)
                    self._buildTree(node.right, currentDepth +1)
                else:
                    # This node cannot be split further, so assign a class and return 
                    node.cls = self._setClass(node)
        else:
            # Max depth reached. Assign class and return 
            node.cls = self._setClass(node)
            return


    def _setClass(self, node):
        u, c = np.unique(node.dataset[:,-1], return_counts=True) 
        return node.dataset[:, -1][np.argmax(c)]

    def _getBestSplit(self, node):
        # If the number of rows received is (0 or 1) OR columns is 1 (i.e. has only classes), cannot split, so return None 
        if (node.dataset.shape[0] <= 1) or (node.dataset.shape[1] == 1):
                return None, None, None
        # Get the best split and also 
        # assign the node.splitval to the best split condition
        splitsInxs = []
        entr = []
        
        # Iterate over the attributes and get the best split value for each chosen attribute
        for attr in range(node.dataset.shape[1] - 1):       # Ignore the last column, has the classes
            spltPointInx, entropyGen = self._getBestSplitVal(node.dataset[:, [attr,-1]])
            splitsInxs.append(spltPointInx)
            entr.append(entropyGen)
        
        attrToSplitOnInx    = entr.index(min(entr))
        valToSplitOnInx     = splitsInxs[attrToSplitOnInx]      # Get the index where an attr gives min entropy
        
        v = valToSplitOnInx                                     # for bervity
        a = attrToSplitOnInx                                    # for bervity
        # sort array with the attr to split on, slice away
        srtdArr = node.dataset[node.dataset[:, a].argsort()][::-1]
        node.splitVal = srtdArr[v][a]
        left  = np.concatenate( (  srtdArr[v:, :a], srtdArr[v:, a+1: ]  ), axis=1)
        right = np.concatenate( (  srtdArr[:v, :a], srtdArr[:v, a+1: ]  ), axis=1)
        # get left, right, delete column, return
        # But before returning, check if either one of the left/right has zero rows, meaning no split was done
        if (left.shape[0] == 0) or (right.shape[0] == 0): 
            return None, None, None
        else:
            return left, right, attrToSplitOnInx

    def _getBestSplitVal(self, arr):
        # We now have a Kx2 array, where the last col = class a row belongs to 
        # Sort the array and check entropy for the top and bottom splits, 
        # return the val and entropy of the least entropy gen. 
        minEntropys = []
        srtdArr = arr[arr[:,0].argsort()][::-1]      # Sort w.r.t. the attribute vals, in dec order to get <= while slicing
        
        for i in range(arr.shape[0]):
            minEntropys.append( (srtdArr[:i].shape[0]/srtdArr.shape[0]) * self._getEntropy(srtdArr[:i][:,-1] ) + 
                                (srtdArr[i:].shape[0]/srtdArr.shape[0]) * self._getEntropy(srtdArr[i:][:,-1] )  )
        inxOfMinEnt = minEntropys.index(min(minEntropys))
        return inxOfMinEnt, minEntropys[inxOfMinEnt]

    def _getEntropy(self, colArray):
        # Given a column vector, gets the scalar entropy
        # by summming over the p/sum(classes) * log2 of the same fraction
        unq, cnt = np.unique(colArray, return_counts=True)
        return sum(-1*cnt/sum(cnt)*np.log2(cnt/sum(cnt)))

    def Test(self, value):
        # Value is a np array containing the attributes and the values. 
        # Iterate over the number of test cases, for each getting the class it belongs to
        retArr = np.zeros(value.shape[0])
        for i in range(value.shape[0]):
            retArr[i] = self._getClass(value[i])
        return retArr

    def _getClass(self, testSample):
        return self._searchTree(self.rootNode, testSample)

    def _searchTree(self, node, arr):
        if node.cls is not None:
            v = node.splitAttr
            if ( arr[v] <= node.splitVal ):
                return self._searchTree(node.left,  np.concatenate(( arr[v: ], arr[:v+1])) )
            else:
                return self._searchTree(node.right, np.concatenate(( arr[v: ], arr[:v+1])) )
        else:
            return node.cls

    def _readCSV(self, datasetPath):  
        return np.genfromtxt(datasetPath, dtype=float, delimiter=",")

    # def _getClasses(self, data):
    #     # gets the number of unique classes
    #     return np.unique(data[:,-1]).shape[0]


if __name__ == "__main__":
    Des = DecisionTree("datasets\\banknote_auth.csv")
    Des.Train()
    print("Done!")