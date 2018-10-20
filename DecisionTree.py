import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
import sys
import time


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

        if self.rootData.shape[0] < 1000:
            self.Config(binPercent=100)
        elif self.rootData.shape[0] < 10000:
            self.Config(binPercent=50, maxDepth=int(self.rootData.shape[1]*0.7))
        else:
            self.Config(
                binPercent=1, 
                maxDepth=int(self.rootData.shape[1]*0.5) 
            )
        
        # Performance Metrics
        self.srtTimes = []
        self.chkTimes = []
        self.DReadT = 0
        self.TrainTime = 0

    def Config(self, binPercent=100, maxDepth=np.Inf):
        # Hyperparameters
        if binPercent == 100 :
            raise ValueError("Skip Ratio /must be a value in the range (0, 1]  ")
        elif binPercent > 50:
            print("Warning! Setting bin% above 0.5 is the same as setting it to 100%!")
            time.sleep(5)
            print("Continuing...")
        
        self.skips = int(100/binPercent)          # Given a percentage, calculate actual skips 
        self.maxDepth = maxDepth

    def Train(self):
        a = time.time()
        self._buildTree(self.rootNode, 0)
        self.TrainTime = time.time() - a

    def _buildTree(self, node, currentDepth):
        if currentDepth < self.maxDepth:

            # If the node has only one type of class objects, assign class and return
            if len(np.unique(node.dataset[:, -1])) == 1:
                node.cls = self._setClass(node)
                return
            else:
                leftData, rightData, relativeAttrToSplitOn, valueToSplitOn = self._getBestSplit(node)
                if (leftData is not None) and (rightData is not None):
                    node.left = Node(leftData)
                    node.right = Node(rightData)
                    # A node's attr to split on only takes the relative index:
                    # If root node has attrs to split on as 0 1 2 3 4 5 6, some node might get
                    # 0 4 5 6 . If this node splits on '4', it is actually stored as '1'.
                    node.splitAttr = relativeAttrToSplitOn
                    node.splitVal = valueToSplitOn
                    self._buildTree(node.left,  currentDepth + 1)
                    self._buildTree(node.right, currentDepth + 1)
                else:
                    # This node cannot be split further, so assign a class and return
                    node.cls = self._setClass(node)
        else:
            # Max depth reached. Assign class and return
            node.cls = self._setClass(node)
            return

    def _setClass(self, node):
        _, c = np.unique(node.dataset[:, -1], return_counts=True)
        return node.dataset[:, -1][np.argmax(c)]

    def _getBestSplit(self, node):
        # If the number of rows received is (0 or 1) OR columns is 1 (i.e. has only classes), cannot split, so return None
        if (node.dataset.shape[0] <= 1) or (node.dataset.shape[1] == 1):
            return None, None, None, None
        # Get the best split and also
        # assign the node.splitval to the best split condition
        splitsInxs = []
        entr = []

        # Iterate over the attributes and get the best split value for each chosen attribute
        # Ignore the last column, has the classes
        for attr in range(node.dataset.shape[1] - 1):
            spltPointInx, entropyGen = self._getBestSplitVal(
                node.dataset[:, [attr, -1]])
            splitsInxs.append(spltPointInx)
            entr.append(entropyGen)

        attrToSplitOnInx = entr.index(min(entr))      # Get the index where an attr gives min entropy
        valToSplitOnInx = splitsInxs[attrToSplitOnInx]

        v = valToSplitOnInx                                     # for bervity
        a = attrToSplitOnInx                                    # for bervity
        # sort array with the attr to split on, slice away
        srtdArr = node.dataset[node.dataset[:, a].argsort()][::-1]
        splV = srtdArr[v][a]
        left  = np.concatenate((srtdArr[v:, :a], srtdArr[v:, a+1: ]), axis=1)
        right = np.concatenate((srtdArr[:v, :a], srtdArr[:v, a+1: ]), axis=1)
        # get left, right, delete column, return
        
        # But before returning, check if either one of the left/right has zero rows, meaning no split was done
        if (left.shape[0] == 0) or (right.shape[0] == 0):
            return None, None, None, None
        else:
            return left, right, attrToSplitOnInx, splV

    def _getBestSplitVal(self, arr):
        # We now have a Kx2 array, where the last col = class a row belongs to
        # Sort the array and check entropy for the top and bottom splits,
        # return the val and entropy of the least entropy gen.
        minEntropys = []
        srtStart = time.time()
        srtdArr = arr[arr[:,0].argsort()][::-1]     # Sort w.r.t. the attribute vals, in dec order to get <= while slicing
        srtEnd = time.time()
        
        self.srtTimes.append(srtEnd-srtStart)
        # For better speed at the cost of accuracy, we skip the entropy check for a few rows
        # and instead check the entropy for every P rows, P = hyperparameter.
        # If P = 0.3, then 30% of the rows are not checked.
        chkSt = time.time()
        for i in range(0, arr.shape[0], self.skips):
            minEntropys.append((srtdArr[:i].shape[0]/srtdArr.shape[0]) * self._getEntropy(srtdArr[:i][:, -1]) +
                               (srtdArr[i:].shape[0]/srtdArr.shape[0]) * self._getEntropy(srtdArr[i:][:, -1]))
        chkEnd = time.time()
        self.chkTimes.append(chkEnd-chkSt)

        inxOfMinEnt = minEntropys.index(min(minEntropys))
        return inxOfMinEnt*self.skips, minEntropys[inxOfMinEnt]

    def _getEntropy(self, colArray):
        # Given a column vector, gets the scalar entropy
        # by summming over the p/sum(classes) * log2 of the same fraction
        _, cnt = np.unique(colArray, return_counts=True)
        return sum(-1*cnt/sum(cnt)*np.log2(cnt/sum(cnt)))

    def Test(self, testArr):
        if len(testArr.shape) != 2:
            raise ValueError("Takes 2D np.array types with with each row as test data")
        # Value is a np array containing the attributes and the values.
        # Iterate over the number of test cases, for each getting the class it belongs to
        retArr = np.zeros(testArr.shape[0])
        for i in range(testArr.shape[0]):
            retArr[i] = self._getClass(testArr[i])
        return retArr

    def _getClass(self, testSample):
        return self._searchTree(self.rootNode, testSample)

    def _searchTree(self, node, arr):
        if node.splitVal is not None:
            v = node.splitAttr
            if arr[v] <= node.splitVal:
                return self._searchTree(node.left,  np.concatenate(( arr[:v], arr[v+1: ])) )
            else:
                return self._searchTree(node.right, np.concatenate(( arr[:v], arr[v+1: ])) )
        else:
            return node.cls

    def _readCSV(self, datasetPath):
        st = time.time()
        arr = np.genfromtxt(datasetPath, dtype=float, delimiter=",")
        self.DReadT = time.time() - st
        print("Data read to memory in ", str.format( "{0:5.3}", self.DReadT) )
        return arr
