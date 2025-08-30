
import numpy as np
from collections import Counter

class Node:
    def __init__(self,feature=None,threshold=None,right=None,left=None,*,value):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value

    def is_leaf_node(self):
        return self.value is not None 



class DecisionTree:
    def __init__(self,n_features=None,min_samples_split=2,max_depth=100,random_state=None):
        self.n_features = n_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self,X,y):
        n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.n_features=n_features
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #check the stopping criteria
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_inds = np.random.choice(n_feats,self.n_features,replace=False)

        #find the best split
        best_feature,best_thresh = self._best_split(X,y,feat_inds)

        # if no valid split is found → make leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        #create child node
        left_inds, right_inds = self._split(X[:,best_feature],best_thresh)
        left = self._grow_tree(X[left_inds,:],y[left_inds],depth+1)
        right = self._grow_tree(X[right_inds,:],y[right_inds],depth+1)
        return Node(feature=best_feature,threshold=best_thresh,right=right,left=left,value=None)

    def _best_split(self,X,y,feat_inds):
        best_gain = -1

        split_ind, split_threshold = None,None

        for feat_ind in feat_inds:
            X_column = X[:,feat_ind]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                #calculate the information gain
                gain = self._information_gain(y,X_column,thr)

                if gain>best_gain:
                    best_gain = gain
                    split_ind = feat_ind
                    split_threshold = thr

        return split_ind, split_threshold

    def _information_gain(self,y,X_column,thr):
        #parent entropy
        parent_entropy = self._entropy(y)

        #children entropy
        left_inds,right_inds = self._split(X_column,thr)

        if len(left_inds)==0 or len(right_inds)==0:
            return 0
        
        lce = self._entropy(y[left_inds])
        rce = self._entropy(y[right_inds])

        ln = len(left_inds)
        rn = len(right_inds)
        #weighted entropy
        weighet_entropy = (lce*ln + rce*rn)/(ln+rn)
        #IG
        IG = parent_entropy - weighet_entropy
        return IG

    def _split(self,X_column,threshold):
        left_inds = np.argwhere(X_column<=threshold).flatten()
        right_inds = np.argwhere(X_column>threshold).flatten()
        return left_inds,right_inds

    def _entropy(self,y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p * np.log2(p) for p in ps if p>0])


    def _most_common_label(self,y):
        most_common = Counter(y).most_common(1)
        return most_common[0][0]

    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        
        return self._traverse_tree(x,node.right)
        
