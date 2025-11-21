import numpy as np
from scipy import stats
from DecisionTreeNode import DecisionTreeNode
from utils import best_feature_to_split_on, entropy


class DecisionTree:
    """
    Object to represent a decision tree model
    """

    def __init__(self, max_depth=None):
        
        """
        Constructor called when object is initialized. Only need
        to store max depth hyperparemeter and initialize root node to None.
        The only hyperparameter is max depth. When it is None this should
        train a decision tree with unconstrained depth.
        """
        
        ###################
        ## DO NOT MODIFY ##
        ###################

        # store hyperparameters
        self.max_depth = max_depth

        # root node
        self.root_node = None

    def train(self, X, y):
        
        """
        trains a decision tree by calling internal _train recursive method
        """

        ###################
        ## DO NOT MODIFY ##
        ###################

        self.root_node = self._train(X, y, depth=0)

    def _train(self, X, y, depth):
        
        """
        Trains a decision tree. You must implement this function recursively following the 
        instructions below. If max depth is None then this should train a decision tree
        with unconstrained depth.

        1) if all the labels are the same then return a leaf decision tree node
        2) if the depth has reached the max depth hyperparmeter, then return 
           a decison tree leaf node with the label being the majority label.
        3) find the best feature to split on using the best_feature_to_split_on function
        4) create a left and right node by recursively calling the _train method (the left
           node should contain all examples with feature value < threshold and the right node should
           contain all examples with feature value >= threshold)
        5) create and return a decision tree split node that points to the previously
           instantiated left and right nodes

        Args:
            X (2D numpy array): A numpy array of size (n x d) where each row a training instance
            y (1D numpy array): A numpy array of size (n) containing the labels

        Returns:
            DecisionTreeNode: root node of decision tree
        """
        
        # 1) If all the labels are the same → return a leaf
        if np.all(y == y[0]):
            return DecisionTreeNode(is_leaf=True, label=y[0])

        # 2) If depth reached max_depth → return majority label leaf
        if self.max_depth is not None and depth >= self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            majority = values[np.argmax(counts)]
            return DecisionTreeNode(is_leaf=True, label=majority)

        # 3) Find the best feature to split on
        best_feature, best_info_gain, best_split_value = best_feature_to_split_on(X, y)
        if best_split_value is None or best_info_gain <= 0:
            values, counts = np.unique(y, return_counts=True)
            majority = values[np.argmax(counts)]
            return DecisionTreeNode(is_leaf=True, label=majority)

        # 4) Split the data
        left_mask = X[:, best_feature] < best_split_value
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # Edge case: if split is invalid → majority leaf
        if len(y_left) == 0 or len(y_right) == 0:
            values, counts = np.unique(y, return_counts=True)
            majority = values[np.argmax(counts)]
            return DecisionTreeNode(is_leaf=True, label=majority)

        # Recursive calls
        left_node = self._train(X_left, y_left, depth + 1)
        right_node = self._train(X_right, y_right, depth + 1)

        # 5) Return split node
        return DecisionTreeNode(
            is_leaf=False,
            split_idx=best_feature,
            split_val=best_split_value,
            entropy=entropy(y),
            left_node=left_node,
            right_node=right_node
        )

    def predict(self, X):
        
        """
        computes predictions for multiple instances by iterating through
        each example and computing its prediction

        Args:
            X (2D numpy array): A numpy array of size (n x d) where each row a training instance

        Returns:
            y_pred: 1d array of predicted labels
        """

        ###################
        ## DO NOT MODIFY ##
        ###################

        # list to store predictions
        y_pred = []

        # iterate through all examples
        for x in X:

            # compute prediction for one example
            pred = self._predict(self.root_node, x)

            # append prediction to list
            y_pred.append(pred)

        return np.array(y_pred)

    def _predict(self, node, x):
        
        """
        Computes decision tree prediction for one example. This function must be implemented
        recursively by following the instructions below.

        1) if the node is a leaf node then return its label
        2) get the node's split idx and compare the input instance's feature val
           with the nodes split val, and return either a rescursive call to _predict
           on the left or right node

        Args:
            x (1d numpy array): 1d numpy array of features for one example to predict

        Returns:
            label: integer label that is the prediction
        """
        
        if node.is_leaf:
            return node.label
        if x[node.split_idx] < node.split_val:
            return self._predict(node.left_node, x)
        else:
            return self._predict(node.right_node, x)

    def accuracy_score(self, X, y):
        
        """
        Compute the decision tree prediction of X and compute the accuracy compared to 
        the actual labels y

        Args:
            X (2d numpy array): 2d numpy array of instance to compute accuracy on
            y (1d numpy array): 1d numpy array of labels for insyances in X

        Returns:
            accuracy (float): accuracy of predictions 
        """
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def visualize_tree(self):
        """
        visualizes decision tree
        """

        ###################
        ## DO NOT MODIFY ##
        ###################

        self._visualize_tree(self.root_node, level=0)

    def _visualize_tree(self, node, level):
        
        """
        visualizes decision tree

        ###################
        ## DO NOT MODIFY ##
        ###################
        """

        if node is not None:
            self._visualize_tree(node.right_node, level + 1)
            print(" " * (level * 30) + str(node))
            self._visualize_tree(node.left_node, level + 1)

    def in_order_split_vals(self):
        
        """
        Performs in order walk on tree and reutns split values. Return None
        value if node is a leaf node.

        ###################
        ## DO NOT MODIFY ##
        ###################
        """

        split_vals = []

        self._in_order_split_vals(self.root_node, split_vals=split_vals)

        return split_vals

    def _in_order_split_vals(self, node, split_vals):
        
        """
        Performs in order walk on tree and reutns split values. Return None
        value if node is a leaf node.

        ###################
        ## DO NOT MODIFY ##
        ###################
        """

        if node is not None:
            self._in_order_split_vals(node.left_node, split_vals)
            split_vals.append(np.round(node.split_val, 3) if not node.is_leaf else None)
            self._in_order_split_vals(node.right_node, split_vals)
