from __future__ import division, print_function
import numpy as np
import progressbar,math

from utils.misc import bar_widgets
from utils import divide_on_feature, train_test_split, standardize, mean_squared_error
from utils import calculate_entropy, accuracy_score, calculate_variance
from BinaryClassifcation.connect import lock1,lock2,lock3,lock4,lock5,lock6,grad,l_grad,r_grad,hes,l_hes,r_hes,results


class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree

# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        # 切割树的方法，gini，方差等
        self._send_data = None
        # Function to determine prediction of y at leaf
        # 树节点取值的方法，分类树：选取出现最多次数的值，回归树：取所有值的平均值
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y,feature_bins, loss=None):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y,feature_bins)
        self.loss = None

    def _build_tree(self, X, y,feature_bins, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""
        largest_impurity = 0
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data
        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)

        if current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(len(feature_bins)):
                # All values of feature_i
                #feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = feature_bins[feature_i]

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold

                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    y1 = np.atleast_2d(Xy1)[:, n_features:]
                    y2 = np.atleast_2d(Xy2)[:, n_features:]
                    self._send_data(y, y1, y2)

        lock5.acquire()
        largest_impurity=results.get()
        best_criteria = results.get()
        leaf_value = results.get()
        lock6.release()
        Xy1, Xy2 = divide_on_feature(Xy, best_criteria["feature_i"], best_criteria["threshold"])
        best_sets = {
            "leftX": np.atleast_2d(Xy1)[:, :n_features],  # X of left subtree
            "lefty": np.atleast_2d(Xy1)[:, n_features:],  # y of left subtree
            "rightX": np.atleast_2d(Xy2)[:, :n_features],  # X of right subtree
            "righty": np.atleast_2d(Xy2)[:, n_features:]  # y of right subtree
        }
        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            if best_criteria["feature_i"] == 1:
                split_func = lambda sample: sample >= best_criteria["threshold"]
            else:
                split_func = lambda sample: sample == best_criteria["threshold"]

            l_bin = np.array([sample for sample in feature_bins[best_criteria["feature_i"]] if split_func(sample)])
            r_bin = np.array([sample for sample in feature_bins[best_criteria["feature_i"]] if not split_func(sample)])
            l_bins = feature_bins
            r_bins = feature_bins
            l_bins[best_criteria["feature_i"]] = l_bin
            r_bins[best_criteria["feature_i"]] = r_bin
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], l_bins, current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], r_bins, current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print(tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class LeastSquaresLoss():
    """Least squares loss"""

    def gradient(self, actual, predicted):
        return actual - predicted

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class XGBoostRegressionTree(DecisionTree):
    """
    Regression tree for XGBoost
    - Reference -
    http://xgboost.readthedocs.io/en/latest/model.html
    """

    def _split(self, y):
        """ y contains y_true in left half of the middle column and
        y_pred in the right half. Split and return the two matrices """
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = np.power((self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2):
        # Split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y):
        # y split into y, y_pred
        y, y_pred = self._split(y)
        gradient = np.sum(self.loss.gradient(y, y_pred),axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation =  gradient / hessian
        return update_approximation

    def _send_grad(self, y, y1, y2):
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)
        lock1.acquire()
        grad.put(self.loss.gradient(y, y_pred))
        l_grad.put(self.loss.gradient(y1, y1_pred))
        r_grad.put(self.loss.gradient(y2, y2_pred))
        hes.put(self.loss.hess(y, y_pred))
        l_hes.put(self.loss.hess(y1, y1_pred))
        r_hes.put(self.loss.hess(y2, y2_pred))
        lock2.release()

    def fit(self, X, y,feature_bins):
        self._send_data = self._send_grad
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y,feature_bins)




class XGBoost(object):
    """The XGBoost classifier.

    Reference: http://xgboost.readthedocs.io/en/latest/model.html

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=200, learning_rate=0.01, min_samples_split=2,
                 min_impurity=1e-7, max_depth=30):
        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity  # Minimum variance reduction to continue
        self.max_depth = max_depth  # Maximum depth for tree

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        # Log loss for classification
        self.loss = LeastSquaresLoss()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss=self.loss)

            self.trees.append(tree)

    def fit(self, X, y,feature_bins):
        # y = to_categorical(y)
        m = X.shape[0]
        y = np.reshape(y, (m, -1))
        y_pred = np.zeros(np.shape(y))
        for i in self.bar(range(self.n_estimators)):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred,feature_bins)
            print(i)
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            y_pred =y_pred+ update_pred

    def predict(self, X):
        y_pred = None
        m = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred += update_pred

        y_bin=[]
        for i in y_pred:
            p=1 / (1 + math.exp(-i[0]))
            if p>=0.59:
                y=1
            else:
                y=0
            y_bin.append(y)
        return np.atleast_2d(y_bin).T
