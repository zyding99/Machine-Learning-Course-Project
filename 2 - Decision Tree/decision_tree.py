###########################################
###########################################
###### Decision Tree & Random Forest ######
###########################################
###########################################


from collections import Counter, defaultdict
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from math import log
import save_csv
from sklearn import metrics
import random


class Node:
    def __init__(self, rule = None, parent = None, left = None, left_set = None, right = None, right_set = None):
        self.parent = parent
        self.rule = rule
        self.left = left
        self.left_set = left_set
        self.right = right
        self.right_set = right_set


class DecisionTree:

    def __init__(self, size = 0, features = []):
        self.features = features
        self.size = size
    
    
    def setting(self, features):
        self.features = features
        self.size = len(self.features)


    @staticmethod
    def calculate_entropy(y):
        H = 0
        count = defaultdict(int)
        prob = []
        for i in range(len(y)):
            count[y[i]] += 1
        for n in count.values():
            prob.append(n / len(y))
        for p in prob:
            H -= p * log(p, 2)                    
        return H


    @staticmethod
    def information_gain(y, y_l, y_r):
        H = DecisionTree.calculate_entropy(y)
        H_yl = DecisionTree.calculate_entropy(y_l)
        H_yr = DecisionTree.calculate_entropy(y_r)
        H_after =(len(y_l) * H_yl + len(y_r) * H_yr) / (len(y_l) + len(y_r))
        gain = H - H_after
        return gain


    @staticmethod
    def split(X, y, idx, thresh):
        x = X[:, idx]
        left_x, left_y,  right_x, right_y = [], [], [], []
        for i in range(len(x)):
            if x[i] >= thresh:
                left_x.append(X[i])
                left_y.append(y[i])
            else:
                right_x.append(X[i])
                right_y.append(y[i])
        left_x, right_x = np.array(left_x), np.array(right_x)
        node = Node(rule = (idx, thresh), left_set= [left_x, left_y], right_set = [right_x, right_y])
        return node


    def segmenter(self, X, y):
        segmenter = []
        for i in range(self.size):
        
            x = X[:, i]
            values = set(x)
            gains = []
            for thresh in values:
                node = DecisionTree.split(X, y, i, thresh)
                y_l, y_r = node.left_set[1], node.right_set[1]
                gain = DecisionTree.information_gain(y, y_l, y_r)
                gains.append((gain, thresh))
            max_gain, best_thresh = max(gains)
            segmenter.append((max_gain, i, best_thresh))
        best = max(segmenter)
        feature_idx, threshold = best[1], best[2]
        return feature_idx, threshold


    def fit(self, X, y, max_depth, mini_size, depth):
        while depth <= max_depth:
            feature_idx, threshold = self.segmenter(X, y)
            desicion_tree = DecisionTree.split(X, y, feature_idx, threshold)
            if depth < max_depth:  
                if len(desicion_tree.left_set[1]) >= mini_size and len(desicion_tree.right_set[1]) >= mini_size:
                    desicion_tree.left = self.fit(desicion_tree.left_set[0], desicion_tree.left_set[1], max_depth, mini_size, depth + 1)
                    desicion_tree.right = self.fit(desicion_tree.right_set[0], desicion_tree.right_set[1],  max_depth, mini_size, depth + 1)
                if len(desicion_tree.left_set[1]) >= mini_size and len(desicion_tree.right_set[1]) < mini_size:
                    desicion_tree.left = self.fit(desicion_tree.left_set[0], desicion_tree.left_set[1], max_depth, mini_size, depth + 1)
                    desicion_tree.right = Node()
                    desicion_tree.right.rule = Counter(desicion_tree.right_set[1]).most_common(1)[0][0]
                    break
                elif len(desicion_tree.left_set[1]) < mini_size and len(desicion_tree.right_set[1]) >= mini_size:
                    desicion_tree.right = self.fit(desicion_tree.right_set[0], desicion_tree.right_set[1],  max_depth, mini_size, depth + 1)
                    desicion_tree.left = Node()
                    desicion_tree.left.rule = Counter(desicion_tree.left_set[1]).most_common(1)[0][0]
                    break
                else:
                    desicion_tree.left, desicion_tree.right = Node(), Node()
                    desicion_tree.left.rule = Counter(desicion_tree.left_set[1]).most_common(1)[0][0]
                    desicion_tree.right.rule = Counter(desicion_tree.right_set[1]).most_common(1)[0][0]
                    break
            else:
                desicion_tree.left,  desicion_tree.right = Node(), Node()
                desicion_tree.left.rule = Counter(desicion_tree.left_set[1]).most_common(1)[0][0]
                desicion_tree.right.rule = Counter(desicion_tree.right_set[1]).most_common(1)[0][0]
                break
        return desicion_tree


    def predict(self, X, decision_tree):
        y = [0] * len(X)
        for i in range(len(X)):
            node = desicion_tree
            while True:
                if type(node.rule) == tuple:
                    idx, thresh = node.rule
                    if X[i][idx] >= thresh:
                        node = node.left
                    else:
                        node = node.right
                else:
                    y[i] = node.rule
                    break
        return y

    def __repr__(self):
        """
        TODO: one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize 
        a tree structure. You might have seen this before in CS61A.
        """
        return


class RandomForest():
    
    def __init__(self, features, number_of_tree = 5):
        self.features = features
        self.number_of_tree = number_of_tree


    def fit_and_predict(self, features, X, y, Z):
        trees = {}
        sub_X = {}
        N_features = len(features)
        N = int(N_features**0.5) + 1
        N_tree = self.number_of_tree
        
        for i in range(N_tree):
            idx = np.random.permutation(N)
            feature = []
            for j in idx:
                feature.append(features[j])
            trees[i] = DecisionTree()
            trees[i].setting(feature)
            sub_X[i] = X[:, idx]
        desicion_tree = {}
        predict = {}
        for j in range(N_tree):
            desicion_tree[j] = trees[j].fit(sub_X[j], y, 5, 30, 1)
            predict[j] = trees[j].predict(Z, desicion_tree[j])
        labels = []
        for k in range(len(Z)):
            label = []
            for i in range(N_tree):
                label.append(predict[i][k])
            label = Counter(label).most_common(1)[0][0]
            labels.append(label)
        return labels


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"

    if dataset == "titanic":
        # Load titanic data       
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter = ',', dtype = None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter = ',', dtype = None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        
        # TODO: preprocess titanic dataset
        # Notes: 
        # 1. Some data points are missing their labels
        # 2. Some features are not numerical but categorical
        # 3. Some values are missing for some features
        
    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam-dataset/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels']) # 1:Spam ; 0:Ham
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

        
        spam_tree = DecisionTree()
        spam_tree.setting(features)
        desicion_tree = spam_tree.fit(X, y, 5, 60, 1)


        labels = spam_tree.predict(X, desicion_tree)
        accuracy = metrics.accuracy_score(y, labels)
        print(accuracy)
        predict = spam_tree.predict(Z, desicion_tree)
        # save_csv.results_to_csv(np.array(predict))

        spam_RF = RandomForest(features)
        train = spam_RF.fit_and_predict(features, X, y, X)
        accuracy = metrics.accuracy_score(y, train)
        print(accuracy)
        test = spam_RF.fit_and_predict(features, X, y, Z)
        # save_csv.results_to_csv(np.array(test))

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)
    