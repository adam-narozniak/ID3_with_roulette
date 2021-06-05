from node import Node
import sys
import numpy as np
import pandas as pd
import logging
import entropy
from tqdm import tqdm

logger = logging.getLogger("ID3")


class ID3Classifier:
    """Handles id3 algorithm operations. Mimics sklearn methods."""

    def __init__(self):
        self._X = None
        self._y = None
        self._most_frequent_y = None
        self._root = None
        self._fit = False
        self._unique_in_columns = None

    def fit(self, X, y):
        logger.info("Fitting started.")
        self._X = X
        self._y = y
        self._most_frequent_y = self.calculate_most_frequent_y(y)
        self._calculate_unique_in_columns()
        self._root = self._id3(self._X, self._y, None)
        self._fit = True
        logger.info("Fitting finished.")

    def predict(self, X):
        """
        Predicts classes of instances.

        Returns:
            y(pd.Series): predictions
        """
        if self._fit == False:
            logger.exception("You can not use predict before fit.")
            sys.exit(1)
        y = []

        for idx, x in X.iterrows():
            current_node = self._root
            predicted = False
            while predicted is False:
                prediction = current_node.get_prediction()
                if prediction is not None:
                    # leaf node reached; move to the next instance
                    y.append(prediction)
                    predicted = True
                    break
                split_feature_name = current_node.get_split_feature_name()
                x_split_val = x[split_feature_name]
                for child in current_node.get_children():
                    prediction = child.get_prediction()
                    if prediction is not None:
                        # leaf node reached; move to the next instance
                        y.append(prediction)
                        predicted = True
                        break
                    if x_split_val == child.get_split_value():
                        current_node = child
                        break
        return pd.Series(y)

    def _id3(self, X, y, split_on_value):

        """
        Implementation of id3 algorithm.

        Args:
            X(pd.DataFrame): part of X from last iteration (smaller than original dataset by some features)
            y(pd.DataFrame): corresponding to X labels
            split_on_value: value on which it was split

        Returns:
            root_node(Node)
        """
        # instances are indistinguishable; return the only possible prediction
        if len(y.unique()) == 1:
            return Node(None, None, y.iloc[0])
        elif X.size == 0:
            return Node(None, None, self._most_frequent_y)  # TO CHECK: middle arg doesn't matter

        features = X.columns.values
        info_gain_dict = {feature: entropy.calculate_info_gain(X, y, feature) for feature in features}
        best_feature = max(info_gain_dict, key=info_gain_dict.get)
        feature_values = self._unique_in_columns
        root_node = Node(split_feature_name=best_feature, split_value=split_on_value, prediction=None)
        for feature_value in feature_values[best_feature]:
            mask = X[best_feature] == feature_value
            sub_X = X[mask]
            sub_y = y[mask]
            node = self._id3(sub_X.drop(best_feature, axis=1), sub_y, feature_value)
            root_node.add_child(node)
        return root_node

    def calculate_most_frequent_y(self, y):
        return y.value_counts().index[0]  # if two counts are equal, the first return by value_counts will be taken

    def _calculate_unique_in_columns(self):
        """Calculate values unique in each columns."""
        unique_in_col = pd.Series(np.empty(self._X.shape[1]), index=self._X.columns.values)
        for col in self._X.columns:
            unique_in_col[col] = self._X[col].unique()
        self._unique_in_columns = unique_in_col
