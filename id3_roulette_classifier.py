"""ID3 algorithm modification - instead of choosing a feature based on the best information gain, roulette wheel is
 implemented - it's probability based choice where the highest score has the highest chance of being chosen, but it
 is not guaranteed."""
from id3_classifier import ID3Classifier
from node import Node
import entropy
import numpy as np


class ID3RouletteClassifier(ID3Classifier):

    def _id3(self, X, y, split_on_value):
        """
        Implementation of id3 algorithm with roulette-based feature choosing.

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
        best_feature = self._roulette_choice(info_gain_dict)
        feature_values = self._unique_in_columns
        root_node = Node(split_feature_name=best_feature, split_value=split_on_value, prediction=None)
        for feature_value in feature_values[best_feature]:
            mask = X[best_feature] == feature_value
            sub_X = X[mask]
            sub_y = y[mask]
            node = self._id3(sub_X.drop(best_feature, axis=1), sub_y, feature_value)
            root_node.add_child(node)
        return root_node

    def _roulette_choice(self, info_gain_dict):
        """Chose feature(key) based on roulette wheel."""
        sum_info_gain = sum(info_gain_dict.values())
        try:
            probabilities = [p/sum_info_gain for p in info_gain_dict.values()]
        except ZeroDivisionError: #happens when the data is identical
            probabilities = [1/len(info_gain_dict) for i in range(len(info_gain_dict))]
        rng = np.random.default_rng()
        return rng.choice(list(info_gain_dict.keys()), p=probabilities)
