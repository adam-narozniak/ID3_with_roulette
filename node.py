"""Represents nodes of a tree for ID3 algorithm."""


class Node():
    def __init__(self, split_feature_name=None, split_value=None, prediction=None):
        self._split_feature_name = split_feature_name  # feature name e.g. x1 which has the highest information_gain therefore we split on it
        self._split_value = split_value  # value of the feature _split_on of its parent
        self._children = []
        self._prediction = prediction

    def add_child(self, node):
        self._children.append(node)

    def set_split_feature_name(self, split_feature_name):
        self._split_feature_name = split_feature_name

    def set_split_value(self, split_value):
        self._split_value = split_value

    def get_split_feature_name(self):
        return self._split_feature_name

    def get_split_value(self):
        return self._split_value

    def get_children(self):
        return self._children

    def get_prediction(self):
        return self._prediction
