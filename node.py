"""Represents nodes of a tree for ID3 algorithm."""


class Node():
    def __init__(self, split_on=None, value=None):
        self._split_on = split_on  # feature name e.g. x1 which has the highest information_gain therefore we split on it
        self._value = value  # value of the feature _split_on of its parent
        self._children = []
        self._prediction = None

    def add_child(self, node):
        self._children.append(node)

    def set_split_on(self, split_on):
        self._split_on = split_on

    def set_value(self, value):
        self._value = value