from node import Node

class ID3Classifier():
    """Handles id3 algorithm operations. Mimics sklearn methods."""
    def __init__(self):
        self._X = None
        self._y = None
        self._root = None

    def fit(self, X, y):
        self._X = X
        self._y = y

        self._labels = 1

    def id3(self, X, split_on):
        """

        Args:
            X(pd.DataFrame): part of X from last iteration (smaller than original dataset by some features)
            y(pd.DataFrame): corresponding to X labels

        Returns:

        """
        if len(X[split_on].unique() <= 1):
            return Node(None, )
