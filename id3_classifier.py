from node import Node
from math import log


def add_id_to_same_value_list(row_d, row_id, list):
    for el in list:
        if el.type == row_d:
            el.id_with_same_type.append(row_id)
            return


class SetElement:
    def __init__(self, type):
        self.type = type
        self.id_with_same_type = []


class FreqElement:
    def __init__(self, type):
        self.type = type
        self.number_of_occurrence = 1
        self.number_of_occurrence_normalized = 0


def first_appearance(y, list):
    for el in list:
        if y == el.type:
            return False
    return True


def add_occurrence(y, list):
    for el in list:
        if y == el.type:
            el.number_of_occurrence += 1
            return


def calculate_entropy(list_of_y):
    frequency_of_y = []
    for y in list_of_y:
        if first_appearance(y, frequency_of_y):
            new_element = FreqElement(y)
            frequency_of_y.append(new_element)
        else:
            add_occurrence(y, frequency_of_y)

    for el in frequency_of_y:
        el.number_of_occurrence_normalize = float(el.number_of_occurrence) / float(len(list_of_y))

    entropy = 0
    for el in frequency_of_y:
        entropy += el.number_of_occurrence_normalize * log(el.number_of_occurrence_normalize)

    entropy *= (-1)
    return entropy


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

    # tu jest liczony wzor z wykladu na I(U)
    # I(U) = - SUM[1..N](f_i * ln(f_i))
    def calculate_total_entropy(self):
        return calculate_entropy(self._y)

    # tu jest liczony wzór na Inf(d, U)
    def calculate_entropy_divided_set(self, d):
        values_for_d_column = []
        id = 0
        for row in self._X:
            if first_appearance(row[d], values_for_d_column):
                new_el = SetElement(row[d])
                new_el.id_with_same_type.append(id)
                values_for_d_column.append(new_el)
            else:
                add_id_to_same_value_list(row[d], id, values_for_d_column)
            id += 1

        entropy = 0
        for val in values_for_d_column:
            y = []
            for idx in val.id_with_same_type:
                y.append(self._y[idx])
            d_entropy = calculate_entropy(y)
            entropy += (len(val.id_with_same_type) / float(id)) * d_entropy

        return entropy

    def id3(self, X, split_on):
        """

        Args:
            X(pd.DataFrame): part of X from last iteration (smaller than original dataset by some features)
            y(pd.DataFrame): corresponding to X labels

        Returns:

        """
        if len(X[split_on].unique() <= 1):
            return Node(None, )


if __name__ == '__main__':
    obj = ID3Classifier()
    obj.fit([
        ['A', 1],
        ['B', 1],
        ['B', 2],
        ['B', 2],
        ['B', 3]
    ], [0, 1, 1, 0, 1])
    print(obj.calculate_total_entropy() - obj.calculate_entropy_divided_set(1))
    print(obj.calculate_total_entropy())
