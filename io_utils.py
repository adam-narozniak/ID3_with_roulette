"""Handle input, output."""
import pandas as pd


def read_dataset(file_path):
    """Reads input following convention: first line - no important, next line y_i value, next line x_i,
    where i is from 0 to the number of samples.

    Returns:
        X(pd.DataFrame), y(pd.Series):
    """
    X = []
    y = []
    with open(file_path, "r") as f:
        f.readline()
        idx = 0
        for line in f:
            if idx % 2 == 0:
                y.append(line.strip('\n'))
            else:
                X.append([*line.strip('\n')])
            idx += 1
    y = pd.Series(y)
    X = pd.DataFrame(X)
    labels = [f"x{number:02}" for number in range(X.shape[1])]
    X.columns = labels
    return X, y
