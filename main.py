from id3_classifier import ID3Classifier
from id3_roulette_classifier import ID3RouletteClassifier
import io_utils
import logger_setup  # do not delete it
import numpy as np
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_predict
import pandas as pd
import entropy

logger = logger_setup.logger
if __name__ == '__main__':
    X, y = io_utils.read_dataset("./data/spliceDTrainKIS.dat.txt")
    logger.info("Dataset loaded correctly")

    id3_roulette_classifier = ID3RouletteClassifier()
    # y_preds = cross_val_predict(id3_classifier, X, y, cv=2)#, scoring=make_scorer(accuracy_score))
    # print(y_preds)
    # print(sklearn.metrics.confusion_matrix(y, y_preds))
    y_preds = []
    conf_matrix = []
    acc_scores = []
    skf = StratifiedKFold(n_splits=2)
    for train_index, test_index in skf.split(X, y):
        id3_classifier = ID3Classifier()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        id3_classifier.fit(X_train, y_train)
        y_pred = id3_classifier.predict(X_test)
        y_preds.append(y_pred)
        conf_matrix.append(sklearn.metrics.confusion_matrix(y_test, y_pred))
        acc_scores.append(accuracy_score(y_test, y_pred))
    for i in range(len(y_preds)):
        print(conf_matrix[i])
        tn, fp, fn, tp = conf_matrix[i].ravel()
        print(f"True negative: {tn}; False positive: {fp}\nFalse negative: {fn}; True positive: {tp}")
        print(acc_scores[i])

    # X = pd.DataFrame([
    #     ['A', 1],
    #     ['B', 1],
    #     ['B', 2],
    #     ['B', 2],
    #     ['B', 3]])
    # y = pd.Series([0, 1, 1, 0, 1])
    # id3_roulette_classifier.fit(X, y)
    # logger.info("Classifier fit done")
    # y_pred = id3_roulette_classifier.predict(X)
    # print(f"accuracy: {np.sum(y_pred==y)/X.shape[0]}")

