from id3_classifier import ID3Classifier
import io_utils
import logger_setup  # do not delete it
import numpy as np
import pandas as pd
import entropy

logger = logger_setup.logger
if __name__ == '__main__':
    X, y = io_utils.read_dataset("./data/spliceDTrainKIS.dat.txt")
    logger.info("Dataset loaded correctly")
    id3_classifier = ID3Classifier()
    # X = pd.DataFrame([
    #     ['A', 1],
    #     ['B', 1],
    #     ['B', 2],
    #     ['B', 2],
    #     ['B', 3]])
    # y = pd.Series([0, 1, 1, 0, 1])
    id3_classifier.fit(X, y)
    logger.info("Classifier fit done")
    y_pred = id3_classifier.predict(X)
    print(f"accuracy: {np.sum(y_pred==y)/X.shape[0]}")

