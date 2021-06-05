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
    id3_classifier.fit(X, y)
    print(X, y)
