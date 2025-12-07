import sys
import time

import tensorflow as tf

from tensorflow.keras.models import Model, load_model

from sklearn.metrics import log_loss, classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

import numpy as np

from kerasVggUtils import GetNClassesFromCsv, LoadDatasetWithVggPreProc, GetVggInputShape
	
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: <test.csv> <trained_network.h5>");
        sys.exit(-1)


    testPath = sys.argv[1]
    modelPath = sys.argv[2]

    nClasses = GetNClassesFromCsv(testPath)
    batch_size = 16

    X_test, Y_test = LoadDatasetWithVggPreProc(testPath) 
    Y_test = np.argmax(Y_test, axis=-1) # Convert one-hot to index

    model = load_model(modelPath)
    model.summary()

    
    print("Start prediction")
    t1 = time.time()
    predictions = model.predict(X_test, batch_size=batch_size, verbose=2)
    t2 = time.time()
    totalTime = t2 - t1
    print("Prediction time - ", totalTime, "s")

    Y_pred = np.argmax(predictions, axis=-1)

    print("Accuracy = ", accuracy_score(Y_test, Y_pred))
    print("Kappa accuracy = ", cohen_kappa_score(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

