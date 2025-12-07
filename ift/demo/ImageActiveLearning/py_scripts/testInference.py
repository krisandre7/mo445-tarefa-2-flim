import time
from keras.models import Model, load_model
import keras
import tensorflow as tf
import json
import numpy as np
import sys
from keras.preprocessing import image

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usage: <keras_model.h5> <image_path>")
        sys.exit(-1)
    inicio = time.time()
    model = load_model(sys.argv[1])

    # Convenient way ot checking output of hidden layers
    #tgt_layer = 21
    #while len(model.layers) > tgt_layer:
    #    model.layers.pop()
    #    model.layers[-1].outbound_nodes = []
    #    model.outputs = [model.layers[-1].output]
    #    #model.pop()
    img = image.load_img(sys.argv[2])
    img = image.img_to_array(img, data_format="channels_last")
    img = np.expand_dims(img, axis = 0)
    res = model.predict(img)[0]
    print(res.shape)

    # For Conv layers (theano)
    #for kernel in range(len(res)):
    #    for row in range(len(res[kernel])):
    #        for col in range(len(res[kernel][row])):
    #            print("Kernel %d row %d col %d = %f" % (kernel, row, col, res[kernel][row][col]))
    # For Conv layers (tensorlow)
    #for row in range(len(res)):
    #    for col in range(len(res[row])):
    #        for kernel in range(len(res[row][col])):
    #            print("row %d col %d kernel %d = %f" % (row, col, kernel, res[row][col][kernel]))

    # For FC layers
    for i in range(len(res)):
        print("Output %d = %f\n" % (i, res[i]))

    # Guide for output shape
    #for l in model.layers:
    #    print(l.output[0].shape)

    fim = time.time()
    total = fim-inicio
    print( "Execution time - ", fim-inicio," sec")
