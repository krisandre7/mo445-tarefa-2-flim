import io
import time

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
import tensorflow.keras
import json
import numpy as np
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usage: <keras_model.h5> <path_to_new_ift_model.json> [path_to_new_ift_model_weights]")
        sys.exit(-1)
    inicio = time.time()
    model = load_model(sys.argv[1])
    save_weights = (len(sys.argv) > 3)
        
    # Initialize IFT model with the default image_layer
    ift_json = {
        "n_layers": 1,
        "z_size": 1,
        "layers": {
            "layer_1": {
                 "layer_type": "image_layer"
            }
        }
    }

    if (model.layers[1].data_format == "channels_first"):
        ift_json["n_bands"] = model.input_shape[1]
        ift_json["x_size"] = model.input_shape[3]
        ift_json["y_size"] = model.input_shape[2]
    else:
        ift_json["n_bands"] = model.input_shape[3]
        ift_json["x_size"] = model.input_shape[2]
        ift_json["y_size"] = model.input_shape[1]

    if (save_weights):
        weight_path = sys.argv[3]
        ift_json["orig_weights_path"] = weight_path
        wFp = open(weight_path, 'w+b')
    
    firstFCLayer = True
    paddingData = 0
    for i in range(len(model.layers)):
        # Skip padding layers as IFT network does not explicitly model them
        if (isinstance(model.layers[i], tensorflow.keras.layers.ZeroPadding2D)):
            paddingData = model.layers[i].padding[0]
            continue

        # Our input layer is hard-coded as some models do not explicitly indicate it
        if (isinstance(model.layers[i], tensorflow.keras.layers.InputLayer)):
            continue

        # Skip flattening layers as IFT network does not explicitly model them
        if (isinstance(model.layers[i], tensorflow.keras.layers.Flatten)):
            continue

        # Skip dropout as we won't perform additional training
        if (isinstance(model.layers[i], tensorflow.keras.layers.Dropout)):
            continue

        ift_json["n_layers"] += 1
        layer_id = "layer_" + str(ift_json["n_layers"])

        # Convolution layer
        if (isinstance(model.layers[i], tensorflow.keras.layers.Conv2D)):
            # Build IFT json data
            ift_json["layers"][layer_id] = {
                "layer_type": "conv_layer",
                "kernel_size": model.layers[i].kernel_size[0],
                "stride":  model.layers[i].strides[0],
                "n_kernels": model.layers[i].filters,
                "activ_func": "relu", # Hardcoded for now as model.layers[i].activation is a function and we only use relu 
                "padding": paddingData
            }

            paddingData = 0
            if (model.layers[i].padding == "same"):
                ift_json["layers"][layer_id]["padding"] = int((ift_json["layers"][layer_id]["kernel_size"] - 1) / 2)

            if (save_weights):
                weights = model.layers[i].get_weights()
                bias = weights[1].reshape(-1)
                weights = weights[0]
                weights = np.swapaxes(weights, 0, 3)
                weights = np.swapaxes(weights, 1, 2)
                weights = weights.reshape(-1)
                wFp.write(bias.tobytes(order='C'))
                wFp.write(weights.tobytes(order='C'))
                ift_json["layers"][layer_id]["weights_format"] = "(bias), (nFilters, nBands, columns, rows)"
                ift_json["layers"][layer_id]["weights_nbytes"] = 4 * (len(bias) + len(weights))


        # Fully connected layer
        if (isinstance(model.layers[i], tensorflow.keras.layers.Dense)):
            ift_json["layers"][layer_id] = {
                "layer_type": "fc_layer",
                "units": model.layers[i].units,
                "activ_func": "relu" # Hardcoded for now as model.layers[i].activation is a function and we only use relu
            }

            if (save_weights):
                weights = model.layers[i].get_weights()
                bias = weights[1].reshape(-1)
                if (firstFCLayer):
                    orig_dim = model.layers[i-1].input[0].shape
                    weights = weights[0].reshape((orig_dim[0], orig_dim[1], orig_dim[2], model.layers[i].units))
                    if (model.layers[i-1].data_format == "channels_first"):
                        weights = np.swapaxes(weights, 0, 3)
                        weights = np.swapaxes(weights, 1, 3)
                    else:
                        print(orig_dim)
                        weights = np.swapaxes(weights, 0, 3)
                        weights = np.swapaxes(weights, 1, 2)
                    weights = weights.reshape(-1)
                    ift_json["layers"][layer_id]["weights_format"] = "(bias), (units, nBands, columns, rows)"
                    firstFCLayer = False
                else:
                    weights = weights[0]
                    weights = np.swapaxes(weights, 0, 1)
                    weights = weights.reshape(-1)
                    ift_json["layers"][layer_id]["weights_format"] = "(bias), (units, input)"
                wFp.write(bias.tobytes(order='C'))
                wFp.write(weights.tobytes(order='C'))
                ift_json["layers"][layer_id]["weights_nbytes"] = 4 * (len(bias) + len(weights))

        # Max pooling layer
        if (isinstance(model.layers[i], tensorflow.keras.layers.MaxPool2D)):
            # Build IFT json data
            ift_json["layers"]["layer_" + str(ift_json["n_layers"])] = {
                "layer_type": "max_pooling_layer",
                "pool_size": model.layers[i].pool_size[0],
                "stride":  model.layers[i].strides[0],
            }

    if (save_weights):
      wFp.close()

    fp = open(sys.argv[2], 'w')
    json.dump(ift_json, fp, indent=2, separators=(',', ': '), sort_keys=True)
    fp.close()
    fim = time.time()
    total = fim-inicio
    print( "Execution time - ", fim-inicio," sec")
