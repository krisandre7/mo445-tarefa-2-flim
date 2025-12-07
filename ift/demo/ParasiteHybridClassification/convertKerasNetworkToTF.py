
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import Model, load_model
import numpy as np

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usage: <keras_model.h5> <path_to_new_TF_model_folder>")
        sys.exit(-1)

    model = load_model(sys.argv[1])
    outputPath = sys.argv[2]

    model.summary()

    # Freezing graph according to following guide
    #   https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
    
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
      x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
            logdir=outputPath,
            name="frozen_model.pb",
            as_text=False)

    #model.save(outputPath)


