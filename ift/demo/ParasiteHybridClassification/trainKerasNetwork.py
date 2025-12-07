import sys
import time

import tensorflow as tf

from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

import numpy as np


from kerasVggUtils import GetNClassesFromCsv, LoadDatasetWithVggPreProc, GetVggInputShape
	
class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))

def LoadVgg16Model(inputShape, nClasses, weightsPath):
  img_input = Input(shape=inputShape)

  model = vgg16.VGG16(include_top=False, weights=None, input_tensor=img_input, pooling=None, classes=nClasses)
  model.load_weights(weightsPath)

  x = model.output
  x = Flatten(name='flatten_a')(x)
  x = Dense(4096, activation='relu', name='fc1_a',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)
  x = Dense(4096, activation='relu', name='fc2_a',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)
  x = Dense(nClasses, activation='softmax', name='predictionsa')(x)
  model = Model(img_input, x)

  # Learning rate is changed to 0.001
  sgd = SGD(lr=1e-5,  momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  return model


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
      print("GPU is not available, training would take too long.")
      sys.exit(-1)

    if len(sys.argv) < 4:
        print("Usage: <train.csv> <imagenet_weights_notop.h5> <output.h5>");
        sys.exit(-1)


    t1 = time.time()

    trainPath = sys.argv[1]
    weightsPath = sys.argv[2]
    outputPath = sys.argv[3]

    nClasses = GetNClassesFromCsv(trainPath)
    nEpochs = 50
    batch_size = 8

    X_train, Y_train = LoadDatasetWithVggPreProc(trainPath) 

    model = LoadVgg16Model(GetVggInputShape(), nClasses, weightsPath)
    model.summary()

    mcp = ModelCheckpoint(outputPath, monitor="loss", mode="min",
            save_best_only=True, save_weights_only=False)
    lr = PrintLearningRate()

    print("Start fine-tuning")
    model.fit(
      X_train, Y_train,
      batch_size=batch_size,
      epochs=nEpochs,
      verbose=2,
      callbacks=[mcp, lr]
    )

    t2 = time.time()
    totalTime = t2 - t1
    print("Execution time - ", int(totalTime / 60), " min")

