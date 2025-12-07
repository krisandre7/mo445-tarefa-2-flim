import time
import sys

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D 
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.callbacks import Callback

import numpy as np
import json
import gc
from sklearn.metrics import log_loss, classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import tensorflow.keras.backend as K

from kerasVggUtils import GetNClassesFromCsv, LoadDatasetWithVggPreProc, GetVggInputShape

def load_target_names(value):
    saida = []
    for i in range(value):
        saida.append(str(i+1))
    return saida
	
class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]
 
		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")


class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery
 
	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)
 
		# return the learning rate
		return float(alpha)

class GridDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, decay=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.decay = decay
 
	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		#decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		self.initAlpha = self.initAlpha - decay
 
		# return the new learning rate
		return float(self.initAlpha)

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power
 
	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay
 
		# return the new learning rate
		return float(alpha)

class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))

def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          kernel_list = None,
          **kwargs):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    #input_shape = _obtain_input_shape(input_shape,
    #                                  default_size=224,
    #                                  min_size=32,
    #                                  data_format=backend.image_data_format(),
    #                                  require_flatten=include_top,
    #                                  weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(kernel_list[0], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)


    x = Conv2D(kernel_list[1], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)



    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2

    x = Conv2D(kernel_list[2], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)


    x = Conv2D(kernel_list[3], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)


    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3

    x = Conv2D(kernel_list[4], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)


    x = Conv2D(kernel_list[5], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)


    x = Conv2D(kernel_list[6], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4

    x = Conv2D(kernel_list[7], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    

    x = Conv2D(kernel_list[8], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)


    x = Conv2D(kernel_list[9], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)



    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5

    x = Conv2D(kernel_list[10], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
   

    x = Conv2D(kernel_list[11], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)

    x = Conv2D(kernel_list[12],(3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1a_x')(x)
        x = Dense(4096, activation='relu', name='fc2a_x')(x)
        x = Dense(classes, activation='softmax', name='predictions_x')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    #if input_tensor is not None:
    #    inputs = keras_utils.get_source_inputs(input_tensor)
    #else:
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


def vgg16_model(shape1, num_classes, weights_path, kernel_list):

  img_input = Input(shape=shape1)

  model = VGG16(include_top=False, weights=None, input_tensor=img_input, pooling=None, classes=num_classes, kernel_list = kernel_list)
  #model.load_weights(weights_path, by_name=True)
  #model.load_weights("/data/home/osaku/imagenet_models/keras_vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
  x = model.output

  x = Flatten(name='flatten_a')(x)
  x = Dense(4096, activation='relu', name='fc1kkk',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)
  x = Dense(4096, activation='relu', name='fc2kkk',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)

  x = Dense(num_classes, activation='softmax', name='predictionskkk')(x)
  model = Model(img_input, x)

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
  sgd = SGD(lr=1e-5, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  return model


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
      print("GPU is not available, pruning would take too long.")
      sys.exit(-1)

    if len(sys.argv) < 5:
        print("Usage: <train.csv> <val.csv> <current_net.h5> <output.h5>" );
        sys.exit(-1)

    t1  = time.time()

    trainPath = sys.argv[1]
    valPath  = sys.argv[2]
    current_net   = sys.argv[3]
    outputPath = sys.argv[4]

    ## Parameters to be set
    nClasses = GetNClassesFromCsv(trainPath)
    batch_size = 8
    nb_epoch = 50
  
    X_train, Y_train = LoadDatasetWithVggPreProc(trainPath) 
    X_teste, Y_teste = LoadDatasetWithVggPreProc(valPath)

    # Load our model
    model1 = load_model(current_net)
    kernels = []
    for i in range(len(model1.layers)):
      if 'conv' in model1.layers[i].name:
        layer_weights = model1.layers[i].get_weights()[0]  
        if len(layer_weights.shape)==4:
          kernels.append(layer_weights.shape[3]) 
    del model1
    K.clear_session()
    gc.collect()
    model = vgg16_model(GetVggInputShape(), num_classes=nClasses, weights_path = None, kernel_list = kernels)
    model.load_weights(current_net, by_name=True)
    model.summary()

    filename_weights = outputPath

    mcp = ModelCheckpoint(filename_weights, monitor="loss", mode="min",
                         save_best_only=True, save_weights_only=False)
    lr = PrintLearningRate()

    # Start Fine-tuning
    print("Finetune network")
    model.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=nb_epoch,
                verbose=2,
                callbacks=[mcp, lr])

    model.load_weights(filename_weights)


    target_names = load_target_names(value=nClasses)
       
    # Show layer's name
    for i in range(len(model.layers)):
      print (str(i) , model.layers[i].name)

    predictions_valid = model.predict(X_teste, batch_size=batch_size, verbose=2)

    Y_pred = np.argmax(predictions_valid, axis=-1)
    Y_test = np.argmax(Y_teste, axis=-1) # Convert one-hot to index

    print("Accuracy = ", accuracy_score(Y_test, Y_pred))

    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print("Kappa accuracy = ", cohen_kappa_score(Y_test, Y_pred))
    print( confusion_matrix(Y_test, Y_pred))
    t2 = time.time()
    total = t2-t1
    print( "Execution time - ", total, " sec")
    sec = total%60
    b =  total / 60
    hour = b / 60
    min = b % 60
    print (int(hour),":", int(min), ":", int(sec))

