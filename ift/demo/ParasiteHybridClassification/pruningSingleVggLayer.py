import time
import sys

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import json
from sklearn.metrics import log_loss, classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import gc
import os

from kerasVggUtils import GetNClassesFromCsv, LoadDatasetWithVggPreProc, GetVggInputShape

def load_target_names(value):
    saida = []
    for i in range(value):
        saida.append(str(i+1))
    return saida

def regression_model(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          index = 1,
          pruningrate = 0.1,
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
    if index==2:
       x = Conv2D(64-int(64*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1',trainable=True)(img_input)
    else:
       x = Conv2D(64-int(64*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1',trainable=True)(img_input)

    if index>2:
      if index==3:
         x = Conv2D(64 - int(64*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2',trainable=True)(x)
      else:
         x = Conv2D(64 - int(64*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2',trainable=True)(x)
      
    else:
      if index==2:
         #print('block1_conv2: untrainable')
         x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2', trainable=True)(x)
         x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
         #x = Flatten(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-5,  momentum=0.9, nesterov=True)
         #sgd = Adam(lr=1e-5)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model


    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    if index>3:
      if index==4:
         x = Conv2D(128-int(128*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1',trainable=True)(x)
      else:
         x = Conv2D(128-int(128*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1',trainable=True)(x)

      #print('block2_conv1: untrainable')
    else:
      if index==3:
         x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1', trainable=True)(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-6,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model

    if index>4:
      if index==5:
         x = Conv2D(128 - int(128*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2',trainable=True)(x)
      else:
         x = Conv2D(128 - int(128*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2',trainable=True)(x)
      #print('block2_conv2: untrainable')
    else:
      if index==4:
         x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2', trainable=True)(x)
         x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-6,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model


    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    if index>5:
      if index==6:
         x = Conv2D(256 - int(256*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
      else:
         x = Conv2D(256 - int(256*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1', trainable=True)(x)
      #print('block3_conv1: untrainable')
    else:
      if index==5:
         x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1', trainable=True)(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-7,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model



    if index>6:
      if index==7:
         x = Conv2D(256 - int(256*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
      else:
         x = Conv2D(256 - int(256*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2', trainable=True)(x)
      #print('block3_conv2: untrainable')
    else:
      if index==6:
         x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2', trainable=True)(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-7,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model


    if index>7:
       if index==8:
          x = Conv2D(256 - int(256*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
       else:
          x = Conv2D(256 - int(256*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3', trainable=True)(x)
       #print('block3_conv3: untrainable')
    else:
      if index==7:
         x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3', trainable=True)(x)
         x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-7,  momentum=0.9, nesterov=True)
         #sgd = Adam(lr=1e-5)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    if index>8:
      if index==9:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
      else:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1', trainable=True)(x)
      #print('block4_conv1: untrainable')
    else:
      if index==8:
         x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1', trainable=True)(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-8,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model

    
    if index>9:  
      if index==10:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
      else:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2', trainable=True)(x)
      #print('block4_conv2: untrainable')
    else:
      if index==9:
         x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2', trainable=True)(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-8,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model

    if index>10:
      if index==11:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
      else:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3', trainable=True)(x)
      #print('block4_conv3: untrainable')
    else:
      if index==10:
        
         x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3', trainable=True)(x)
         x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-8,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model



    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    if index>11:
      if index==12:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)

      else:
         x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1', trainable=True)(x)

      #print('block5_conv1: untrainable')
    else:
      if index==11:
         x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1', trainable=True)(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-8,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model

   
    if index>12:
       if index==13:
          x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
       else:
          x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2', trainable=True)(x)

       #print('block5_conv2: untrainable')
    else:
      if index==12:
         x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2', trainable=True)(x)
         inputs = img_input
         # Create model.
         model = Model(inputs, x, name='vgg16')


         sgd = SGD(lr=1e-8,  momentum=0.9, nesterov=True)
         model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
         return model
    if index>13:
      x = Conv2D(512 - int(512*pruningrate), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
      #print('block5_conv3: untrainable')
    else:
      if index==13:
         x = Conv2D(512,(3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3', trainable=True)(x)
         x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
      else:
         x = Conv2D(512,(3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3', trainable=True)(x)
         x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
         x = Flatten(x)
         x = Dense(4096, activation='relu', name='fc1_a', trainable=True)(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    #if input_tensor is not None:
    #    inputs = keras_utils.get_source_inputs(input_tensor)
    #else:
      inputs = img_input
    # Create model.
      model = Model(inputs, x, name='vgg16')


      sgd = SGD(lr=1e-8,  momentum=0.9, nesterov=True)
      model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])

      return model
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1_a', trainable=True)(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')


    sgd = SGD(lr=1e-8,  momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])

    return model

def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          index = 1,
          num_drop = 1,
          kernel_list = [],
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
                         '(pre-training on 7ImageNet), '
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
    if index==1:

      x = Conv2D(64 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    else:
      x = Conv2D(kernel_list[0], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    if index==2:
      x = Conv2D(64 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    else:
      x = Conv2D(kernel_list[1], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)



    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    if index==3:
      x = Conv2D(128 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    else:
      x = Conv2D(kernel_list[2], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)

    if index==4:
      x = Conv2D(128 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    else:
      x = Conv2D(kernel_list[3], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)


    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    if index==5:
      x = Conv2D(256 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    else:
      x = Conv2D(kernel_list[4], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)

    if index==6:
      x = Conv2D(256 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    else:
      x = Conv2D(kernel_list[5], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    if index==7:
       x = Conv2D(256 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    else:
       x = Conv2D(kernel_list[6], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    if index==8:
      x = Conv2D(512 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    else:
      x = Conv2D(kernel_list[7], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    
    if index==9:  
      x = Conv2D(512 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    else:
      x = Conv2D(kernel_list[8], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)

    if index==10:
      x = Conv2D(512 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    else:
      x = Conv2D(kernel_list[9], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)



    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    if index==11:
      x = Conv2D(512 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    else:
      x = Conv2D(kernel_list[10], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
   
    if index==12:
       x = Conv2D(512 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    else:
       x = Conv2D(kernel_list[11], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    if index==13:
      x = Conv2D(512 - num_drop, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    else:
      x = Conv2D(kernel_list[12], (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
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





def vgg16_model(shape1, num_classes, index, num_drop, kernel_list = []):

  img_input = Input(shape=shape1)

  model = VGG16(include_top=False, weights=None, input_tensor=img_input, pooling=None, classes=num_classes, index = index, num_drop = num_drop, kernel_list = kernel_list)

  #model.load_weights("/data/home/osaku/imagenet_models/keras_vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
  x = model.output

  x = Flatten(name='flatten_a')(x)
  x = Dense(4096, activation='relu', name='fc1_a',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)
  x = Dense(4096, activation='relu', name='fc2_a',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)
  x = Dense(num_classes, activation='softmax', name='predictionsa')(x)
  model = Model(img_input, x)
  #model.add(Dense(num_classes, activation='softmax_final'))

  #weights_path = './weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


  #model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
  #model.layers.pop()
  #model.outputs = [model.layers[-1].output]
  #model.layers[-1].outbound_nodes = []
  #model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
  sgd = SGD(lr=1e-5,  momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  return model


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
      print("GPU is not available, pruning would take too long.")
      sys.exit(-1)

    if len(sys.argv) < 8:
        print("Usage: <train.csv> <val.csv> <layer (0,1...12)]> <pruning percent [0,1]> <baseline.h5> <current_net.h5> <output.h5>" );
        sys.exit(-1)

    t1 = time.time()

    trainPath = sys.argv[1] 
    valPath = sys.argv[2] 
    layer_idx = int(sys.argv[3])   ###{0...12}
    percent = float(sys.argv[4])  ## percentage of pruning. Ex: 0.5 for 50% of pruning
    baseline_net = sys.argv[5]  
    current_net = sys.argv[6] 
    output_path = sys.argv[7]

    nClasses = GetNClassesFromCsv(trainPath)
    batch_size = 8

    # Indexes for relevant layers
    layer_index = [1,2,4,5,7,8,9 ,11,12,13,15,16,17]
    next_index  = [2,4,5,7,8,9,11,12,13,15,16,17,20]
    output_index= [1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 20]
    fc = [20, 22, 24]
    layer = layer_index[layer_idx]
    if next_index[layer_idx] in fc:
          fc_changed = next_index[layer_idx]
          cv_changed = -1
    else:
          fc_changed = -1
          cv_changed = next_index[layer_idx]

    X_train, Y_train = LoadDatasetWithVggPreProc(trainPath)
    X_teste, Y_teste = LoadDatasetWithVggPreProc(valPath)
  
    print(current_net)
    model1 = load_model(current_net)
    kernels = []
    for i in range(len(model1.layers)):
        if 'conv' in model1.layers[i].name:
          layer_weights = model1.layers[i].get_weights()[0]  
          if len(layer_weights.shape)==4:
            kernels.append(layer_weights.shape[3]) 
            if i==layer:
               total_drop = int(layer_weights.shape[3]*percent)

    model1.summary()
    # Load our model
    drop_kernels = []
    drop = 1
    flag = True
    predictions_valid = model1.predict(X_train, batch_size=batch_size, verbose=2)

    Y_pred = np.argmax(predictions_valid, axis=-1)
    Y_test = np.argmax(Y_train, axis=-1) # Convert one-hot to index
    initial = cohen_kappa_score(Y_test, Y_pred)
    print("Accuracy = ", initial)

    model2 = vgg16_model(GetVggInputShape(), num_classes = nClasses, index = layer_idx+1, num_drop = drop, kernel_list = kernels)
    
    filename_weights = output_path  

    target_names = load_target_names(value=nClasses)

    for i in range(len(model1.layers)):
          if len(model1.layers[i].get_weights())> 0:  
                weights = model1.layers[i].get_weights()
                weights2 = model2.layers[i].get_weights()
                if weights[0].shape==weights2[0].shape:
                   model2.layers[i].set_weights(weights)
    
    weights = model1.layers[layer].get_weights()
    ker = weights[0].shape[3] 
    best_drop = 0
    best_accuracy = 0
    kernel_result = np.zeros(ker)
    kernel_acc = np.zeros(ker)
    for i in range(ker):
        index = 0
        weights = model1.layers[layer].get_weights()
        weights_2 = model2.layers[layer].get_weights() 
        ka = weights[0][:,:,:,i].copy()
        ka = ka.reshape(-1)
        if len(weights)==2:
           np.insert(ka, 0, weights[1][i])
        mean = np.mean(ka)
        mini = np.min(ka)
        maxi = np.max(ka)
        neg = np.count_nonzero(ka<=0)
        pos = np.count_nonzero(ka>0)
        for k in range(weights[0].shape[3]):
            if k !=i:
               weights_2[0][:,:,:,index] = weights[0][:,:,:,k].copy()
               index+=1

                #print(weights_2[0].shape)
        
        if len(weights)==2: 
           index = 0
           for k in range(weights[1].shape[0]):
                    if k !=i:
                       weights_2[1][index] = weights[1][k].copy()
                       index+=1 
        model2.layers[layer].set_weights(weights_2)
        if cv_changed>-1:
                weights = model1.layers[cv_changed].get_weights()
                weights_2 = model2.layers[cv_changed].get_weights() 

                index = 0
                for k in range(weights[0].shape[2]):
                 if k !=i:
                    weights_2[0][:,:,index, :] = weights[0][:,:,k,:].copy()
                    index+=1
                if len(weights)==2:
                   weights_2[1] = weights[1].copy()
                model2.layers[cv_changed].set_weights(weights_2)
        else:
                  weights = model1.layers[fc_changed].get_weights()
                  weights_2 = model2.layers[fc_changed].get_weights() 
                  shape = model1.layers[fc_changed-1].input_shape
                  index = 0
                  cont = 0
                  for j in range(shape[1]):
                    for k in range(shape[2]):
                      for l in range(shape[3]):
                        if l!=i:
                           weights_2[0][index] = weights[0][cont].copy()  
                           index+=1
                        cont+=1  
                  if len(weights)==2:
                    weights_2[1] = weights[1].copy()   
                  model2.layers[fc_changed].set_weights(weights_2)
        predictions_valid = model2.predict(X_train, batch_size=batch_size, verbose=2)

        Y_pred = np.argmax(predictions_valid, axis=-1)
        Y_test = np.argmax(Y_train, axis=-1) # Convert one-hot to index
        acc = cohen_kappa_score(Y_test, Y_pred)
        loss_1 = log_loss(Y_train, predictions_valid)
        print ("loss [",i,"]= ", loss_1)
        kernel_result[i] = loss_1
        kernel_acc[i] = 1-acc
    ke = np.argsort(kernel_result)
    drop_kernels = ke[:total_drop]
 
    #saving model
    model1 = load_model(current_net)
    model2 = vgg16_model(GetVggInputShape(), num_classes=nClasses, index = layer_idx+1, num_drop = len(drop_kernels), kernel_list = kernels)
    
    filename_weights = output_path

    target_names = load_target_names(value=nClasses)
    
    for i in range(len(model1.layers)):
          if len(model1.layers[i].get_weights())> 0:  
                weights = model1.layers[i].get_weights()
                weights2 = model2.layers[i].get_weights()
                if weights[0].shape==weights2[0].shape:
                   model2.layers[i].set_weights(weights)
    
    weights = model1.layers[layer].get_weights()
    ker = weights[0].shape[3] 
    index = 0
    weights = model1.layers[layer].get_weights()
    weights_2 = model2.layers[layer].get_weights() 
    for k in range(weights[0].shape[3]):
            if k not in drop_kernels:
               weights_2[0][:,:,:,index] = weights[0][:,:,:,k].copy()
               index+=1
              
    if len(weights)==2: 
         index = 0
         for k in range(weights[1].shape[0]):
                    if k not in drop_kernels:
                       weights_2[1][index] = weights[1][k].copy()
                       index+=1 
    model2.layers[layer].set_weights(weights_2)
    if cv_changed>-1:
                weights = model1.layers[cv_changed].get_weights()
                weights_2 = model2.layers[cv_changed].get_weights() 

                index = 0
                for k in range(weights[0].shape[2]):
                 if k not in drop_kernels:
                    weights_2[0][:,:,index, :] = weights[0][:,:,k,:].copy()
                    index+=1
                if len(weights)==2:
                   weights_2[1] = weights[1].copy()
                model2.layers[cv_changed].set_weights(weights_2)
    else:
                  weights = model1.layers[fc_changed].get_weights()
                  weights_2 = model2.layers[fc_changed].get_weights() 
                  print (weights[0].shape, weights_2[0].shape)
                  shape = model1.layers[fc_changed-1].input_shape
                  index = 0
                  cont = 0
                  print(shape)
                  for j in range(shape[1]):
                    for k in range(shape[2]):
                      for l in range(shape[3]):
                        if l not in drop_kernels:
                           weights_2[0][index] = weights[0][cont].copy()  
                           index+=1
                        cont+=1  
                  if len(weights)==2:
                    weights_2[1] = weights[1].copy()   
                  model2.layers[fc_changed].set_weights(weights_2)
    drop_kernels+=1
    kernels = []
    for i in range(len(drop_kernels)):
       kernels.append(int(drop_kernels[i]))
    
    print(filename_weights)

    print("saving model")
    model2.save(filename_weights)

    print("====Results1====")
    predictions_valid = model2.predict(X_teste, batch_size=batch_size, verbose=2)
    Y_pred = np.argmax(predictions_valid, axis=-1)
    Y_test = np.argmax(Y_teste, axis=-1) # Convert one-hot to index

    print("Accuracy = ", accuracy_score(Y_test, Y_pred))

    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print("Kappa accuracy = ", cohen_kappa_score(Y_test, Y_pred))
    print( confusion_matrix(Y_test, Y_pred))
    model4 = load_model(baseline_net)
    print(model4.layers[output_index[layer_idx+1]].name)
    intermediate_layer_model = Model(inputs=model4.input,
                                 outputs=model4.get_layer(model4.layers[output_index[layer_idx+1]].name).output)
    intermediate_output = intermediate_layer_model.predict(X_train, verbose=2)
    del model1
    del model4

    if layer_idx <= 12:
       del model2
       K.clear_session()
       gc.collect()

       print("Recovering activation map")
       img_input = Input(shape=GetVggInputShape())

       model3 = regression_model(include_top=False, weights=None, input_tensor=img_input, pooling=None, classes=nClasses, index = layer_idx + 2, pruningrate = percent)
       model3.summary()
       model3.load_weights(filename_weights, by_name=True)
       mcp = ModelCheckpoint("./tmp_pruning_net.h5", monitor="loss", mode="min",
                             save_best_only=True, save_weights_only=False)

       model3.fit(X_train, intermediate_output,
                   batch_size=batch_size,
                   epochs=40,
                   verbose=2, #,

                   )
       model3.save("./tmp_pruning_net.h5")
       del model3
       K.clear_session()
       gc.collect()
       model2 = load_model(filename_weights) 
       model2.load_weights("./tmp_pruning_net.h5", by_name=True)
       os.remove("./tmp_pruning_net.h5")

       model2.save(filename_weights)     



    print("====Results2====")
    predictions_valid = model2.predict(X_teste, batch_size=batch_size, verbose=2)
    Y_pred = np.argmax(predictions_valid, axis=-1)
    Y_test = np.argmax(Y_teste, axis=-1) # Convert one-hot to index

    print("Accuracy = ", accuracy_score(Y_test, Y_pred))

    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print("Kappa accuracy = ", cohen_kappa_score(Y_test, Y_pred))
    print( confusion_matrix(Y_test, Y_pred))
    t2 = time.time()
    total = t2 - t1
    print( "Execution time - ", total," sec")
    sec = total%60
    b =  total / 60
    hour = b / 60
    min = b % 60
    print (int(hour),":", int(min), ":", int(sec))



