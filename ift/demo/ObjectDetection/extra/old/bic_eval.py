#!/usr/bin/python2.7
"""
Dependencies: 
    pyift, which can be generated from pyift.i by executing the Makefile contained in this folder. Check the Makefile to ensure that the correct python library and include paths are set.
    numpy,
    sklearn (http://scikit-learn.org/stable/), already available in most Linux distributions.

"""
import sys
import os
import argparse
import re

import numpy
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
import json

import pyift

class BICDescriptor():
    def __init__(self, bins_per_band):
        self.bins_per_band = bins_per_band
        
    def transform(self, image_path):
        return pyift.iftPyExtractBICFeatures(image_path, self.bins_per_band)
        
def split_dataset(path, fraction, class_expression = '[0-9]+'):
    #Loads images and their classes
    images = [f for f in os.listdir(path) if f.endswith(".pgm")]
    p = re.compile(class_expression)
    classes = numpy.array([ int(p.match(name).group()) for name in images])
    images = numpy.array([ os.path.join(path,f) for f in images])
    
    #Splits the dataset into two pairs of images and classes
    if fraction < 1.0:
        indices = list(StratifiedShuffleSplit(classes,1,train_size = fraction))[0]
        
        first_set = images[indices[0]], classes[indices[0]]
        second_set = images[indices[1]], classes[indices[1]]
    else:
        first_set = (images,classes)
        second_set = ([], [])
    
    #Returns the two pairs of images and classes
    return first_set, second_set

def load_dataset(args):
    train, val, test = None, None, None
    
    if args.testing_datapath:
        #Testing subset
        (test, _ ) = split_dataset(args.testing_datapath, args.fraction_testing)
    
    if args.validation_datapath:
        #Training and validation images are in separate folders
        (train, _ ) = split_dataset(args.training_datapath, args.fraction_training)
        (val, _ ) = split_dataset(args.validation_datapath, args.fraction_validation)
    else:
        #Training and validation images are in the same folder
        assert(args.fraction_training + args.fraction_validation == 1.0)
        (train, val) = split_dataset(args.training_datapath, args.fraction_training)
    
    return train, val, test
   
def main():
    parser = argparse.ArgumentParser(description = 'Bic parameter evaluation.')
    
    parser.add_argument('fraction_training', help = 'Fraction of the training data to be used.', type = float)
    parser.add_argument('fraction_validation', help = 'Fraction of the validation data to be used.', type = float)
    parser.add_argument('fraction_testing', help = 'Fraction of the testing data to be used', type = float)
    parser.add_argument('training_datapath', help = 'Path to the training data')
    
    parser.add_argument('-v','--validation_datapath', help = 'Path to the validation data')
    parser.add_argument('-s','--testing_datapath', help = 'Path to the testing data')
    
    args = parser.parse_args()
    
    #Loads train, val and test
    train, val, test = load_dataset(args)
    train_images, ytrain = train[0], train[1]
    val_images, yval = val[0], val[1]
    if test:
        test_images, ytest = test[0], test[1]
    
    #Iterates over all architectures
    bic = BICDescriptor(4)
    print("Extracting features from training image 1/{0}".format(len(train_images)))
    
    #Using the first image_path to discover the feature space dimension
    first_output = bic.transform(train_images[0])
    feature_dimension = len(first_output)
    
    Xtrain = numpy.zeros((len(train_images), feature_dimension))
    Xtrain[0] = first_output
    
    #Extracts attributes for all training images
    for (i, img_name) in enumerate(train_images[1:], 1):
        print("Extracting features from training image {0}/{1}".format( i+1, len(train_images)))
        Xtrain[i] = bic.transform(img_name)
        
    Xval = numpy.zeros( (len(val_images), feature_dimension))
    #Extracts attributes for all validation images
    for (i, img_name) in enumerate(val_images):
        print("Extracting features from validation image {0}/{1}".format( i+1, len(val_images)))
        Xval[i] = bic.transform(img_name)
    
    #Trains and tests svm
    knn = KNeighborsClassifier(n_neighbors = 1, metric = 'l1')
    knn.fit(Xtrain, ytrain)
    acc = knn.score(Xval, yval)
    
    print('Validation accuracy: {0}'.format(acc))
    
    print("Best validation accuracy: {0}. \nParameters:\n {1}".format(best_acc,best_convnet))
            
    if test:
        Xtest = numpy.zeros( ((len(test_images), best_dimension)) )
        
        for (i, img_name) in enumerate(test_images):
            print("Extracting features from testing image {0}/{1}".format( i+1, len(test_images)))
            Xtest[i] = bic.transform(img_name)
            
        acc = knn.score(Xtest,ytest)
        
        print("Testing accuracy: {0}. \nParameters:\n {1}".format(acc,best_convnet))
    
if __name__ == "__main__":
    main()
