#!/usr/bin/python2.7
import argparse
import numpy
import json
import pascal
import joblib
import random
from skimage import io
import os

import param_eval
from pyift import iftconv

def draw_region_list(p, path, regions):
    """Draws a list of regions using path as preffix"""
    for i,r in enumerate(regions):
        image = p.get_image(r.img_name)
        overlay = r.draw(image)
        io.imsave(path + '{0}.jpg'.format(i+1), overlay)

def filter_positive_regions(positive_regions):
    """Filter out positive regions that have high intersection score with other positive regions"""
    positive_regions = sorted(positive_regions, key = lambda x: x.confidence, reverse = True)
    
    eliminated = numpy.zeros(len(positive_regions), dtype = 'bool')
    
    for i,ri in enumerate(positive_regions[:-1]):
        if not eliminated[i]:
            for j,rj in enumerate(positive_regions[i+1:],i+1):
                if ri.img_name == rj.img_name:
                    intersection_area = ri.intersection_area(rj)
                    if intersection_area != 0 and min(ri.area(), rj.area())/intersection_area >= 0.5:
                        eliminated[j] = True
        
    return [r for i,r in enumerate(positive_regions) if not eliminated[i]]

def evaluate_detection_by_region(p, positive_class, validation_images, candidates, svm, convnet, frame_size, ratios, output_path = None):
    """Return the precision and recall curve for a given classifier and convolutional network on the images.
    
    Parameters
    ----------
    p: instance of pascal.Pascal
    positive_class: name of the positive class
    validation_images: list of validation images
    candidates: list of candidate windows (frames)
     
    svm: support vector machine classifier (sklearn.SVC)
    convnet: convolutional network (pyift.iftconv)
    
    frame_size: side of the input image (frame_size x frame_size) for the convolutional network
    output_path:  path in which to write the detected regions and precision and recall curve.
    """
    positive_regions = []
    
    for i,r in enumerate(candidates):
        r = r.reframe_by_ratios(ratios, p.get_image(r.img_name))
        
        if r:
            r.save_ift('tmp.pgm', frame_size)
            
            print("Extracting features from region {0}/{1}".format(i+1, len(candidates)))
            
            xi = convnet.transform('tmp.pgm')
            yi = int(svm.predict(xi))
    
            if yi == 1:
                r.confidence = numpy.abs(float(svm.decision_function(xi)))
                r.class_name = positive_class
                positive_regions.append(r)
    
    positive_regions = filter_positive_regions(positive_regions)
    
    precision,recall,thresholds = p.evaluate_detection(validation_images, positive_regions)
    
    if output_path:
        if not os.path.exists(os.path.join(output_path,'detected')):
            os.makedirs(os.path.join(output_path,'detected'))
            
        draw_region_list(p, os.path.join(output_path, 'detected/detected_'), positive_regions)    
            
        numpy.save(os.path.join(output_path,'precision.npy'), precision)
        numpy.save(os.path.join(output_path,'recall.npy'), recall)
        numpy.save(os.path.join(output_path,'thresholds.npy'), thresholds)
        
    return precision,recall,thresholds

def main():
    """Script used to evaluate the detection efficacy of a convolutional network and classifier on the PASCAL VOC validation images"""
    parser = argparse.ArgumentParser(description = 'Pascal detection evaluation.')
    parser.add_argument('parameters_file', help = 'Path to the parameters file. See extra/detection_params.txt for an example.')
    parser.add_argument('-o','--output', help = 'Path to the output file, containing the precision, recall and thresholds.', required = False)

    args = parser.parse_args()
    
    params = json.load(open(args.parameters_file))
    
    p = pascal.Pascal(params['pascal_path'])
    
    positive_class = params['positive_class']
    
    pos_validation = p.get_positive_images(p.validation_list, positive_class)
    random.shuffle(pos_validation)
    neg_validation = p.get_negative_images(p.validation_list,positive_class)
    random.shuffle(neg_validation) 
    
    pos_validation = pos_validation[:params['n_positive']]
    neg_validation = neg_validation[:params['n_negative']]
    
    validation_images = pos_validation + neg_validation
    
    svm = joblib.load(params['svm_path'])
    frame_size = params['frame_size']
    
    convnet = iftconv.IFTConvNetwork(path = str(params['convnet_path']))
    
    output_path = None
    if args.output:
        output_path = args.output
        
    windows_per_image = params['windows_per_image']
    candidates = p.get_regions_kde(validation_images, positive_class, windows_per_image)
        
    #FIXME: ratios
    ratios = [1.0]
    evaluate_detection_by_region(p, positive_class, validation_images, candidates, svm, convnet, frame_size, ratios, output_path)

if __name__ == '__main__':
    main()
