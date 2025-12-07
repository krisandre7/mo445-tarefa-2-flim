#!/usr/bin/python2.7
import sys
import os
import argparse
import numpy as np

from cyft import cyft

def read_superpixel_dataset(dataset_path):
    datasets = [ f for f in sorted(os.listdir(os.path.join(dataset_path, "orig/datasets")))]
    
    Xs, ys = [], []
    
    for dataset in datasets:
        Xi, yi = cyft.dataset_to_Xy(os.path.join(dataset_path, "orig/datasets",dataset))
        
        Xs.append(Xi)
        ys.append(yi)
        
    return np.concatenate(Xs), np.concatenate(ys)

def main():
    parser = argparse.ArgumentParser(description = 'Superpixel classification evaluation.')
    parser.add_argument('dataset_path', help = 'Path to the melanoma images dataset')
    
    args = parser.parse_args()
    
    X,y= read_superpixel_dataset(args.dataset_path)

if __name__ == "__main__":
    main()