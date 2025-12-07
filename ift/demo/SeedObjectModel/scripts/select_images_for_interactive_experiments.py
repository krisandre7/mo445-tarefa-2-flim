#!/usr/bin/python

from os.path import *
import argparse as ap
import os, shutil

def main () :
    parser = ap.ArgumentParser()
    parser.add_argument('--selected_imgs', required = True, type = str)
    parser.add_argument('--kfold_dir', required = True, type=str)
    parser.add_argument('--output_dir', required = True, type=str)
    parser.add_argument('--basins_dir', required = True, type=str)
    parser.add_argument('--model_seeds_basepath', required = True, type=str)
    parser.add_argument('--label_name', required = True, type=str)
    args = vars(parser.parse_args())
    selected_imgs_file = args['selected_imgs']
    kfold_dir = abspath(args['kfold_dir'])
    output_dir = args['output_dir']
    basins_dir = args['basins_dir']
    model_seeds_basepath = args['model_seeds_basepath']
    label_name = args['label_name']

    selected_imgs = map(lambda x:x.strip(), open(selected_imgs_file,'r').readlines())

    prev_img = ''
    
    folds_imgs = {}
    #Filtering selected images
    for line in selected_imgs:
        if line.strip().endswith(':'):
            prev_img = line.strip().strip(':').strip()
        elif line.startswith('***'):
            fold = int(line.strip('***').split()[0])
            if fold not in folds_imgs:
                folds_imgs[fold] = [prev_img]
            else:
                folds_imgs[fold].append(prev_img)
            
    print folds_imgs
    
    #Copying images
    for fold in folds_imgs:
        fold_str = '%02d' % fold

        if not exists(join(output_dir, fold_str)):
            os.makedirs(join(output_dir, fold_str))
            
            #Copying model markers
            shutil.copy(abspath(join(kfold_dir, fold_str, model_seeds_basepath)), abspath(join(output_dir, fold_str, basename(model_seeds_basepath))))
            
        for img in folds_imgs[fold]:
            print fold, img
            shutil.copy(abspath(join(kfold_dir, fold_str, 'registered', img)), abspath(join(output_dir, fold_str, img)))
            
            if not exists(join(output_dir, fold_str, 'mapped_labels', label_name)):
                os.makedirs(join(output_dir, fold_str, 'mapped_labels', label_name))

            #Copying original registered image
            shutil.copy(abspath(join(kfold_dir, fold_str, 'registered', img)), abspath(join(output_dir, fold_str, img)))
            #Copying GT
            shutil.copy(abspath(join(kfold_dir, fold_str, 'mapped_labels', label_name, img)), abspath(join(output_dir, fold_str, 'mapped_labels', label_name, img)))
            
            basins_name = splitext(img)[0] + '_basins' +  splitext(img)[1]
            shutil.copy(abspath(join(kfold_dir, fold_str, basins_dir, basins_name)), abspath(join(output_dir, fold_str, basins_name)))
        
           
           
            
if __name__ == "__main__":
    main()
