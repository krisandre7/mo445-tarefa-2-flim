#!/usr/bin/python

import argparse as ap
import json
import os
import shutil
from os.path import *

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--input_all_imgs', required = True, help = 'File with all image paths')
    parser.add_argument('--input_train_labels', required = True, help = 'File or directory with training label images')
    #parser.add_argument('--input_label_dir', required = True, help = 'Parent folder containing all label directories that must be transformed by the deformation field')
    parser.add_argument('--output', required = True, help = 'Output dir')
    parser.add_argument('--score_function', required = False, type = str, default = "DICE", help = 'Score function (DICE, ASSD, RANDOM)')
    parser.add_argument('--image_depth', required = True, type = int)
    parser.add_argument('--affine_params', required = True, help = 'Elastix affine parameters file')
    parser.add_argument('--deformable_params', required = True, help = 'Elastix deformable parameters file')
    parser.add_argument('--num_threads', required = False, type = int, default = 1, help = 'Number of registration threads')
    parser.add_argument('--skip_existing', required = False, default = False, action='store_true',  help = 'If set, skips existing registration results.')

    args = vars(parser.parse_args())
    
    all_images_pathnames = abspath(args['input_all_imgs'])
    input_train_labels = abspath(args['input_train_labels'])
    #input_label_dir = abspath(args['input_label_dir'])
    score_function = args['score_function']
    output_dir = abspath(args['output'])
    image_depth = args['image_depth']
    aff_param_file = abspath(args['affine_params'])
    def_param_file = abspath(args['deformable_params'])
    num_threads    =args['num_threads']
    skip_existing  = args['skip_existing']


    reference_image_json_file = os.path.join(output_dir, 'reference-image.json')


    cmd = 'iftFindReferenceImage -i %s -o %s -m %s' % (input_train_labels, reference_image_json_file, score_function)

    print cmd
    os.system(cmd)

    reference_image = None

    with open(reference_image_json_file, 'r') as f:        
        reference_image_json  = json.load(f)
        reference_image = reference_image_json['reference-image']
        f.close()

    fixed_image = basename(reference_image).split('_')
    
    if isdir(all_images_pathnames):
        all_images_dir = all_images_pathnames
    else:
        with open(all_images_pathnames) as f:
            all_images_dir = dirname(f.readlines()[0].strip())
            f.close()

    fixed_image = join(all_images_dir, fixed_image[0] + '_' + fixed_image[1] + '_Orig' + splitext(reference_image)[1])
    print 'Reference label', reference_image, '\nCorresponding fixed image ', fixed_image

    output_registered_dir = os.path.join(output_dir, 'registered')
    cmd = 'iftRegisterImageSetByElastix -f %s -m %s -d %d -a %s -b %s -o %s -t %d' % (fixed_image, all_images_pathnames, image_depth, aff_param_file, def_param_file, output_registered_dir, num_threads)

    cmd = cmd + ' -s' if skip_existing else cmd

    print cmd
    os.system(cmd)

#    for label in os.listdir(input_label_dir):
#        output_mapped_label_dir = os.path.join(output_dir, 'mapped_labels', label)
#        if not os.path.exists(output_mapped_label_dir):
#            os.makedirs(output_mapped_label_dir)

#        cmd = 'python $NEWIFT_DIR/demo/Registration/apply_transformix.py %s %s %s' % (os.path.join(input_label_dir, label), output_registered_dir, output_mapped_label_dir)

#        # Copying the reference image's label
#        shutil.copy(os.path.join(input_label_dir, label, os.path.basename(reference_image)), os.path.join(output_mapped_label_dir, os.path.basename(reference_image)))

#        print cmd
#        os.system(cmd)

    

if __name__ == "__main__":
    main()
