#!/usr/bin/python
from os.path import *
import os
import json
import sys

import argparse as ap

# Computes the output directory name for the resuming seeds by defining a standard naming convention
def compute_resuming_output_dir(experiment_dir, label_name, resuming_type, output_dir):
    return abspath(join(experiment_dir, label_name, 'models', resuming_type, output_dir, 'markers'))

def compute_seeds(experiment_dir, label_name, train_file_basename, full_seed_estimation_config_file, output_dir):
    # Computing input/output filenames and directories following a pre-defined convention
    full_training_file = abspath(join(experiment_dir, train_file_basename))
    full_label_name = abspath(join(experiment_dir, 'mapped_labels', label_name))

    # Verifying if the resume segmentation config json file exists
    if not exists(full_seed_estimation_config_file):
        print 'Please provide file %s!' % (full_seed_estimation_config_file)

        return None

    # Loading the json file to determine the resuming type, which is needed for our naming convention
    resuming_json = json.load(open(full_seed_estimation_config_file, 'r'))
    resuming_type = resuming_json['resuming_type'].lower()

    full_output_dir = compute_resuming_output_dir(experiment_dir, label_name, resuming_type, output_dir)

    cmd = 'iftResumeImageSetSegmentation %s %s %s %s' % (full_training_file, full_label_name, full_seed_estimation_config_file, full_output_dir)

    print cmd,'\n'
    os.system(cmd)

    return resuming_type # Returning the resuming type for the next functions

# Computes the output directory name for the seed object model by defining a standard naming convention
def compute_seed_object_model_output_dir(experiment_dir, label_name, label_input_basedir, resuming_type):
    return abspath(join(experiment_dir, label_name, 'models', resuming_type, label_input_basedir, 'model'))

def create_seed_object_model(experiment_dir, label_name, label_input_basedir, resuming_type, seed_object_model_config_file, segmentation_config_file, model_output_basename):
    # Computing input/output filenames and directories following a pre-defined convention
    full_label_input_dir               = abspath(compute_resuming_output_dir(experiment_dir, label_name, resuming_type, label_input_basedir))
    full_reference_image_file          = abspath(join(experiment_dir, 'reference_image.json'))
    full_original_image_dir            = abspath(join(experiment_dir, 'registered'))
    full_label_name                    = abspath(join(experiment_dir, 'mapped_labels', label_name))

    full_seed_object_model_config_file = abspath(join(experiment_dir, seed_object_model_config_file))

    full_segmentation_config_file = join(compute_resuming_output_dir(experiment_dir, label_name, resuming_type, label_input_basedir), basename(segmentation_config_file))

    # Verifying if the segmentation config json file exists
    if not exists(full_segmentation_config_file):
        print 'Please estimate resuming seeds first to create file %s!' % (full_segmentation_config_file)

        return None

    # Issuing command to create the seed object model
    full_output_dir                    = join(compute_seed_object_model_output_dir(experiment_dir, label_name, label_input_basedir, resuming_type), model_output_basename)

    cmd = 'iftCreateSeedObjectModel %s %s %s %s %s %s %s' % (full_reference_image_file, full_label_input_dir, full_original_image_dir, full_label_name, full_seed_object_model_config_file, full_segmentation_config_file, full_output_dir)

    print cmd,'\n'
    os.system(cmd)

# Computes the output directory name for the seed object model by defining a standard naming convention
def compute_seed_object_model_segmentation_output_dir(experiment_dir, label_name, label_input_basedir, resuming_type):
    return abspath(join(experiment_dir, label_name, 'models', resuming_type, label_input_basedir, 'results'))

def test_seed_object_model_without_search(experiment_dir, label_name, label_input_basedir, resuming_type, segmentation_config_file, model_basename, test_file_basename):
    full_test_image_file = abspath(join(experiment_dir, test_file_basename))
    full_label_name      = abspath(join(experiment_dir, 'mapped_labels', label_name))
    full_segmentation_config_file = join(compute_resuming_output_dir(experiment_dir, label_name, resuming_type, label_input_basedir), basename(segmentation_config_file))

    full_model_basename  = join(compute_seed_object_model_output_dir(experiment_dir, label_name, label_input_basedir, resuming_type), model_basename)

    full_output_dir  = compute_seed_object_model_segmentation_output_dir(experiment_dir, label_name, label_input_basedir, resuming_type)

    cmd = 'iftSegmentImageSetWithSeedObjectModelAndNoSearch %s %s %s %s %s' % (full_test_image_file, full_label_name, full_model_basename, full_segmentation_config_file, full_output_dir)

    print cmd, '\n'
    os.system(cmd)

def get_resuming_type_if_undefined(args, resuming_type):
    # if resuming_type is None, then the user does not want the seeds to be recomputed because 'compute_seeds' returns
    # the name that was used. Hence, the user must provide resuming_type manually
    if resuming_type is None:
        resuming_type = args['resuming_type']

        if resuming_type is None:
            print 'Please run the resuming algorithm to compute seeds or provide the resuming type via command line (check --help)'
            return None

    return resuming_type

def main ():

    parser = ap.ArgumentParser()
    parser.add_argument('--experiment_configuration',type=str,required=True, help='Experiment configuration file in JSON format')
    parser.add_argument('--experiment_dir', type=str, required=True, help = 'Experiment directory')
    parser.add_argument('--resuming_type',type=str, default = None, required=False, help='Resuming type for estimating object seeds. This MUST be passed when the resuming seeds are not to be computed.')
    do_compute_seeds = False

    args = vars(parser.parse_args())

    experiment_dir = abspath(args['experiment_dir'])
    config_file = abspath(args['experiment_configuration'])
    config_file_dir = dirname(config_file)


    if not exists(config_file):
        print 'Experiment configuration file %s does not exist!'
        return
        
    
    experiment_config_json = json.load(open(config_file, 'r'))
    
    print experiment_config_json
    
    # Data for creating seeds
    train_file_basename           = experiment_config_json['train_file_basename']
    label_name                    = experiment_config_json['label_name']
    # We expect the seed estimation config file to be in the same folder as the experiment configuration file
    seed_estimation_config_file   = join(config_file_dir, experiment_config_json['seed_estimation_config_file'])
    experiment_id                 = experiment_config_json['experiment_id']
    DO_COMPUTE_SEEDS              = experiment_config_json['COMPUTE_SEEDS']
    
    # Data for creating the model
    # We expect the model config file to be in the same folder as the experiment configuration file
    model_config_file             = join(config_file_dir, experiment_config_json['model_config_file'])
    model_output_basename         = experiment_config_json['model_output_basename']
    DO_CREATE_SEED_OBJECT_MODEL   = experiment_config_json['CREATE_SEED_OBJECT_MODEL']

    test_file_basename            = experiment_config_json['test_file_basename']
    DO_TEST_SEGMENTATION          = experiment_config_json['TEST_SEGMENTATION']

    resuming_type = None

    if DO_COMPUTE_SEEDS:
        # Storing the resuming type in a variable so it may be used to compute the filenames and directories according
        # to our convention
        resuming_type = compute_seeds(experiment_dir, label_name, train_file_basename, seed_estimation_config_file, experiment_id)

    if DO_CREATE_SEED_OBJECT_MODEL:
        resuming_type = get_resuming_type_if_undefined(args, resuming_type)

        if resuming_type is None:
            return

        create_seed_object_model(experiment_dir, label_name, experiment_id, resuming_type, model_config_file, seed_estimation_config_file, model_output_basename)

    if DO_TEST_SEGMENTATION:
        resuming_type = get_resuming_type_if_undefined(args, resuming_type)

        if resuming_type is None:
            return

        test_seed_object_model_without_search(experiment_dir, label_name, experiment_id, resuming_type, seed_estimation_config_file, model_output_basename, test_file_basename)


if __name__ == "__main__":
    main()
