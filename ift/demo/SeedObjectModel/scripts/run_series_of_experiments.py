#!/usr/bin/python

from os.path import *
import os
import json
import sys

import subprocess
import argparse as ap
from multiprocessing import Process

def run_experiment(config_file, exp_dir):
    cmd = 'python $NEWIFT_DIR/demo/SeedObjectModel/experiment.py --experiment_configuration %s --experiment_dir %s' % (config_file, exp_dir)
    print cmd
    os.system(cmd)

def read_kfold_dir(kfold_dir):
    return map(lambda y: join(kfold_dir, y), sorted(filter(lambda x: x.isdigit() and isdir(join(kfold_dir,x)), os.listdir(kfold_dir))))            

def read_config_file_dir(input_config_dir):
    return sorted(map(lambda x: join(input_config_dir, x), os.listdir(input_config_dir)))

def validate_number_list(specific_ids): 
    if len(specific_ids) > 0:
        if '...' in specific_ids:
            if specific_ids[-1] == '...' and len(specific_ids) == 2 and specific_ids[0].isdigit():
                specific_ids = xrange(int(specific_ids[0]), 1001) # Really large number as the last experiment id
            else:
                print 'Error: --specific_ids may be \"1 3 8\" or \"2 ...\" to specify that experiments should start from 2 and go until the end'
                return
        else: 
            specific_ids = map(int, specific_ids)
    else:
        specific_ids = xrange(1, 1001) # Really large number as the last experiment id

    return specific_ids

def main(): 
    parser = ap.ArgumentParser()
    parser.add_argument('--input_config_dir',type=str,required=True, help='Directory with series of experiment configuration json files')
    parser.add_argument('--kfold_dir', type=str, required=True, help='Directory containing the k-fold split in which each of the configuration files should be run')
    parser.add_argument('--specific_experiments', required = False, type = str, default = '', help='Runs only the specified experiments from specific configuration files. It may be \"1 3 8\" or \"3 ...\"to specify that experiments should start from 3 and go until the end')
    parser.add_argument('--specific_folds', required = False, type=str, default = '', help='Runs only the specified folds (in case of error). It may be \"1 3 8\" or \"3 ...\"to specify that experiments should start from fold 3 and go until the end')
    args = vars(parser.parse_args())

    input_config_dir = abspath(args['input_config_dir'])
    kfold_dir = abspath(args['kfold_dir'])
    specific_folds = args['specific_folds'].split()
    specific_experiments = args['specific_experiments'].split()

    specific_folds = validate_number_list(specific_folds)
    specific_experiments = validate_number_list(specific_experiments)
    
    # Going through all experiment configuration directories/files
    config_file_directories = read_config_file_dir(input_config_dir)

    for config_file_dir in config_file_directories:
        config_file = abspath(join(config_file_dir, 'experiment_config.json'))
        if not exists(config_file):
            print 'File %s does not exist!' % config_file
            return
        
        # Getting experiment id
        exp_id = int(json.load(open(config_file, 'r'))['experiment_id'])

        if exp_id in specific_experiments:
            processes = {}
            # Going through numeric experiment folders representing one out of k folds
            kfold_directories = read_kfold_dir(kfold_dir)
            for exp_dir in kfold_directories:          
                if not exists(exp_dir):
                    print 'Directory %s does not exist!' % exp_dir
                    return
                if int(basename(exp_dir)) in specific_folds:
                    p = Process(target = run_experiment, args=(config_file, exp_dir))
                    processes[config_file_dir + '_' + exp_dir] = p
                    p.start()


            for p in processes:
                processes[p].join()
   

if __name__ == "__main__":
    main()
