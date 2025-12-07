#!/usr/bin/python

from os.path import *
import sys
import argparse as ap
import json
import os
import itertools
from operator import itemgetter
#import numpy as np

from multiprocessing import Process
from run_series_of_experiments import read_kfold_dir, read_config_file_dir, validate_number_list
from experiment import compute_seed_object_model_segmentation_output_dir

# Workaround to evaluate experiments in Phong, since I was having a problem with numpy on it
def my_mean(iterable): 
    return sum(iterable)/float(len(iterable))
     
def read_config_file_data(config_file_dir, config_file, fold_dir):
    if not exists(config_file):
        print 'Experiment configuration file %s does not exist!'
        return

    experiment_config_json = json.load(open(config_file, 'r'))
    
#    print experiment_config_json
    
    # Data for creating seeds
    train_file_basename           = experiment_config_json['train_file_basename']
    label_name                    = experiment_config_json['label_name']
    # We expect the seed estimation config file to be in the same folder as the experiment configuration file
    full_seed_estimation_config_file   = join(config_file_dir, experiment_config_json['seed_estimation_config_file'])
    experiment_id                 = experiment_config_json['experiment_id']

    # Verifying if the resume segmentation config json file exists
    if not exists(full_seed_estimation_config_file):
        print 'Please provide file %s!' % (full_seed_estimation_config_file)

        return None

    # Loading the json file to determine the resuming type, which is needed for our naming convention
    resuming_json = json.load(open(full_seed_estimation_config_file, 'r'))
    resuming_type = resuming_json['resuming_type'].lower()
    
    # Determining the ground truth and segmentation directories
    gt_dir = abspath(join(fold_dir, 'mapped_labels', label_name))
    segmentation_result_dir = compute_seed_object_model_segmentation_output_dir(fold_dir, label_name, experiment_id, resuming_type)

    output_score_file = join(segmentation_result_dir, 'scores.txt')

    return resuming_type, label_name,  segmentation_result_dir, gt_dir, output_score_file

def parse_scores_file(scores_filename):
    """
    We assume that the score files follow this estructure:
    000019.scn
    Dice:
    Mean: 0.778633
    Object 01: 0.778633
    Object 02: 0.778633
    ASSD:
    Mean: 0.895050
    Object 01: 0.895050
    Object 02: 0.895050

    000022.scn
    Dice:
    Mean: 0.895050
    Object 01: 0.895050
    Object 02: 0.895050
    ASSD:
    Mean: 0.895050
    Object 01: 0.895050
    Object 02: 0.895050

    Hence, we parse the scores  as above to separate them
    """
    scores_file = map(lambda x: x.strip(), open(scores_filename).readlines())
    img_scores = [list(x[1]) for x in itertools.groupby(scores_file, lambda x: x == '') if not x[0]]
   
    dice_scores = {}
    assd_scores = {}

    per_img_scores = {}

    for img in img_scores:
        prev_measure = ''
        per_img_scores[img[0]] = {}
            
        for f in img[1:]:
            if f.strip(':').lower() == 'dice':
                prev_measure = 'dice'
            elif f.strip(':').lower() == 'assd':
                prev_measure = 'assd'
            elif len(f.strip().split()) > 1:
                if prev_measure == 'dice':
                    obj_or_mean = float(f.split(':')[1])
                    k = f.split(':')[0] # Object name/label (Mean included) is the key
                    if k not in dice_scores:
                        dice_scores[k] = [obj_or_mean]
                    else:
                        dice_scores[k].append(obj_or_mean)

                    per_img_scores[img[0]]['Dice ' + k] = obj_or_mean

                elif prev_measure == 'assd':
                    obj_or_mean = float(f.split(':')[1])
                    k = f.split(':')[0] # Object name/label (Mean included) is the key
                    if k not in assd_scores:
                        assd_scores[k] = [obj_or_mean]
                    else:
                        assd_scores[k].append(obj_or_mean)
                    per_img_scores[img[0]]['ASSD ' + k] = obj_or_mean

    
    return dice_scores, assd_scores, per_img_scores
      
def evaluate_single_fold(config_file_dir, config_file, fold_dir, accuracy_type):
    _, _, segmentation_result_dir, gt_dir, output_score_filename = read_config_file_data(config_file_dir, config_file, fold_dir)
    
    cmd = 'python $NEWIFT_DIR/demo/Segmentation/segmentation_errors.py %s %s %d %s' % (segmentation_result_dir, gt_dir, accuracy_type, output_score_filename)
    print cmd
    os.system(cmd)

    dice_scores, assd_scores, per_img_scores = parse_scores_file(output_score_filename)

    print per_img_scores

    mean_dice = {}
    mean_assd = {}

    for k in dice_scores:
        mean_dice[k] = my_mean(dice_scores[k])
        
    for k in assd_scores:
        mean_assd[k] = my_mean(assd_scores[k])
    
    output_score_file = open(output_score_filename, 'a+')
    
    output_score_file.write('\nOverall Mean Dice:\n')
    for k in mean_dice:
        output_score_file.write('%s: %f\n' % (k, mean_dice[k]))

    if len(mean_assd) > 0:
        output_score_file.write('\nOverall Mean ASSD:\n')
        for k in mean_assd:
            output_score_file.write('%s: %f\n' % (k, mean_assd[k]))

    return mean_dice, mean_assd, per_img_scores

def evaluate_all_folds(kfold_dir, exp_id, config_file_dir, config_file,  accuracy_type, mean_dice_folds, mean_assd_folds, per_img_scores_folds):
    kfold_directories = read_kfold_dir(kfold_dir)

    if len(kfold_directories) <= 0:
        raise 'Please select a directory with valid k-folds'
    
    fold_dir = kfold_directories[0]
    resuming_type, label_name, _, _, _ = read_config_file_data(config_file_dir, config_file, fold_dir)

    if resuming_type is not None and label_name is not None:
        output_exp_dir = join(kfold_dir, 'results', label_name, resuming_type, '%03d' % exp_id)
        
        # Creating directory that will save the mean scores for all folds
        if not exists(output_exp_dir):
            os.makedirs(output_exp_dir)

        output_exp_score_file = open(join(output_exp_dir, 'scores.txt'), 'w')

        mean_dice_for_all_folds = {}
        mean_assd_for_all_folds = {}
        per_img_scores_for_all_folds = {}

        for fold in xrange(len(mean_dice_folds)):
            for obj in mean_dice_folds[fold]:
                if obj not in mean_dice_for_all_folds:
                    mean_dice_for_all_folds[obj] = [mean_dice_folds[fold][obj]]
                else:
                    mean_dice_for_all_folds[obj].append(mean_dice_folds[fold][obj])


            for obj in mean_assd_folds[fold]:
                if obj not in mean_assd_for_all_folds:
                    mean_assd_for_all_folds[obj] = [mean_assd_folds[fold][obj]]
                else:
                    mean_assd_for_all_folds[obj].append(mean_assd_folds[fold][obj])

            
        for obj in sorted(mean_dice_for_all_folds):
            output_exp_score_file.write('Mean Dice for %s: %lf\n' % (str(obj), my_mean(mean_dice_for_all_folds[obj])))

        for obj in sorted(mean_assd_for_all_folds):
            output_exp_score_file.write('Mean ASSD for %s: %lf\n' % (str(obj), my_mean(mean_assd_for_all_folds[obj])))

        

        for fold in xrange(len(per_img_scores_folds)):
            for img in per_img_scores_folds[fold]:
                if img not in per_img_scores_for_all_folds:
                    per_img_scores_for_all_folds[img] = {fold: per_img_scores_folds[fold][img]}
                else:
                    per_img_scores_for_all_folds[img][fold] = per_img_scores_folds[fold][img]

        print per_img_scores_for_all_folds
        print '\n'
        for img in sorted(per_img_scores_for_all_folds):
            print img, ':'
            for fold in sorted(per_img_scores_for_all_folds[img]):
                if 'ASSD Mean' in per_img_scores_for_all_folds[img][fold]:                    
                    print fold+1, per_img_scores_for_all_folds[img][fold]['ASSD Mean']
            print ''
    
def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--input_config_dir',type=str,required=True, help='Directory with series of experiment configuration json files')
    parser.add_argument('--kfold_dir', type=str, required=True, help='Directory containing the k-fold split in which each of the configuration files should be run') 
    parser.add_argument('--accuracy_type', required = True, type = int, help = 'Accuracy tipe (DICE - 0, 1 - ASSD, 2 - IFT_BOTH)')
    parser.add_argument('--specific_experiments', required = False, type = str, default = '', help='Evaluates only the specified experiments from specific configuration files. It may be \"1 3 8\" or \"3 ...\"to specify that experiments should start from 3 and go until the end')
    parser.add_argument('--specific_folds', required = False, type=str, default = '', help='Evaluates the specified folds (in case of error). It may be \"1 3 8\" or \"3 ...\"to specify that experiments should start from fold 3 and go until the end')
    args = vars(parser.parse_args())

    input_config_dir     = abspath(args['input_config_dir'])
    kfold_dir            = abspath(args['kfold_dir'])
    specific_folds       = args['specific_folds'].split() 
    specific_experiments = args['specific_experiments'].split()          
    accuracy_type        = args['accuracy_type']

    specific_folds = validate_number_list(specific_folds)
    specific_experiments = validate_number_list(specific_experiments)
    
    # Going through all experiment configuration directories/files
    config_file_directories = read_config_file_dir(input_config_dir)

    for config_file_dir in config_file_directories:
        config_file = abspath(join(config_file_dir, 'experiment_config.json'))

        print config_file

        if not exists(config_file):
            print 'File %s does not exist!' % config_file
            return
        
        # Getting experiment id
        exp_id = int(json.load(open(config_file, 'r'))['experiment_id'])

        if exp_id in specific_experiments:
            # processes = {}
            # Going through numeric experiment folders representing one out of k folds
            kfold_directories = read_kfold_dir(kfold_dir)

            mean_dice_folds = []
            mean_assd_folds = []
            per_img_scores_folds = []

            for fold_dir in kfold_directories:          
                if not exists(fold_dir):
                    print 'Directory %s does not exist!' % fold_dir
                    return
                
                if int(basename(fold_dir)) in specific_folds:
                    mean_dice, mean_assd, per_img_scores = evaluate_single_fold(config_file_dir, config_file, fold_dir, accuracy_type)
                    mean_dice_folds.append(mean_dice)
                    mean_assd_folds.append(mean_assd)
                    per_img_scores_folds.append(per_img_scores)

                    # p = Process(target = evaluate_single_fold, args=(config_file_dir, config_file, fold_dir, accuracy_type))
                    # processes[config_file_dir + '_' + fold_dir] = p
                    # p.start()


            # for p in processes:
            #    processes[p].join()

            evaluate_all_folds(kfold_dir, exp_id, config_file_dir, config_file, accuracy_type, mean_dice_folds, mean_assd_folds, per_img_scores_folds)

if __name__ == '__main__':
    main()
