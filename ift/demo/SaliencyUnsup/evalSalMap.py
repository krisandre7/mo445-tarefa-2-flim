import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import os
import sys
import math
import multiprocessing
import scipy
from joblib import Parallel, delayed
from scipy import misc, ndimage
from sklearn import metrics
from scipy.interpolate import spline

def computeFbwScore(gt_path, img_path):
    beta = 1.0;

    # Read images
    sm_img = misc.imread(img_path, "L");
    gt_img = misc.imread(gt_path, "L");

    gt_img = (gt_img[:,:] - np.min(gt_img[:,:]))/(np.max(gt_img[:,:]) - np.min(gt_img[:,:]));
    sm_img = (sm_img[:,:] - np.min(sm_img[:,:]))/(np.max(sm_img[:,:]) - np.min(sm_img[:,:]));

    gt_mask = np.array(gt_img, dtype=np.bool);
    not_gt_mask = np.logical_not(gt_img)
    gt = np.array(gt_img, dtype=sm_img.dtype)

    E = np.abs(sm_img - gt)
    dist, idx = scipy.ndimage.morphology.distance_transform_edt(not_gt_mask, return_indices=True)

    # Pixel dependency
    Et = np.array(E)
    # To deal correctly with the edges of the foreground region:
    Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]
    sigma = 5.0
    EA = scipy.ndimage.gaussian_filter(Et, sigma=sigma, truncate=3 / sigma,
                                       mode='constant', cval=0.0)
    min_E_EA = np.minimum(E, EA, where=gt_mask, out=np.array(E))

    # Pixel importance
    B = np.ones(gt.shape)
    B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])
    Ew = min_E_EA * B

    # Final metric computation
    eps = np.spacing(1)
    TPw = np.sum(gt) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])
    R = 1 - np.mean(Ew[gt_mask])  # Weighed Recall
    P = TPw / (eps + TPw + FPw)  # Weighted Precision

    Q = (1 + beta**2) * (R * P) / (eps + R + (beta * P))

    return Q

def computeImageScores(gt_path, img_path):
  # Read images
  sm_img = misc.imread(img_path, "L").flatten();
  gt_img = misc.imread(gt_path, "L").flatten();

  # Normalize {0,1}
  gt_img = (gt_img - min(gt_img))/(max(gt_img) - min(gt_img));
  inv_gt_img = 1.0 - gt_img;

  # Get possible thresh values
  thresh_vals = np.arange(0,256,1);
  num_threshs = thresh_vals.size;

  # Init lists of values
  img_recall = np.zeros(num_threshs); 
  img_false_alarm = np.zeros(num_threshs);
  img_precision = np.zeros(num_threshs);

  # For each threshold
  for i in range(num_threshs):
    # Init binary image
    bin_img = np.zeros(sm_img.size);

    # Apply threshold
    t = thresh_vals[i];
    bin_img[sm_img >= t] = 1.0;
    bin_img[sm_img < t] = 0.0;

    # Get inverted images for
    inv_bin_img = 1.0 - bin_img
    
    # Get basic evaluation metrics
    tp_i = bin_img[gt_img == bin_img].sum()
    tn_i = inv_bin_img[inv_gt_img == inv_bin_img].sum();
    fp_i = bin_img[inv_gt_img == bin_img].sum()
    fn_i = inv_bin_img[gt_img == inv_bin_img].sum();
      
    # Get more complex evaluation metrics
    recall_i = tp_i/(tp_i + fn_i + 0.0001);
    false_alarm_i = fp_i/(tn_i + fp_i + 0.0001);
    precision_i = tp_i/(tp_i + fp_i + 0.0001);

    # Append values to list
    img_false_alarm[i] = false_alarm_i;
    img_recall[i] = recall_i;
    img_precision[i] = precision_i;

  return img_precision, img_recall, img_false_alarm;

def computeDatasetScores(gt_dir_path, img_dir_path, ds_name):
  ds_overall_result_fp = open(ds_name + "_overall_results.csv", "w");
  ds_img_result_fp = open(ds_name + "_img_results.csv", "w");

  ds_overall_result_fp.write("fbw,auc,fm,precision,recall\n");
  ds_img_result_fp.write("id,fbw,auc,fm,precision,recall\n");

  img_files = glob.glob(img_dir_path + "/*");
  gt_files = glob.glob(gt_dir_path + "/*");

  if(len(img_files) == 0 or len(gt_files) == 0):
    print "ERROR! The directory is empty";
    exit();

  if(len(img_files) != len(gt_files)):
    print "ERROR! The amount of files differ!";
    exit();
  
  num_files = len(img_files);

  auc = 0.0;
  fmeasure = 0.0;
  fbw = 0.0;

  # Compute the step values in the chart
  num_points = 25;
  step = 1.0 / num_points;
    
  # Init the values for the mean computation
  roc_recall_sum = np.zeros(num_points + 1);
  roc_false_alarm_vals = np.arange(0.0, 1.0 + step, step);
  fm_precision_sum = np.zeros(num_points + 1);
  fm_recall_vals = np.arange(0.0, 1.0 + step, step);
  roc_recall_count = np.zeros(num_points + 1);
  fm_precision_count = np.zeros(num_points + 1);

  print "Evaluating each saliency map";
  for i in range(num_files):
    # # Get image path names
    img_basename = os.path.basename(img_files[i]);
    img_filename = os.path.splitext(img_basename)[0];

    print "\tProgress: " + str(i) + "/" + str(num_files);

    img_precision, img_recall, img_false_alarm = computeImageScores(gt_files[i], img_files[i]);
    img_fbw = computeFbwScore(gt_files[i], img_files[i]);

    img_fm = (2 * img_precision * img_recall)/(img_precision + img_recall + 0.0001);
    img_auc = metrics.auc(img_false_alarm, img_recall);

    img_max_fm_index = np.argmax(img_fm);
    
    result_text = img_filename + "," + str(img_fbw) + "," + str(img_auc) + "," + str(img_fm[img_max_fm_index]) + "," + str(img_precision[img_max_fm_index]) + "," + str(img_recall[img_max_fm_index]) + "\n";

    ds_img_result_fp.write(result_text);

    fbw = fbw + img_fbw;

    # Interpolate the values to the respective bins in the chart
    for j in range(img_recall.size):
      img_precision_j = img_precision[j];
      img_recall_j = img_recall[j];
      img_false_alarm_j = img_false_alarm[j];

      # Get the respective bin
      index_recall = np.int(round(img_recall_j/step));
      index_false_alarm = np.int(round(img_false_alarm_j/step));

      # Sum the values indicated by the bin
      roc_recall_sum[index_false_alarm] = roc_recall_sum[index_false_alarm] + img_recall_j;
      roc_recall_count[index_false_alarm] = roc_recall_count[index_false_alarm] + 1;
      fm_precision_sum[index_recall] = fm_precision_sum[index_recall] + img_precision_j;
      fm_precision_count[index_recall] = fm_precision_count[index_recall] + 1;

  #Lists for the plotting
  roc_recall_plot = []
  roc_false_alarm_plot = []
  fm_precision_plot = []
  fm_recall_plot = []

  # Simplify the data for the plotting by avoiding 'non-visited' values
  print "Preparing the data for plotting";
  for k in range(num_points + 1):
    if(roc_recall_count[k] > 0):
      roc_recall_plot.append(roc_recall_sum[k] / roc_recall_count[k]);
      roc_false_alarm_plot.append(k * step); # Every value in the X axis is visited once

    if(fm_precision_count[k] > 0):
      fm_precision_plot.append(fm_precision_sum[k] / fm_precision_count[k]);
      fm_recall_plot.append(k * step); # Every value in the X axis is visited once

  # Get numpy representation of the lists
  fm_xaxis = np.array(fm_recall_plot);
  fm_yaxis = np.array(fm_precision_plot);
  roc_xaxis = np.array(roc_false_alarm_plot);
  roc_yaxis = np.array(roc_recall_plot);

  # Get perfomance values
  auc = metrics.auc(roc_xaxis, roc_yaxis);
  fm = (2.0 * fm_xaxis * fm_yaxis) / (fm_xaxis + fm_yaxis);
  fbw = fbw/num_files;

  max_fm_index = np.argmax(fm);

  # Plot the charts and save in the disk
  print "Writing result images";
  plt.figure(1);
  plt.xlabel('Recall');
  plt.ylabel('Precision');
  plt.grid(True);
  plt.plot(fm_xaxis, fm_yaxis, lw = 2);
  plt.ylim(0.0, 1.05);
  plt.xlim(0.0, 1.05);
  plt.autoscale(enable = False, axis = "both");
  plt.savefig(ds_name + "_fmeasure_lines.png");
  
  plt.figure(2);
  plt.xlabel('False-Alarm');
  plt.ylabel('Recall');
  plt.grid(True);
  plt.plot(roc_xaxis, roc_yaxis, lw = 2);
  plt.ylim(0.0, 1.05);
  plt.xlim(-0.05, 1.05);
  plt.autoscale(enable = False, axis = "both");
  plt.savefig(ds_name + "_roc.png");

  plt.figure(3);
  plt.bar(0.25, fm_yaxis[max_fm_index], color = "b", width = 0.125);
  plt.bar(0.5, fm_xaxis[max_fm_index], color = "g", width = 0.125);
  plt.bar(0.75, fm[max_fm_index], color = "r", width = 0.125);
  plt.xticks([0.25, 0.5, 0.75], ["Precision", "Recall", "F-measure"]);
  plt.ylim(0.0, 1.05);
  plt.xlim(-0.05, 1.05);
  plt.autoscale(enable = False, axis = "both");
  plt.savefig(ds_name + "_fmeasure_bars.png");
  
  result_text = str(fbw) + "," + str(auc) + "," + str(fm[max_fm_index]) + "," + str(fm_yaxis[max_fm_index]) + "," + str(fm_xaxis[max_fm_index]) + "\n";

  ds_overall_result_fp.write(result_text);

  ds_overall_result_fp.close();
  ds_img_result_fp.close();


  # Print non-plotted data
  print "\nAlgorithm performance on Dataset";
  print "-----------------------------------------------";
  print "Area Under The Curve: " + str(auc);
  print "Mean Weighted F-measure: " + str(fbw);
  print "Best F-measure: " + str(fm[max_fm_index]);
  print "Best F-measure Precision: " + str(fm_yaxis[max_fm_index]);
  print "Best F-measure Recall: " + str(fm_xaxis[max_fm_index]);
  print "-----------------------------------------------\n";
#--------------------------------------------------------------------------------------------------

if(len(sys.argv) != 4):
  print "\nUsage: evalSalMap [1] [2] [3]";
  print "----------------------------------";
  print "[1] - Ground-truth directory path";
  print "[2] - Saliency map directory path\n";
  print "[3] - Directory name\n";
  print "ERROR! Too many/few arguments!\n";
  exit();
else:
  computeDatasetScores(sys.argv[1], sys.argv[2], sys.argv[3]);

