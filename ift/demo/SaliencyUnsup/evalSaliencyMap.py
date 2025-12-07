#--------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import os
import sys
import math
import scipy
from joblib import Parallel, delayed
from scipy import misc, ndimage
from sklearn import metrics
#--------------------------------------------------------------------------------------------------
THRESH_VALS = np.arange(0,256,1);
NUM_THRESHS = THRESH_VALS.size;
NUM_INTERP_PTS = 10;
BETA = 1;
EPS = 0.0000001;
#--------------------------------------------------------------------------------------------------
def computePrecRecallCurves(tp, fp, fn):
  yaxis_interp = [];
  xaxis_interp = [];

  precision = tp/(tp + fp + EPS);
  recall = tp/(tp + fn + EPS);
  
  idx = np.argsort(recall);
  
  pr_xaxis = np.zeros(idx.size + 1);
  pr_yaxis = np.zeros(idx.size + 1);
  true_positives = np.zeros(idx.size + 1);
  false_positives = np.zeros(idx.size + 1);
  
  for i in range(1,idx.size + 1):
    pr_xaxis[i] = recall[idx[i - 1]];
    pr_yaxis[i] = precision[idx[i - 1]];
  
  pr_xaxis[0] = 0.0
  pr_yaxis[0] = pr_yaxis[1]
  
  for i in range(pr_xaxis.size - 1):
    x_i = np.linspace(pr_xaxis[i], pr_xaxis[i+1], NUM_INTERP_PTS, endpoint = False);
    y_i = np.linspace(pr_yaxis[i], pr_yaxis[i+1], NUM_INTERP_PTS, endpoint = False);
    
    for k in range(NUM_INTERP_PTS):
      xaxis_interp.append(x_i[k]);
      yaxis_interp.append(y_i[k]);
  
  xaxis_interp.append(pr_xaxis[pr_xaxis.size - 1]);
  yaxis_interp.append(pr_yaxis[pr_yaxis.size - 1]);
  
  return np.array(xaxis_interp), np.array(yaxis_interp);
def computeROCCurves(tp, tn, fp, fn):
  yaxis_interp = [];
  xaxis_interp = [];

  false_alarm = fp/(fp + tn + EPS);
  recall = tp/(tp + fn + EPS);
  
  idx = np.argsort(false_alarm);
  
  pr_xaxis = np.zeros(idx.size + 1)
  pr_yaxis = np.zeros(idx.size + 1)
  
  for i in range(0,idx.size):
    pr_xaxis[i] = false_alarm[idx[i]]
    pr_yaxis[i] = recall[idx[i]]
  
  pr_xaxis[idx.size] = 1.0
  pr_yaxis[idx.size] = pr_yaxis[idx.size - 1]
  
  xaxis_interp.append(0.0);
  yaxis_interp.append(0.0);
  
  for i in range(1,pr_xaxis.size - 1):
    x_i = np.linspace(pr_xaxis[i], pr_xaxis[i+1], NUM_INTERP_PTS, endpoint = False);
    y_i = np.linspace(pr_yaxis[i], pr_yaxis[i+1], NUM_INTERP_PTS, endpoint = False);
    
    for k in range(NUM_INTERP_PTS):
      xaxis_interp.append(x_i[k]);
      yaxis_interp.append(y_i[k]);
  
  xaxis_interp.append(pr_xaxis[pr_xaxis.size - 1]);
  yaxis_interp.append(pr_yaxis[pr_yaxis.size - 1]);
  
  return np.array(xaxis_interp), np.array(yaxis_interp);
def computeFbwScore(gt_path, img_path):
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

  Q = (1 + BETA**2) * (R * P) / (eps + R + (BETA * P))

  return Q
def computeImageScores(tuple_path):
  gt_path, img_path = tuple_path;
  
  # Read images
  sm_img = misc.imread(img_path, "L").flatten();
  gt_img = misc.imread(gt_path, "L").flatten();

  # Normalize {0,1}
  gt_img = (gt_img - min(gt_img))/(max(gt_img) - min(gt_img));
  inv_gt_img = 1.0 - gt_img;

  # Init lists of values
  img_tp = np.zeros(NUM_THRESHS); 
  img_tn = np.zeros(NUM_THRESHS);
  img_fp = np.zeros(NUM_THRESHS); 
  img_fn = np.zeros(NUM_THRESHS);
  img_error = np.zeros(NUM_THRESHS);

  # For each threshold
  for i in range(NUM_THRESHS):
    # Init binary image
    bin_img = np.zeros(sm_img.size);

    # Apply threshold
    t = THRESH_VALS[i];
    bin_img[sm_img >= t] = 1.0;
    bin_img[sm_img < t] = 0.0;

    # Get inverted images for
    inv_bin_img = 1.0 - bin_img
    
    # Get basic evaluation metrics
    img_tp[i] = bin_img[gt_img == bin_img].sum()
    img_tn[i] = inv_bin_img[inv_gt_img == inv_bin_img].sum();
    img_fp[i] = bin_img[inv_gt_img == bin_img].sum()
    img_fn[i] = inv_bin_img[gt_img == inv_bin_img].sum();
    img_error[i] = metrics.mean_squared_error(gt_img, bin_img);
  
  pr_xaxis, pr_yaxis = computePrecRecallCurves(img_tp, img_fp, img_fn);
  roc_xaxis, roc_yaxis = computeROCCurves(img_tp, img_tn, img_fp, img_fn);
  
  fbeta = (1.0 + BETA**2)*(pr_xaxis * pr_yaxis)/((BETA**2)*pr_yaxis + pr_xaxis + EPS);
  argmax_fbeta = np.argmax(fbeta);
  
  img_mse = np.mean(img_error);
  img_fbw = computeFbwScore(gt_path, img_path);
  img_auc = metrics.auc(roc_xaxis, roc_yaxis);
  img_fb = fbeta[argmax_fbeta];
  img_prec = pr_yaxis[argmax_fbeta];
  img_rec = pr_xaxis[argmax_fbeta];
  
  return img_tp, img_tn, img_fp, img_fn, img_fb, img_prec, img_rec, img_auc, img_mse, img_fbw
def computeDatasetScores(gt_dir_path, img_dir_path, ds_name):
  # Reading the files names within the dirs
  img_files = glob.glob(img_dir_path + "/*");
  gt_files = glob.glob(gt_dir_path + "/*");
  
  # Creating a dataset text file
  ds_overall_result_fp = open(ds_name + "_overall_results.csv", "w");
  ds_img_result_fp = open(ds_name + "_img_results.csv", "w");
  
  ds_overall_result_fp.write("fbw,auc,mse,fm,precision,recall\n");
  ds_img_result_fp.write("id,fbw,auc,mse,fm,precision,recall\n");

  if(len(img_files) == 0 or len(gt_files) == 0):
    print "ERROR! The directory is empty";
    exit();

  if(len(img_files) != len(gt_files)):
    print "ERROR! The amount of files differ!";
    exit();
  
  num_files = len(img_files);
    
  # Init the values for the mean computation
  tp_sum = np.zeros(NUM_THRESHS);
  tn_sum = np.zeros(NUM_THRESHS);
  fp_sum = np.zeros(NUM_THRESHS);
  fn_sum = np.zeros(NUM_THRESHS);
  fbw_mean = 0.0;
  mse_mean = 0.0;

  print "Evaluating each saliency map";
  # All CPU's but one are used
  ds_inputs = [(gt_files[i], img_files[i]) for i in range(num_files)];
  ds_results = Parallel(n_jobs = -2, verbose = 50)(map(delayed(computeImageScores),ds_inputs));
  
  # For each image result
  for i in range(num_files):
    img_basename = os.path.basename(img_files[i]);
    img_filename = os.path.splitext(img_basename)[0];
    
    img_result = ds_results[i];
    img_tp = img_result[0];
    img_tn = img_result[1];
    img_fp = img_result[2]; 
    img_fn = img_result[3];
    img_fb = img_result[4];
    img_prec = img_result[5];
    img_rec = img_result[6]; 
    img_auc = img_result[7]; 
    img_mse = img_result[8]; 
    img_fbw = img_result[9];
    
    tp_sum += img_tp; fp_sum += img_fp;
    tn_sum += img_tn; fn_sum += img_fn;
    fbw_mean += img_fbw;
    mse_mean += img_mse;
    
    result_text = img_filename + "," + str(img_fbw) + "," + str(img_auc) + "," + str(img_mse) + "," + str(img_fb) + "," + str(img_prec) + "," + str(img_rec) + "\n";

    ds_img_result_fp.write(result_text);
  
  mse_mean /= num_files;
  fbw_mean /= num_files;
  
  # Compute the interpolated values for the PR and ROC curves
  pr_xaxis, pr_yaxis = computePrecRecallCurves(tp_sum, fp_sum, fn_sum);
  roc_xaxis, roc_yaxis = computeROCCurves(tp_sum, tn_sum, fp_sum, fn_sum);
  
  auc = metrics.auc(roc_xaxis, roc_yaxis);
  ds_fbeta = (1.0 + BETA**2)*(pr_xaxis * pr_yaxis)/((BETA**2)*pr_yaxis + pr_xaxis + EPS);
  argmax_fbeta = np.argmax(ds_fbeta);
  
  fb = ds_fbeta[argmax_fbeta];
  prec = pr_yaxis[argmax_fbeta];
  rec = pr_xaxis[argmax_fbeta];
  
  result_text = str(fbw_mean) + "," + str(auc) + "," + str(mse_mean) + "," + str(fb) + "," + str(prec) + "," + str(rec) + "\n";

  ds_overall_result_fp.write(result_text);
  
  # Plot the charts and save in the disk
  print "Writing result images";
  plt.figure(1);
  plt.xlabel('Recall');
  plt.ylabel('Precision');
  plt.grid(True);
  plt.plot(pr_xaxis, pr_yaxis, lw = 2);
  plt.ylim(0.0, 1.01);
  plt.xlim(0.0, 1.01);
  plt.autoscale(enable = False, axis = "both");
  plt.savefig(ds_name + "_pr.png");
  
  plt.figure(2);
  plt.xlabel('False-Alarm');
  plt.ylabel('Recall');
  plt.grid(True);
  plt.plot(roc_xaxis, roc_yaxis, lw = 2);
  plt.plot([0.0, 1.01], [0.0, 1.01], "--" ,lw = 1, alpha = 0.2, color = "gray");
  plt.ylim(0.0, 1.01);
  plt.xlim(-0.01, 1.01);
  plt.autoscale(enable = False, axis = "both");
  plt.savefig(ds_name + "_roc.png");
  
  # Show the perfomance to the user
  print "\n------------------------------------------------";
  print "  Algorithm perfomance on dataset";
  print "------------------------------------------------";
  print "  Fbw = %f" % fbw_mean;
  print "  AUC = %f" % auc;
  print "  MSE = %f" % mse_mean;
  print "  Fb = %f" % fb;
  print "  Precision = %f" % prec;
  print "  Recall = %f" % rec;
  print "------------------------------------------------";
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

