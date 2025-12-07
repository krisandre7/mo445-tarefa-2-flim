#!/usr/bin/python

# author: Cesar Castelo
# date: Dec 6, 2019
# description: This program takes a dataset folder containing learn, train and test sets and performs:
# - Interest point detection and local feats extraction (using the BoVW program iftBoVWLocalFeatsExtraction)
#   obtaining the local feature vectors
# - Trains a clustering model using the learn set
# - Computes the fisher vectors for the train and test sets using the clustering model previously trained
# - Performs classification with the resulting fisher vectors from the train and test sets

import os, sys
import shutil
import subprocess
import json
import numpy as np
from iftBoVWCommonFunctions import create_func_params_json, convert_method_names_to_numbers
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
import warnings
import torch
import torch.nn as nn
from collections import OrderedDict
import pandas as pd
import math
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import resource

patch_size = 9
stride = 7
col_space = 'lab'
msgs_width = 100

#==========================================================================#
# class to define the CNN for feature extraction
#==========================================================================#
class RandomNetNoMLP(nn.Module):

  def __init__(self, network_params):
    super(RandomNetNoMLP, self).__init__()

    #define the layers according to the network architecture
    layers = OrderedDict() 
    for l in range(1, network_params["n_layers"]+1):
        layer = network_params["layer_"+str(l)]
        layers["conv"+str(l)] = nn.Conv2d(in_channels=layer["conv"]["n_bands"], out_channels=layer["conv"]["n_kernels"],
            kernel_size=(layer["conv"]["kernel_size"], layer["conv"]["kernel_size"]), stride=layer["conv"]["stride"],
            padding=layer["conv"]["padding"], bias=True)
        if layer["relu"] == "yes":
            layers["relu"+str(l)] = nn.ReLU()
        if "max-pool" in layer:
            layers["pool"+str(l)] = nn.MaxPool2d(kernel_size=(layer["max-pool"]["size"], layer["max-pool"]["size"]),
                stride=layer["max-pool"]["stride"], padding=layer["max-pool"]["padding"])

    # create the network from the ordered dict
    self.feature_extractor = nn.Sequential(layers)

    #initialize weights
    self._initialize_weights()

  def _initialize_weights(self):
    #for each submodule of our network
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            #get the number of elements in the layer weights
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

            #initialize layer weights with random values generated from a normal
            #distribution with mean = 0 and std = sqrt(2. / n))
            m.weight.data.normal_(mean=0, std=math.sqrt(2. / n))

            if m.bias is not None:
                #initialize bias with 0 (why?)
                m.bias.data.zero_()

  def forward(self, x):
      #extract features
      x = self.feature_extractor(x)

      return x

#==================================================#
# class to represent the image dataset (ift format)
#==================================================#
class MyDataset(Dataset):
    def __init__(self, fileset, transform=None):
        files = pd.read_csv(fileset, header=None)
        self.image_paths = [f for f in files.iloc[:,0]]
        self.labels = [int(os.path.splitext(os.path.basename(f))[0].split("_")[0]) for f in self.image_paths]
        self.transform = transform

    def __getitem__(self, index):
        # Load actual image here
        x = Image.open(self.image_paths[index])
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.image_paths)

#========================================================================#
# function to execute the iftBoVWIntPointsDetection program
#========================================================================#
def execute_ift_int_points_detection(filename, filename_mask, int_point_detection_method, feat_extraction_method,
    func_params, save_patches, save_extra_info, output_dir_basename, output_suffix):
    cmd = "iftBoVWIntPointsDetection {} {} {} {} '{}' {} {} {} {} {}".format(filename, filename_mask, int_point_detection_method["id"],
        feat_extraction_method["id"], json.dumps(func_params, sort_keys=False), save_patches, save_extra_info, output_dir_basename,
        output_suffix, 1)
    if (os.system(cmd) != 0):
            sys.exit("Error in iftBoVWIntPointsDetection")

    # add the name of the file containing the interest points detected in the learning set
    extra_info_json = json.load(open(os.path.join(output_dirname, "extra_info_{}.json".format(output_suffix))))
    func_params["int_point_detec"]["int_points_file"] = extra_info_json["int_points_file"]

#========================================================================#
# function to compute the shape of the local feats vector
#========================================================================#
def compute_local_feats_vect_shape(dataset, feat_extraction_method, func_params):
    if feat_extraction_method["str"] == "conv-ml":
        # read the parameters
        vect_constr = func_params['local_feat_extr']['vect_constr']
        if vect_constr == 'bic_in_patch':
            n_bins_per_band = func_params['local_feat_extr']['n_bins_per_band']

        # get n_imgs, n_rois, n_pts_roi, n_bands
        n_imgs = len(dataset)

        int_points_file = open(func_params["int_point_detec"]["int_points_file"], "r")
        if int_points_file.readline().rstrip() != "ROI_ARRAY":
            exit("The file {} must contain a ROI array".format(func_params["int_point_detec"]["int_points_file"]))
        n_rois = int(int_points_file.readline().rstrip())
        n_pts_roi = int(int_points_file.readline().rstrip())
        int_points_file.close()

        network_params = func_params['local_feat_extr']['network_params']
        n_layers = network_params["n_layers"]
        n_bands = network_params["layer_"+str(n_layers)]["conv"]["n_kernels"]

        # compute the shape of the local feats vector (the shape is based on a 2D array for compatibility with Python and C numpy arrays)
        if vect_constr == "pixel_vals_in_patch":
            local_feats_shape = (n_imgs*n_rois, n_pts_roi*n_bands)
        elif vect_constr == "mean_val_in_patch":
            local_feats_shape = (n_imgs*n_rois, n_bands)
        elif vect_constr == "bic_in_patch":
            local_feats_shape = (n_imgs*n_rois, n_bands*n_bins_per_band*2)

    return local_feats_shape

#========================================================================#
# function to build the local feature vectors from the filtered images
#========================================================================#
def build_local_feat_vector_from_filt_imgs(filt_imgs, func_params):
    # read the parameters
    vect_constr = func_params['local_feat_extr']['vect_constr']
    if vect_constr == 'bic_in_patch':
        n_bins_per_band = func_params['local_feat_extr']['n_bins_per_band']
    n_imgs, n_bands = filt_imgs.shape[0], filt_imgs.shape[1]
    local_feats = None

    # save the filtered images
    aux_dir = os.path.join(output_dirname, "filtered_imgs")
    if os.path.isdir(aux_dir):
        shutil.rmtree(aux_dir)
    os.makedirs(aux_dir)
    for i in range(n_imgs):
        for b in range(n_bands):
            img = filt_imgs[i,b,:,:]
            img = (img - np.min(img)) / max(np.max(img) - np.min(img), 1) * 255
            Image.fromarray(np.uint8(img)).save(os.path.join(aux_dir, "img_{}_band_{}.png".format(i+1,b+1)))
    
    # execute the C program
    cmd = "iftBoVWLocalFeatsExtrFromFiltImgs {} {} {} {} {} {} {} {} {}".format(aux_dir, n_imgs, n_bands, vect_constr,
        func_params["int_point_detec"]["int_points_file"], n_bins_per_band if vect_constr == "bic_in_patch" else -1,
        output_dirname, output_suffix, "0")
    if (os.system(cmd) != 0):
            sys.exit("Error in iftBoVWLocalFeatsExtrFromFiltImgs")
    
    # read the local feature vectors
    dataset_name = os.path.join(output_dirname, "local_feats_{}.npy".format(output_suffix))
    local_feats = np.load(dataset_name)
    os.remove(dataset_name)

    return local_feats

#==============================================================================#
# function to perform local feats extraction and/or fisher vectors estimation
#==============================================================================#
def perform_feats_extraction(filename, feat_extraction_method, func_params, torch_device, batch_size, mode, gmm=None):
    assert mode in ["learning","training","testing"], "Mode must be one from ['learning','training','testing']"

    if mode == "learning":
        print("\n--> Extracting the local feature vectors ... ")
    else:
        print("\n--> Extracting the local feature vectors and the final fisher vectors... ")

    local_feats_extr_time, fisher_vect_comp_time = 0, 0
    if feat_extraction_method["str"] == "conv-ml":
        # read the parameters
        network_params = func_params['local_feat_extr']['network_params']
        resize_image = func_params['local_feat_extr']['resize_image']

        # determine the torch device to be used
        if torch_device != "cpu" and not torch.cuda.is_available():
            print("[PyTorch] CUDA is not available ... using CPU instead!")
            torch_device = "cpu"
        device = torch.device(torch_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # define the image transformations
        norm_mean, norm_stdev = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_stdev)
        ])
            
        # read the images from the directory
        dataset = MyDataset(filename, data_transforms)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        n_imgs = len(dataset)

        # create the network
        model = RandomNetNoMLP(network_params)
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(device)

        # pre-allocate memory for the feature vectors
        local_feats_shape = compute_local_feats_vect_shape(dataset, feat_extraction_method, func_params)
        if mode == "learning":
            local_feats = np.zeros(local_feats_shape, dtype=np.float32)
        else:
            fisher_vectors = np.zeros((n_imgs, 2*local_feats_shape[1]*gmm.means_.shape[0]+gmm.means_.shape[0]), dtype=np.float32)

        # execute the entire process for each batch
        model.train()
        bt, n_feat_vec = 0, 0
        n_rois = int(local_feats_shape[0]/len(dataset))

        for inputs, labels in dataloader:
            print("Batch: {}/{}".format(bt+1, len(dataloader)))
            
            # perform feed-forward with the model
            time_start_1 = time.time()
            print("- Computing the convolutional features ...")
            inputs = inputs.to(device)
            labels = labels.to(device)
            torch.set_grad_enabled(False)
            output = model(inputs)
            filt_imgs = output.cpu().detach().numpy()

            # resize the filtered images
            n_imgs_in_batch, n_bands = filt_imgs.shape[0], filt_imgs.shape[1]
            if resize_image == "yes":
                filt_imgs_1 = np.zeros((n_imgs_in_batch, n_bands, inputs.shape[2], inputs.shape[3]), dtype=np.float32)
                for i in range(n_imgs_in_batch):
                    for b in range(n_bands):
                        filt_imgs_1[i,b,:,:] = Image.fromarray(np.uint8(filt_imgs[i,b,:,:])).resize((inputs.shape[2], inputs.shape[3]))
                filt_imgs = filt_imgs_1

            if inputs.shape[2] != filt_imgs.shape[2] or inputs.shape[3] != filt_imgs.shape[3]:
                exit("The resulting filtered images must have the same size than the original ones or be resized: ({}x{} vs {}x{})"
                    .format(inputs.shape[2], inputs.shape[3], filt_imgs.shape[2], filt_imgs.shape[3]))

            # build the local feature vectors
            feats = build_local_feat_vector_from_filt_imgs(filt_imgs, func_params)
            if mode == "learning":
                local_feats[n_feat_vec:n_feat_vec+n_imgs_in_batch*n_rois, :] = feats
                n_feat_vec += n_imgs_in_batch*n_rois
            local_feats_extr_time += time.time()-time_start_1

            # compute fisher vectors (only for training/testing modes)
            if mode in ["training", "testing"]:
                time_start_2 = time.time()
                print("- Computing the fisher vectors ... ({} resulting features)".format(fisher_vectors.shape[1]))
                for i in range(n_imgs_in_batch):
                    print("  Image {}/{} ...\r".format(i+1, n_imgs_in_batch), end="")
                    fisher_vectors[n_feat_vec,:] = np.atleast_2d(compute_fisher_vector(feats[i*n_rois:(i+1)*n_rois], gmm))
                    n_feat_vec += 1
                fisher_vect_comp_time += time.time()-time_start_2
                print()
            bt += 1
        print()
    else:
        exit("Error: Local feat extraction method not implemented in Python ({})".format(feat_extraction_method["str"]))

    if mode == "learning":
        print('Done! Feature extraction time: {:.2f} seconds'.format(local_feats_extr_time))
        return (local_feats, local_feats_extr_time)
    else:
        print('Done! Feature extraction time: {:.2f} seconds, Fisher vector computation time: {:.2f} seconds'.format(
            local_feats_extr_time, fisher_vect_comp_time))
        return (fisher_vectors, dataset.labels, local_feats_extr_time, fisher_vect_comp_time)

#========================================================================#
# function to perform GMM clustering
#========================================================================#
def perform_clustering_gmm(learn_set, n_groups_gmm):
    time_start = time.time()
    print("\n--> Creating the GMM clustering model ... (n_samples: {}, n_local_feats_per_img: {}, n_groups: {})".format(
        learn_set.shape[0], learn_set.shape[1], n_groups_gmm))

    # run GMM to determine the number of different clusters that will be found
    gmm = GaussianMixture(n_components=n_groups_gmm, covariance_type='diag', random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm.fit(learn_set)
    n_unique_clust_centers = np.unique(np.round(gmm.means_, decimals=5), axis=0).shape[0]

    if n_unique_clust_centers != n_groups_gmm:
        print("- GMM found {} unique clusters. That number will be used to compute the fisher vectors instead of {}".format(
            n_unique_clust_centers, n_groups_gmm))
        
        # run GMM again with the number of clusters that was found
        gmm = GaussianMixture(n_components=n_unique_clust_centers, covariance_type='diag', random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(learn_set)
        n_groups_gmm = n_unique_clust_centers

    n_local_feats = learn_set.shape[1]
    fisher_vector_size = 2*n_groups_gmm*learn_set.shape[1] + n_groups_gmm
    clust_time = time.time()-time_start

    print('Done! Time elapsed: {:.2f} seconds'.format(clust_time))
    return (gmm, n_groups_gmm, n_local_feats, fisher_vector_size, clust_time)

#========================================================================#
# function to compute the fisher vector from an image given a GMM model
#========================================================================#
def compute_fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

#========================================================================#
# function to perform SVM training and testing from train and test sets
#========================================================================#
def perform_svm_classification(classification_method, feat_vects_train, labels_train, feat_vects_test, labels_test):
    # perform training
    time_start = time.time()
    print("\n--> Training the {} classifier ... ".format(classification_method))

    if classification_method == "svm-linear-ova":
        classif = LinearSVC(random_state=0, tol=1e-5)
    elif classification_method == "svm-linear-ovo":
        classif = SVC(kernel='linear', gamma='auto', random_state=0, tol=1e-5, decision_function_shape='ovo')
    elif classification_method == "svm-rbf-ova":
        classif = SVC(kernel='rbf', gamma='auto', random_state=0, tol=1e-5, decision_function_shape='ovr')
    elif classification_method == "svm-rbf-ovo":
        classif = SVC(kernel='rbf', gamma='auto', random_state=0, tol=1e-5, decision_function_shape='ovo')
    
    classif.fit(feat_vects_train, labels_train)
    svm_train_time = time.time()-time_start
    print('Done! Time elapsed: {:.2f} seconds'.format(svm_train_time))

    # perform classification
    time_start = time.time()
    print("\n--> Classifying the testing data ... ")
    test_set_labels_predict = classif.predict(feat_vects_test)
    svm_test_time = time.time()-time_start
    print('Done! Time elapsed: {:.2f} seconds'.format(svm_test_time))

    # measure accuracy and kappa
    acc = accuracy_score(labels_test, test_set_labels_predict)
    kappa = cohen_kappa_score(labels_test, test_set_labels_predict)
    print("- accuracy: {:.6f}".format(acc))
    print("- kappa: {:.6f}".format(kappa))

    return (acc, kappa, svm_train_time, svm_test_time)

# main function
if __name__ == "__main__":

    # verify input parameters
    if(len(sys.argv) != 12):
        print("#"*msgs_width)
        print("usage: iftBoVWFisherVectors.py <...>")
        print("[1] dataset_dirname: Dirname of the image dataset")
        print("[2] n_splits: Number of splits in the dataset")
        print("[3] use_samp_masks: Use masks for sampling ['yes', 'no']")
        print("[4] int_point_detection_method: Method to detect interest points")
        print("    'random': Random sampling")
        print("    'grid': Grid sampling")
        print("    'unsup-spix-isf': Unsupervised superpixels by ISF")
        print("    'sup-spix-isf': Supervised superpixels by ISF")
        print("[5] feat_extraction_method: Method to extract the local features")
        print("    'raw': Raw pixel values")
        print("    'bic': Border/Interior Pixel Classification (BIC)")
        print("    'lbp': Local Binary Patterns (LBP)")
        print("    'brief': BRIEF descriptor")
        print("    'conv': Convolutional features")
        print("    'conv-ml': Multi Layer Convolutional features")
        print("    'deep-feats-mimg': Deep features from mimage")
        print("[6] classification_method:")
        print("    'svm-linear-ova': SVM with linear kernel and OVA multiclass strategy (faster LinearSVC implementation)")
        print("    'svm-linear-ovo': SVM with linear kernel and OVO multiclass strategy")
        print("    'svm-rbf-ova': SVM with linear kernel and OVA multiclass strategy")
        print("    'svm-rbf-ovo': SVM with linear kernel and OVO multiclass strategy")
        print("[7] n_groups_gmm: Number of groups for the GMM model")
        print("[8] torch_device: Device to be used for PyTorch ('cpu','cuda:0','cuda:1',etc)")
        print("[9] batch_size: Batch size to execute the CNN (only if feat_extraction_method=conv-ml, -1 otherwise)")
        print("[10] save_feat_vect: Whether or not to save the feature vectors ['yes', 'no']")
        print("[11] output_suffix: Suffix to be added to the output files")
        print("#"*msgs_width)
        sys.exit(-1)

    # read input parameters
    dataset_dirname = sys.argv[1].rstrip("/")
    n_splits = int(sys.argv[2])
    use_samp_masks = sys.argv[3]
    int_point_detection_method = sys.argv[4]
    feat_extraction_method = sys.argv[5]
    classification_method = sys.argv[6]
    n_groups_gmm = int(sys.argv[7])
    torch_device = sys.argv[8]
    batch_size = int(sys.argv[9])
    save_feat_vect = sys.argv[10]
    output_suffix = sys.argv[11]

    # set extra params
    save_patches = 0
    perform_tsne = 0
    save_extra_info = 1

    # set resource limits
    resource.setrlimit(resource.RLIMIT_DATA, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    
    # print input parameters
    print("#"*msgs_width)
    print("Input parameters (iftBoVWFisherVectors.py script):")
    print("#"*msgs_width)
    print("dataset_dirname:", dataset_dirname)
    print("n_splits:", n_splits)
    print("use_samp_masks:", use_samp_masks)
    print("int_point_detection_method:", int_point_detection_method)
    print("feat_extraction_method:", feat_extraction_method)
    print("classification_method:", classification_method)
    print("n_groups_gmm:", n_groups_gmm)
    print("output_suffix:", output_suffix)

    # convert method names to numbers (according to   definition in LibIFT)
    [int_point_detection_method_int, feat_extraction_method_int, _, _, _] = convert_method_names_to_numbers(int_point_detection_method,
        feat_extraction_method, "", "", "")
    int_point_detection_method = {"str": int_point_detection_method, "id": int_point_detection_method_int}
    feat_extraction_method = {"str": feat_extraction_method, "id": feat_extraction_method_int}

    # create output dir
    output_dir_basename = "fisher_vectors_" + dataset_dirname
    output_dirname = output_dir_basename + '_' + int_point_detection_method["str"] + '_' + feat_extraction_method["str"]

    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    # create JSON with function params
    func_params = create_func_params_json(int_point_detection_method["str"], feat_extraction_method["str"], "", "", "",
        use_samp_masks, patch_size, stride, col_space)
    print("parameters for each method:")
    i = 1
    for key in func_params:
        print("[{}] {}: {}".format(i, key, func_params[key]))
        i += 1

    # create dictionary to save the results
    dataset_learn = MyDataset(os.path.join(dataset_dirname, "learn_001.csv"), None)
    results_json = {
        "chosen_methods": {"int_point_detector": int_point_detection_method["str"], "local_feat_extractor": feat_extraction_method["str"],
            "clust_meth": "gmm", "classif_meth": classification_method},
        "func_params": func_params,
        "n_classes": len(set(dataset_learn.labels)),
        "method_info": {"n_gmm_groups": [], "n_local_feats": [], "fisher_vector_size": []},
        "learn_set_stats": {"n_imgs": [], "local_feats_extr_time": [], "clust_learn_time": []},
        "train_set_stats": {"n_imgs": [], "local_feats_extr_time": [], "fisher_vect_comp_time": []},
        "test_set_stats": {"n_imgs": [], "local_feats_extr_time": [], "fisher_vect_comp_time": []},
        "classif_results": {"acc": [], "kappa": [], "svm_train_time": [], "svm_test_time": []}}

    # execute the BoVW pipeline for each split
    for split in range(1, int(n_splits)+1):
        split = str(split)
        print("\n", "#"*msgs_width, sep="")
        print("SPLIT: {} of {}".format(split, n_splits))
        print("#"*msgs_width)

        # variables for each iteration
        filename_learn = os.path.join(dataset_dirname, "learn_" + str(split).zfill(3) + ".csv")
        filename_train = os.path.join(dataset_dirname, "train_" + str(split).zfill(3) + ".csv")
        filename_test = os.path.join(dataset_dirname, "test_" + str(split).zfill(3) + ".csv")

        # set filenames for masks
        filename_learn_mask = "-1"
        if(use_samp_masks == "yes"):
            filename_learn_mask = os.path.join(dataset_dirname, "learn_" + str(split).zfill(3) + "_masks.csv")

        # detect interest points and extract local features from the learning set
        print("\n", "="*msgs_width, sep="")
        print("PROCESSING LEARNING SET ({} images)".format(len(list(open(filename_learn)))))
        print("="*msgs_width)

        time_start = time.time()
        # detect the interest points
        execute_ift_int_points_detection(filename_learn, filename_learn_mask, int_point_detection_method, feat_extraction_method,
            func_params, save_patches, save_extra_info, output_dir_basename, output_suffix)
        int_point_detection_time = time.time()-time_start
        
        # extract local features
        local_feats_learn, local_feats_extr_time = perform_feats_extraction(filename_learn, feat_extraction_method, func_params, torch_device,
            batch_size, "learning")
        results_json["learn_set_stats"]["n_imgs"].append(len(list(open(filename_learn))))
        results_json["learn_set_stats"]["local_feats_extr_time"].append(local_feats_extr_time + int_point_detection_time)

        # perform clustering with the learning set (GMM model)
        gmm, n_groups_gmm, n_local_feats, fisher_vector_size, clust_learn_time = perform_clustering_gmm(local_feats_learn, n_groups_gmm)
        results_json["method_info"]["n_gmm_groups"].append(n_groups_gmm)
        results_json["method_info"]["n_local_feats"].append(n_local_feats)
        results_json["method_info"]["fisher_vector_size"].append(fisher_vector_size)
        results_json["learn_set_stats"]["clust_learn_time"].append(clust_learn_time)

        # extract local features from the training set and compute the fisher vectors
        print("\n", "="*msgs_width, sep="")
        print("PROCESSING TRAINING SET ({} images)".format(len(list(open(filename_train)))))
        print("="*msgs_width)

        feat_vects_train, labels_train, local_feats_extr_time, fisher_vect_comp_time = perform_feats_extraction(filename_train,
            feat_extraction_method, func_params, torch_device, batch_size, "training", gmm)
        results_json["train_set_stats"]["n_imgs"].append(len(list(open(filename_train))))
        results_json["train_set_stats"]["local_feats_extr_time"].append(local_feats_extr_time)
        results_json["train_set_stats"]["fisher_vect_comp_time"].append(fisher_vect_comp_time)

        if save_feat_vect == "yes":
            time_start = time.time()
            print("--> Saving the fisher vectors ...")
            dataset_name = os.path.join(output_dirname, "feat_vect_{}.npz".format(output_suffix + "_train_split_" + str(split).zfill(3)))
            np.savez_compressed(dataset_name, feat_vect=feat_vects_train, labels=labels_train)
            print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))

        # extract local features from the testing set and compute the fisher vectors
        print("\n", "="*msgs_width, sep="")
        print("PROCESSING TESTING SET ({} images)".format(len(list(open(filename_test)))))
        print("="*msgs_width)

        feat_vects_test, labels_test, local_feats_extr_time, fisher_vect_comp_time = perform_feats_extraction(filename_test,
            feat_extraction_method, func_params, torch_device, batch_size, "testing", gmm)
        results_json["test_set_stats"]["n_imgs"].append(len(list(open(filename_test))))
        results_json["test_set_stats"]["local_feats_extr_time"].append(local_feats_extr_time)
        results_json["test_set_stats"]["fisher_vect_comp_time"].append(fisher_vect_comp_time)

        if save_feat_vect == "yes":
            time_start = time.time()
            print("--> Saving the fisher vectors ...")
            dataset_name = os.path.join(output_dirname, "feat_vect_{}.npz".format(output_suffix + "_test_split_" + str(split).zfill(3)))
            np.savez_compressed(dataset_name, feat_vect=feat_vects_test, labels=labels_test)
            print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))

        # perform SVM classificaion
        acc, kappa, svm_train_time, svm_test_time = perform_svm_classification(classification_method, feat_vects_train, labels_train,
            feat_vects_test, labels_test)
        results_json["classif_results"]["acc"].append(acc)
        results_json["classif_results"]["kappa"].append(kappa)
        results_json["classif_results"]["svm_train_time"].append(svm_train_time)
        results_json["classif_results"]["svm_test_time"].append(svm_test_time)

    # summarize results
    results_json["method_info"]["n_gmm_groups"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["method_info"]["n_gmm_groups"]), 
        np.std(results_json["method_info"]["n_gmm_groups"]))
    results_json["method_info"]["n_local_feats"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["method_info"]["n_local_feats"]),
        np.std(results_json["method_info"]["n_local_feats"]))
    results_json["method_info"]["fisher_vector_size"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["method_info"]["fisher_vector_size"]),
        np.std(results_json["method_info"]["fisher_vector_size"]))
    results_json["learn_set_stats"]["n_imgs"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["learn_set_stats"]["n_imgs"]),
        np.std(results_json["learn_set_stats"]["n_imgs"]))
    results_json["learn_set_stats"]["local_feats_extr_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["learn_set_stats"]["local_feats_extr_time"]),
        np.std(results_json["learn_set_stats"]["local_feats_extr_time"]))
    results_json["learn_set_stats"]["clust_learn_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["learn_set_stats"]["clust_learn_time"]),
        np.std(results_json["learn_set_stats"]["clust_learn_time"]))
    results_json["train_set_stats"]["n_imgs"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["train_set_stats"]["n_imgs"]),
        np.std(results_json["train_set_stats"]["n_imgs"]))
    results_json["train_set_stats"]["local_feats_extr_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["train_set_stats"]["local_feats_extr_time"]),
        np.std(results_json["train_set_stats"]["local_feats_extr_time"]))
    results_json["train_set_stats"]["fisher_vect_comp_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["train_set_stats"]["fisher_vect_comp_time"]),
        np.std(results_json["train_set_stats"]["fisher_vect_comp_time"]))
    results_json["test_set_stats"]["n_imgs"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["test_set_stats"]["n_imgs"]),
        np.std(results_json["test_set_stats"]["n_imgs"]))
    results_json["test_set_stats"]["local_feats_extr_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["test_set_stats"]["local_feats_extr_time"]),
        np.std(results_json["test_set_stats"]["local_feats_extr_time"]))
    results_json["test_set_stats"]["fisher_vect_comp_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["test_set_stats"]["fisher_vect_comp_time"]),
        np.std(results_json["test_set_stats"]["fisher_vect_comp_time"]))
    results_json["classif_results"]["acc"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["classif_results"]["acc"]),
        np.std(results_json["classif_results"]["acc"]))
    results_json["classif_results"]["kappa"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["classif_results"]["kappa"]),
        np.std(results_json["classif_results"]["kappa"]))
    results_json["classif_results"]["svm_train_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["classif_results"]["svm_train_time"]),
        np.std(results_json["classif_results"]["svm_train_time"]))
    results_json["classif_results"]["svm_test_time"] = "{:.2f} +- {:.2f}".format(np.mean(results_json["classif_results"]["svm_test_time"]),
        np.std(results_json["classif_results"]["svm_test_time"]))
    
    # print results
    print("\n", "="*msgs_width, sep="")
    print("FINAL RESULTS")
    print("="*msgs_width)
    print(json.dumps(results_json, sort_keys=False, indent=4))
    
    filename = os.path.join(output_dirname, "results_{}.json".format(output_suffix))
    fp = open(filename, 'w')
    json.dump(results_json, fp, sort_keys=False, indent=4)