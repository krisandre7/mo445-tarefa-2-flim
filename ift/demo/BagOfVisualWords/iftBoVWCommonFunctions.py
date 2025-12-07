#!/usr/bin/python

# author: Cesar Castelo
# date: Mar 03, 2019
# description: This small library contains python functions that are used in several python scripts for BoVW

from math import sqrt

def create_func_params_json(int_point_detection_method, feat_extraction_method, dict_estimation_method, coding_method,
                            classification_method, use_samp_masks, patch_size, stride, col_space):

    func_params = dict()

    # interest point detection method
    if int_point_detection_method != "":
        func_params['int_point_detec'] = dict()
        if int_point_detection_method == 'random':
            func_params['int_point_detec']['n_points'] = 324
            func_params['int_point_detec']['patch_size'] = patch_size

        if int_point_detection_method == 'grid':
            func_params['int_point_detec']['patch_size'] = patch_size
            func_params['int_point_detec']['stride'] = stride

        if int_point_detection_method == 'unsup-spix-isf':
            func_params['int_point_detec']['n_superpixels'] = 200
            func_params['int_point_detec']['patch_size'] = patch_size

        if int_point_detection_method == 'sup-spix-isf':
            func_params['int_point_detec']['point_position'] = 'boundary'
            func_params['int_point_detec']['patch_shape'] = 'square'
            func_params['int_point_detec']['patch_size'] = patch_size
            func_params['int_point_detec']['int_pts_stride'] = stride
            func_params['int_point_detec']['spix_seeds_stride'] = 10
            func_params['int_point_detec']['adj_rel_rad'] = sqrt(2.0)
            func_params['int_point_detec']['alpha'] = 0.12
            func_params['int_point_detec']['beta'] = 12.0
            func_params['int_point_detec']['n_iter'] = 10
            func_params['int_point_detec']['col_space'] = col_space
    
    if use_samp_masks == 'yes':
            func_params['int_point_detec']['mask_join_operation'] = 'union'

    # local feature extraction method
    if feat_extraction_method != "":
        func_params['local_feat_extr'] = dict()
        if feat_extraction_method == 'raw':
            func_params['local_feat_extr']['col_space'] = col_space

        if feat_extraction_method == 'bic':
            func_params['local_feat_extr']['n_bins'] = 8

        if feat_extraction_method == 'lbp':
            func_params['local_feat_extr']['circ_sym_neighb_rad'] = 2.0
            func_params['local_feat_extr']['circ_sym_neighb_n_points'] = 16

        if feat_extraction_method == 'brief':
            func_params['local_feat_extr']['patch_size'] = patch_size
            func_params['local_feat_extr']['descriptor_size'] = 512

        if feat_extraction_method == 'conv':
            func_params['local_feat_extr']['col_space'] = col_space
            func_params['local_feat_extr']['n_kernels'] = 8
            func_params['local_feat_extr']['kernel_size'] = patch_size
            func_params['local_feat_extr']['conv_stride'] = 1
            func_params['local_feat_extr']['pool_size'] = 4
            func_params['local_feat_extr']['pool_stride'] = 1
            func_params['local_feat_extr']['max_pool_type'] = 'max_pool'
            func_params['local_feat_extr']['resize_image'] = 'no'
            func_params['local_feat_extr']['vect_constr'] = 'pixel_vals_in_patch'
            if func_params['local_feat_extr']['vect_constr'] == 'bic_in_patch':
                func_params['local_feat_extr']['n_bins_per_band'] = 8

        if feat_extraction_method == 'conv-ml':
            func_params['local_feat_extr']['col_space'] = col_space
            func_params['local_feat_extr']['network_params'] = create_network_params_dict('alexnet_no_stride')
            func_params['local_feat_extr']['max_pool_type'] = 'max_pool'
            func_params['local_feat_extr']['resize_image'] = 'no'
            func_params['local_feat_extr']['vect_constr'] = 'bic_in_patch'
            if func_params['local_feat_extr']['vect_constr'] == 'bic_in_patch':
                func_params['local_feat_extr']['n_bins_per_band'] = 8

        if feat_extraction_method == 'deep-feats-mimg':
            # AQUI DEVO INDICAR A PASTA E DEPOIS PEGAR OS ARQUIVOS SEGUNDO OS SPLITS!
            func_params['local_feat_extr']['feature_mimgs_dir'] = '/media/cesar/Disk/Documentos/UNICAMP_PhD/Pesquisa/LIDSConvNet/experiments_proto_no_imp3535/layer1_train_test_output/train1_test1.csv'
            func_params['local_feat_extr']['resize_image'] = 'no'
            func_params['local_feat_extr']['vect_constr'] = 'pixel_vals_in_patch'
            if func_params['local_feat_extr']['vect_constr'] == 'pixel_vals_in_patch':
                func_params['local_feat_extr']['n_bins_per_band'] = 16

    # dictionary estimation method
    if dict_estimation_method != "":
        func_params['dict_estim'] = dict()
        if dict_estimation_method in ['unsup-kmeans', 'sup-kmeans-pix-label', 'sup-kmeans-img-class', 'sup-kmeans-image',
                            'sup-kmeans-position', 'sup-kmeans-img-class-position']:
            func_params['dict_estim']['ret_real_cent'] = 'true'
            func_params['dict_estim']['n_groups'] = 1000
            func_params['dict_estim']['max_iter'] = 20
            func_params['dict_estim']['min_improv'] = 0.001

        if dict_estimation_method in ['unsup-opf', 'sup-opf-pix-label', 'sup-opf-img-class', 'sup-opf-image',
                            'sup-opf-position', 'sup-opf-img-class-position']:
            func_params['dict_estim']['ret_real_cent'] = 'false'
            func_params['dict_estim']['clust_samp_perc'] = 0.1
            func_params['dict_estim']['knn_graph_n_neighb'] = 10.0
            func_params['dict_estim']['known_n_groups'] = 'false'
            func_params['dict_estim']['n_groups'] = -1
            func_params['dict_estim']['use_prot_as_cent'] = 'true'
            func_params['dict_estim']['find_optm_knn_for_pdf'] = 'false'

    # coding method
    if coding_method != "":
        func_params['cod_func'] = dict()
        if coding_method in ['soft-asgmt', 'soft-asgmt-batch']:
            func_params['cod_func']['n_near_neighb'] = 15
            func_params['cod_func']['wgt_func'] = 'gaussian'
    
    # classification method
    if classification_method != "":
        func_params['classif_met'] = dict()
        if classification_method in ['svm-linear', 'svm-rbf']:
            func_params['classif_met']['multiclass'] = 'ovo'
            func_params['classif_met']['c'] = 1e5
            func_params['classif_met']['kernelize'] = 'true'

    return func_params

def convert_method_names_to_numbers(int_point_detection_method, feat_extraction_method, dict_estimation_method, coding_method, classification_method):

    # interest point detection method    
    if int_point_detection_method == 'random':
        int_point_detection_method = 0
    if int_point_detection_method == 'grid':
        int_point_detection_method = 1
    if int_point_detection_method == 'unsup-spix-isf':
        int_point_detection_method = 2
    if int_point_detection_method == 'sup-spix-isf':
        int_point_detection_method = 3

    # local feature extraction method
    if feat_extraction_method == 'raw':
        feat_extraction_method = 0
    if feat_extraction_method == 'bic':
        feat_extraction_method = 1
    if feat_extraction_method == 'lbp':
        feat_extraction_method = 2
    if feat_extraction_method == 'brief':
        feat_extraction_method = 3
    if feat_extraction_method == 'conv':
        feat_extraction_method = 4
    if feat_extraction_method == 'conv-ml':
        feat_extraction_method = 5
    if feat_extraction_method == 'deep-feats-mimg':
        feat_extraction_method = 6

    # dictionary estimation method
    if dict_estimation_method == 'unsup-kmeans':
        dict_estimation_method = 0
    if dict_estimation_method == 'sup-kmeans-pix-label':
        dict_estimation_method = 1
    if dict_estimation_method == 'sup-kmeans-img-class':
        dict_estimation_method = 2
    if dict_estimation_method == 'sup-kmeans-image':
        dict_estimation_method = 3
    if dict_estimation_method == 'sup-kmeans-position':
        dict_estimation_method = 4
    if dict_estimation_method == 'sup-kmeans-img-class-position':
        dict_estimation_method = 5
    if dict_estimation_method == 'unsup-opf':
        dict_estimation_method = 6
    if dict_estimation_method == 'sup-opf-pix-label':
        dict_estimation_method = 7
    if dict_estimation_method == 'sup-opf-img-class':
        dict_estimation_method = 8
    if dict_estimation_method == 'sup-opf-image':
        dict_estimation_method = 9
    if dict_estimation_method == 'sup-opf-position':
        dict_estimation_method = 10
    if dict_estimation_method == 'sup-opf-img-class-position':
        dict_estimation_method = 11
    if dict_estimation_method == 'sup-manual-img-class':
        dict_estimation_method = 12
    if dict_estimation_method == 'sup-manual-position':
        dict_estimation_method = 13
    if dict_estimation_method == 'sup-manual-img-class-position':
        dict_estimation_method = 14

    # coding method
    if coding_method == 'hard-asgmt':
        coding_method = 0
    if coding_method == 'soft-asgmt':
        coding_method = 1
    if coding_method == 'hard-asgmt-batch':
        coding_method = 2
    if coding_method == 'soft-asgmt-batch':
        coding_method = 3
    if coding_method == '2nd-order-stats-fv':
        coding_method = 4

    # classification_method method
    if classification_method == 'svm-linear':
        classification_method = 0
    if classification_method == 'svm-rbf':
        classification_method = 1
    if classification_method == 'opf':
        classification_method = 2

    return [int_point_detection_method, feat_extraction_method, dict_estimation_method, coding_method, classification_method]

def create_network_params_dict(network_name):
    dictionary = dict()
    
    if network_name == '2layers_no_stride':
        dictionary = {
            'n_layers': 2,
            'layer_1': {
                'conv': {'n_bands': 3, 'n_kernels': 8, 'kernel_size': 9, 'stride': 1, 'padding': 4},
                'relu': 'yes',
                'max-pool': {'size': 4, 'stride': 1, 'padding': 2}
            },
            'layer_2': {
                'conv': {'n_bands': 8, 'n_kernels': 16, 'kernel_size': 9, 'stride': 1, 'padding': 4},
                'relu': 'yes',
                'max-pool': {'size': 4, 'stride': 1, 'padding': 2}
            }
        }

    if network_name == '3layers_no_stride':
        dictionary = {
            'n_layers': 3,
            'layer_1': {
                'conv': {'n_bands': 3, 'n_kernels': 8, 'kernel_size': 9, 'stride': 1, 'padding': 4},
                'relu': 'yes',
                'max-pool': {'size': 4, 'stride': 1, 'padding': 2}
            },
            'layer_2': {
                'conv': {'n_bands': 8, 'n_kernels': 16, 'kernel_size': 9, 'stride': 1, 'padding': 4},
                'relu': 'yes',
                'max-pool': {'size': 4, 'stride': 1, 'padding': 2}
            },
            'layer_3': {
                'conv': {'n_bands': 16, 'n_kernels': 32, 'kernel_size': 9, 'stride': 1, 'padding': 4},
                'relu': 'yes',
                'max-pool': {'size': 4, 'stride': 1, 'padding': 2}
            }
        }

    if network_name == 'alexnet':
        dictionary = {
            'n_layers': 5,
            'layer_1': {
                'conv': {'n_bands': 3, 'n_kernels': 64, 'kernel_size': 11, 'stride': 4, 'padding': 5},
                'relu': 'yes',
                'max-pool': {'size': 3, 'stride': 2, 'padding': 1}
            },
            'layer_2': {
                'conv': {'n_bands': 64, 'n_kernels': 192, 'kernel_size': 5, 'stride': 1, 'padding': 2},
                'relu': 'yes',
                'max-pool': {'size': 3, 'stride': 2, 'padding': 1}
            },
            'layer_3': {
                'conv': {'n_bands': 192, 'n_kernels': 384, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_4': {
                'conv': {'n_bands': 384, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_5': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 3, 'stride': 2, 'padding': 1}
            }
        }

    if network_name == 'alexnet_no_stride':
        dictionary = {
            'n_layers': 5,
            'layer_1': {
                'conv': {'n_bands': 3, 'n_kernels': 64, 'kernel_size': 11, 'stride': 1, 'padding': 5},
                'relu': 'yes',
                'max-pool': {'size': 3, 'stride': 1, 'padding': 1}
            },
            'layer_2': {
                'conv': {'n_bands': 64, 'n_kernels': 192, 'kernel_size': 5, 'stride': 1, 'padding': 2},
                'relu': 'yes',
                'max-pool': {'size': 3, 'stride': 1, 'padding': 1}
            },
            'layer_3': {
                'conv': {'n_bands': 192, 'n_kernels': 384, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_4': {
                'conv': {'n_bands': 384, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_5': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 3, 'stride': 1, 'padding': 1}
            }
        }

    if network_name == 'vgg19':
        dictionary = {
            'n_layers': 16,
            'layer_1': {
                'conv': {'n_bands': 3, 'n_kernels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_2': {
                'conv': {'n_bands': 64, 'n_kernels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 2, 'padding': 1}
            },
            'layer_3': {
                'conv': {'n_bands': 64, 'n_kernels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_4': {
                'conv': {'n_bands': 128, 'n_kernels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 2, 'padding': 1}
            },
            'layer_5': {
                'conv': {'n_bands': 128, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_6': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_7': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_8': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 2, 'padding': 1}
            },
            'layer_9': {
                'conv': {'n_bands': 256, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_10': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_11': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_12': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 2, 'padding': 1}
            },
            'layer_13': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_14': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_15': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_16': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 2, 'padding': 1}
            }
        }

    if network_name == 'vgg19_no_stride':
        dictionary = {
            'n_layers': 16,
            'layer_1': {
                'conv': {'n_bands': 3, 'n_kernels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_2': {
                'conv': {'n_bands': 64, 'n_kernels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 1, 'padding': 1}
            },
            'layer_3': {
                'conv': {'n_bands': 64, 'n_kernels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_4': {
                'conv': {'n_bands': 128, 'n_kernels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 1, 'padding': 1}
            },
            'layer_5': {
                'conv': {'n_bands': 128, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_6': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_7': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_8': {
                'conv': {'n_bands': 256, 'n_kernels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 1, 'padding': 1}
            },
            'layer_9': {
                'conv': {'n_bands': 256, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_10': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_11': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_12': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 1, 'padding': 1}
            },
            'layer_13': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_14': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_15': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes'
            },
            'layer_16': {
                'conv': {'n_bands': 512, 'n_kernels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                'relu': 'yes',
                'max-pool': {'size': 2, 'stride': 1, 'padding': 1}
            }
        }

    return dictionary
