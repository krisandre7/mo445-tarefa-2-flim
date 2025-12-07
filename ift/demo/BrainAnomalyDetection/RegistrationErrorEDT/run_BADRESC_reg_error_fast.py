import argparse
import os
import shutil
import tempfile
import timeit

SUPERVOXEL_SCRIPT_PATH = "$NEWIFT_DIR/demo/BrainAnomalyDetection/RegistrationErrorEDT/isf_on_reg_error_magnitude_fast.py"
FEAT_EXTRACTION_SCRIPT_PATH = "$NEWIFT_DIR/demo/BrainAnomalyDetection/RegistrationErrorEDT/extract_svoxel_feats.py"
OC_CLASSIFICATION_SCRIPT_PATH = "$NEWIFT_DIR/demo/BrainAnomalyDetection/RegistrationErrorEDT/classify_svoxel_datasets_by_ocsvm.py"

os.system("make iftAttenuatedRegErrorMagnitude -C $NEWIFT_DIR/demo/BrainAnomalyDetection/RegistrationErrorEDT")

def build_argparse():
    prog_desc = \
'''
Detects brain anomalies by BADRESC (Brain Anomaly Detection by Registration Error Magnitude and Supervoxel Classification).
'''
        
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image', type=str,
                        required=True, help='Registered Image pathname..')
    parser.add_argument('-t', '--template', type=str,
                        required=True, help='Template (Reference Image).')
    parser.add_argument('-m', '--label-image', type=str, required=True,
                        help='Label Image with with the target object where the svoxel segmentation will happen.')
    parser.add_argument('-x', '--training-reg-error-mag-set', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training control registration errors.')
    parser.add_argument('-o', '--out-classification-mask', type=str, required=True,
                        help='Output Label Image with the detected lesions.')

    parser.add_argument('-s', '--bias', type=str, help='Pathname from the bias (e.g. normal reg error magnitude) used '
                                                       'to attenuate the resulting registration error magnitude')
    
    parser.add_argument('-a', '--alpha-factors', type=float, nargs="+", default=[0.08],
                        help='Alpha factor of ISF, one for each Object of the Label Image. '
                             'Ex: -a 0.08,0.1,0.09 (3 objects) or -a 0.8 (same value for all objects). Default: 0.08')
    parser.add_argument('-b', '--beta-factors', type=float, nargs="+", default=[5],
                        help='Beta factor of ISF, one for each Object of the Label Image. '
                             'Ex: -b 5,2,4 (3 objects) or -b 5 (same value for all objects). Default: 5')
    parser.add_argument('-f', '--threshold-otsu-factors', type=float, nargs="+", default=[0.5],
                        help='Factor used to increase/decrease on otsu threshold to binarized reg error components. '
                             'Ex: -f 0.5 1.0 2.0 (3 objects) or -f 0.5 (same value for all objects). Default: 0.5')
    parser.add_argument('-d', '--min-euclidean-distances', type=float, nargs="+", default=[5],
                        help='Minimum euclidean distance to the each target object borders that the initial ISF seeds '
                             'must have. Ex: -d 0 1 2 (3 objects) or -f 5 (same value for all objects). '
                             'Default: 0.0 (no restriction).')
    parser.add_argument('-y', '--number-seeds-on-correct-regions', type=int, nargs="+", default=[50],
                        help='Number of Seeds on Correct Regions. Ex: -y 50 50 100 (3 objects) or -y 50 '
                             '(same value for all objects). Default: 50')
    
    parser.add_argument('-n', '--bins', type=int, default=128,
                        help='Number of histogram bins. Default: 128.')
    
    parser.add_argument('-k', '--kernel', type=str, default='linear',
                        help='SVM kernel: linear or rbf. Default: linear')
    parser.add_argument('-u', '--nu', type=float, default=0.1,
                        help='NU parameter. Default: 0.1')
    parser.add_argument('-g', '--gamma', type=float,
                        default=0.01, help='GAMMA parameter. Default: 0.01')

    parser.add_argument('-p', '--aux-dir', type=str,
                        help='Directory to save auxiliary files')

    parser.add_argument('--skip-registration-error-computation', action='store_true',
                        help='Skip the computation of registration error magnitudes for the input image. ' \
                             'It only works if the reg. error magnitude is already saved in the auxiliary directory')
    parser.add_argument('--skip-supervoxel-segmentation', action='store_true',
                        help='Skip Supervoxel Segmentation. It only works if the '
                              'supervoxel image is already saved in the auxiliary directory')
    parser.add_argument('--skip-feat-extraction', action='store_true',
                        help='Skip Feature Extraction. It only works if the '
                              'datasets are already saved in the auxiliary directory')




    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image: {args.image}')
    print(f'- Template: {args.template}')
    print(f'- Label Image: {args.label_image}')
    print(f'- Output Classification Mask: {args.out_classification_mask}')
    print('--------------------------------------------')
    print('### REGISTRATION ERRORS')
    if args.bias:
        print(f'- Bias: {args.bias}')
    print('--------------------------------------------')
    print('### SUPERVOXEL SEGMENTATION')
    print(f'- Alpha Factors: {args.alpha_factors}')
    print(f'- Beta Factors: {args.beta_factors}')
    print(f'- Threshold Otsu Factors: {args.threshold_otsu_factors}')
    print(f'- Min. Euclidean Distance from initial seeds to Object Borders: {args.min_euclidean_distances}')
    print(f'- Number of Seeds on Correct Regions: {args.number_seeds_on_correct_regions}')
    print('--------------------------------------------')
    print('### FEATURE EXTRACTION')
    print(f'- Training Set: {args.training_reg_error_mag_set}')
    print(f'- Number of Bins: {args.bins}')
    print('--------------------------------------------')
    print('### ONE-CLASS CLASSIFICATION')
    print('- Kernel: %s' % args.kernel)
    print('- NU: %f' % args.nu)
    print('- GAMMA: %f' % args.gamma)
    print('--------------------------------------------')
    if args.aux_dir:
        print(f"- Auxiliary Dir: {args.aux_dir}")
    print('--------------------------------------------\n\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    working_dir = ""
    if args.aux_dir:
        if not os.path.exists(args.aux_dir):
            os.makedirs(args.aux_dir)
        working_dir = args.aux_dir
    else:
        working_dir = tempfile.mkdtemp(prefix="BRADESC_")

    t1 = timeit.default_timer()

    reg_error_path = os.path.join(working_dir, "reg_error_mag.nii.gz")

    if not args.skip_registration_error_computation:
        reg_error_mag_cmd = f"iftAttenuatedRegErrorMagnitude --image-path {args.image} " \
                            f"--template {args.template} --label-image-path {args.label_image} " \
                            f"--out-reg-error-mag-map {reg_error_path} --attenuation-function none"
        if args.bias:
            reg_error_mag_cmd = f"{reg_error_mag_cmd} --bias {args.bias}"
        
        os.system(reg_error_mag_cmd)


    svoxels_path = os.path.join(working_dir, "svoxels.nii.gz")

    if not args.skip_supervoxel_segmentation:
        print("####### SVOXEL SEGMENTATION")
        svoxel_cmd_args = f"--image {args.image} --reg-error-mag {reg_error_path} --template {args.template} " \
                          f"--label-image {args.label_image} --out-supervoxel-image {svoxels_path} " \
                          f"--alpha-factors {' '.join(map(str, args.alpha_factors))} " \
                          f"--beta-factors {' '.join(map(str, args.beta_factors))} "\
                          f"--threshold-otsu-factors {' '.join(map(str, args.threshold_otsu_factors))} " \
                          f"--min-euclidean-distances {' '.join(map(str, args.min_euclidean_distances))} " \
                          f"--number-seeds-on-correct-regions {' '.join(map(str, args.number_seeds_on_correct_regions))}"
        
        os.system(f"python {SUPERVOXEL_SCRIPT_PATH} {svoxel_cmd_args}")


    datasets_dir = os.path.join(working_dir, "datasets")

    if not args.skip_feat_extraction:
        print("\n\n####### Feature Extraction")
        feat_extract_cmd_args = f"--test-reg-error-mag {reg_error_path} --test-supervoxels {svoxels_path} " \
                                f"--training-reg-error-mag-set {args.training_reg_error_mag_set} " \
                                f"--output-dir {datasets_dir} --bins {args.bins}"

        os.system(f"python {FEAT_EXTRACTION_SCRIPT_PATH} {feat_extract_cmd_args}")


    print("\n\n####### One-Class SVM")
    ocsvm_cmd_args = f"--supervoxels {svoxels_path} --supervoxels-datasets-entry {datasets_dir} " \
                     f"--output-anomalous-supervoxels-img {args.out_classification_mask} " \
                     f"--kernel {args.kernel} --nu {args.nu}"
    if args.kernel == "rbf":
        ocsvm_cmd_args = f"{ocsvm_cmd_args} -g {args.gamma}"
    
    os.system(f"python {OC_CLASSIFICATION_SCRIPT_PATH} {ocsvm_cmd_args}")



    print("\nDone...")
    print(f"#### Total Time: {timeit.default_timer() - t1} secs")

    if not args.aux_dir:
        shutil.rmtree(working_dir)

if __name__ == "__main__":
    main()
