import argparse
import os
import shutil
import tempfile

os.system(f"make iftSymmISF -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD")
os.system(f"make iftExtractSupervoxelHAAFeats -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD")



def build_argparse():
    prog_desc = \
        '''
        Detects abnormal asymmetries by SAAD.
        '''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image-path', type=str, required=True,
                        help='Input Image registered on a common template.')
    parser.add_argument('-m', '--right-hemisphere-mask-path', type=str, required=True,
                        help='Binary Mask with the right brain hemisphere.')
    parser.add_argument('-t', '--train-image-set-entry', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training control images "'
                             'on a common template.')
    parser.add_argument('-o', '--out-classification-mask', type=str, required=True,
                        help='Output Label Image with the detected lesions.')
    parser.add_argument('-s', '--normal-asymmetry-map', type=str, help='Normal asymmetry map used for SymmISF and "'
                                                                       'feature extraction.')
    parser.add_argument('-a', '--symmisf_alpha', type=float, default=0.06, help='Alpha factor of SymmISF. Default: 0.06')
    parser.add_argument('-b', '--symmisf_beta', type=float, default=5.0, help='Beta factor of SymmISF. Default: 5.0')
    parser.add_argument('-f', '--symmisf_otsu_threshold_factor', type=float, default=0.5,
                        help='Factor used to increase/decrease the otsu threshold used to binarize the assymetry map."'
                             'Default: 0.5')
    parser.add_argument('-d', '--symmisf_min_euclidean_distance', type=float, default=0.0,
                        help='Minimum euclidean distance to the binary mask borders that the initial seeds must have.'
                             'Default: 0.0 (no restriction).')
    parser.add_argument('-y', '--symmisf_num_seeds_on_symmetric_regions', type=int, default=100,
                        help='Number of Seeds on Symmetric Regions. Default: 100.')
    parser.add_argument('-n', '--bins', type=int, default=128, help='Number of histogram bins. Default: 128.')
    parser.add_argument('-k', '--ocsvm_kernel', type=str, default="linear", help='SVM kernel: linear or rbf. '
                                                                                 'Default: linear.')
    parser.add_argument('-u', '--ocsvm_nu', type=float, default=0.1, help='SVM NU parameter: Default: 0.1.')
    parser.add_argument('-g', '--ocsvm_gamma', type=float, default=0.01, help='GAMMA parameter for kernel rbf.'
                                                                              'Default: 0.01')
    parser.add_argument('-x', '--aux_dir', type=str, help='Directory to save auxiliary files')


    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Input Image: {args.image_path}')
    print(f'- Right Hemisphere Mask: {args.right_hemisphere_mask_path}')
    print(f'- Training Image Set Entry: {args.train_image_set_entry}')
    print(f'- Output Classification Mask: {args.out_classification_mask}')
    print('--------------------------------------------')
    if args.normal_asymmetry_map:
        print(f"- Normal Asymmetry Map: {args.normal_asymmetry_map}")
    print('--------------------------------------------')
    print("### SymmISF")
    print(f"- Alpha: {args.symmisf_alpha}")
    print(f"- Beta: {args.symmisf_beta}")
    print(f"- Factor on Otsu Threshold: {args.symmisf_otsu_threshold_factor}")
    print(f"- Erosion Radius: {args.symmisf_min_euclidean_distance}")
    print('--------------------------------------------')
    print("### Feature Extraction")
    print(f"- Number of bins: {args.bins}")
    print('--------------------------------------------')
    print("### OC-SVM")
    print(f"- Kernel: {args.ocsvm_kernel}")
    print(f"- NU: {args.ocsvm_nu}")
    if args.ocsvm_kernel == "rbf":
        print(f"- Gamma: {args.ocsvm_gamma}")
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
        working_dir = tempfile.mkdtemp(prefix="SAAD_")

    # SymmISF
    print("####### SymmISF")
    symm_svoxels_path = os.path.join(working_dir, "svoxels.nii.gz")
    symmisf_cmd = f"iftSymmISF --image-path {args.image_path} --binary-mask {args.right_hemisphere_mask_path} "\
                  f"--output-supervoxel-image-path {symm_svoxels_path} --alpha {args.symmisf_alpha} "\
                  f"--beta {args.symmisf_beta} --threshold-otsu-factor {args.symmisf_otsu_threshold_factor} "\
                  f"--min-euclidean-distance {args.symmisf_min_euclidean_distance} "\
                  f"--number-seeds-on-symmetric-regions {args.symmisf_num_seeds_on_symmetric_regions}"
    if args.normal_asymmetry_map:
        symmisf_cmd = f"{symmisf_cmd} --normal-asymmetry-map {args.normal_asymmetry_map}"

    print(symmisf_cmd)
    os.system(symmisf_cmd)

    # Feature Extraction: HAA feats
    print("\n\n####### Feature Extraction")
    datasets_dir = os.path.join(working_dir, "datasets")
    feat_extraction_cmd = f"iftExtractSupervoxelHAAFeats --test-image {args.image_path} "\
                          f"--test-symmetric-supervoxels {symm_svoxels_path} "\
                          f"--training-set {args.train_image_set_entry} --output-dir {datasets_dir} "\
                          f"--num-of-bins {args.bins}"
    if args.normal_asymmetry_map:
        feat_extraction_cmd = f"{feat_extraction_cmd} --normal-asymmetry-map {args.normal_asymmetry_map}"

    os.system(feat_extraction_cmd)

    # One-Class SVM
    print("\n\n####### One-Class SVM")
    ocsvm_cmd = f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/classify_asymmetries_by_ocsvm.py "\
                f"--symmetric-supervoxels-path {symm_svoxels_path} --supervoxels-datasets-entry {datasets_dir} "\
                f"--output-anomalous-supervoxels-img {args.out_classification_mask} "\
                f"--kernel {args.ocsvm_kernel} --nu {args.ocsvm_nu}"
    if args.ocsvm_kernel == "rbf":
        ocsvm_cmd = f"{ocsvm_cmd} --gamma {args.ocsvm_gamma}"

    os.system(ocsvm_cmd)

    if not args.aux_dir:
        shutil.rmtree(working_dir)



if __name__ == "__main__":
    main()
