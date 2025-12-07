import argparse
import glob
from os.path import join as pjoin
import os
import shutil
import sys
import timeit


def build_argparse():
    prog_desc = \
        '''
- Run SAAD (Supervoxel-based Abnormal Asymmetry Detection) on NATIVE test image space. 
- Given a Test Image T and a AdaPro Model P, SAAD on NATIVE space has the following modules: 
  (1) Segment the Test Image T by P for the considered pair of symmetric objects (A, B); 
      (1.1) AdaPro's template R is registered on T during segmentation, resulting into R_on_T; 
      (1.2) Let the histogram of T equals to that of R_on_T,  
            only considering the regions of (A, B); 
  (2) Register T_A on T_B or viceversa (T_A is the object A of T); 
      (2.1) Let T_A with the same histogram of T_B after registration; 
  (3) For each image I of a training control set (previously registered on R): 
      (3.1) Mapping I on T by using the resulting deformation fields of (1); 
          (3.1.1) Let the histogram of I equals to that of R_on_T; 
      (3.2) Mapping I_L on I_R; 
          (3.2.1) Let the histogram of I_L equals to that of I_R after registration; 
      (3.3) Compute the asymmetry map ASYM(I) between I_L and I_R after registration; 
  (4) Compute the normal asymmetry map NASYM; 
  (5) Extracts the symmetric supervoxels S for T by using (2) and (4); 
  (6) Train a classifier for each supervoxel of S by using (3); 
  (7) Classify each supervoxel in S for T by using (7) 
The AdaPro model has the considered template T, only the pair of symmetric objects to be analized, and the
considered Elastix Parameter files that are used in SAAD"
        '''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-t', '--test-image', type=str, required=True, help='Pathname from the Test Image')
    parser.add_argument('-r', '--template', type=str, required=True, help='Template (Reference Image).')
    parser.add_argument('-m', '--adapro', type=str, required=True, help='AdaPro model used for segmentation (*.zip).')
    parser.add_argument('-i', '--train-set-entry', type=str, required=True,
                        help='Dir. or CSV with the pathname from the Train. Images (previously registered on to the template).')
    parser.add_argument('-l', '--pairs-of-symmetric-objects', type=str, required=True,
                        help='Define the pair of labels for registration: The first is registered on to the second.\n'
                             'e.g. 4:3 (object 4 is registered on to 3).')
    parser.add_argument('-n', '--number-of-supervoxels', type=int, required=True, help='Required number of supervoxels.')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--elastix-params', type=str, required=True, nargs='+',
                        help='Elastix Transformation Files to register the moving object to the fixed one.')
    parser.add_argument('-k', '--num-atlases', type=int, required=False, default=0, help='Number of chosen/selected atlases. Default: no atlas selection')
    parser.add_argument('-a', '--alpha', type=float, required=False, default=0.08, help='Alpha factor of SymmISF. Default: 0.08.')
    parser.add_argument('-b', '--beta', type=float, required=False, default=3, help='Beta factor of SymmISF. Default: 3.')
    parser.add_argument('-z', '--num-of-bins', type=int, required=False, default=128, help='Number of bins of the histogram. Default: 128.')
    parser.add_argument('--nu', type=float, required=False, default=0.1, help='NU parameter for linear oc-SVM. Default: 0.1')
    parser.add_argument('--skip-segmentation', action='store_true', help='Skip adapro segmentation')
    parser.add_argument('--skip-atlas-selection', action='store_true', help='Skip Atlas Selection')
    parser.add_argument('--skip-symm-object-registration', action='store_true',
                        help='Skip Registration between the symmetric objects')
    parser.add_argument('--skip-template-registration', action='store_true',
                        help='Skip Registration of the Template on NATIVE test image space')
    parser.add_argument('--skip-train-asym-maps-extraction', action='store_true',
                        help='Skip Extraction of the Train. Asymmetry Maps')
    parser.add_argument('--skip-native-SymmISF', action='store_true', help='Skip Native SymmISF')
    parser.add_argument('--skip-haa-feat-extraction', action='store_true', help='Skip Extraction of HAA feats')
    parser.add_argument('--skip-classification', action='store_true', help='Skip One-Class Training and Classification')

    return parser


def print_args(args):
    print("---------------------------------")
    print("- Test Image: %s" % args.test_image)
    print("- Template (Reference Image): %s" % args.template)
    print("- AdaPro Model: %s" % args.adapro)
    print("- Train Set Entry: %s" % args.train_set_entry)
    print("- Output Dir: %s" % args.output_dir)
    print("---------------------------------")
    print("- Elastix Params: {0}".format(args.elastix_params))
    print("- Number of Selected Atlas/training images: %d" % args.num_atlases)
    print("---------------------------------")
    print("- Moving Object: %d" % int(args.pairs_of_symmetric_objects.split(":")[0]))
    print("- Fixed Object: %d" % int(args.pairs_of_symmetric_objects.split(":")[1]))
    print("---------------------------------")
    print("- Alpha: %f" % args.alpha)
    print("- Beta: %f" % args.beta)
    print("---------------------------------")
    print("- Number of Bins: %d" % args.num_of_bins)
    print("---------------------------------")
    print("- NU parameter: %f" % args.nu)
    print("---------------------------------\n")


def formatted_time(time_in_secs):
    secs = time_in_secs

    days = secs // 86400
    secs = secs % 86400

    hours = secs // 3600
    secs = secs % 3600

    mins = secs // 60
    secs = secs % 60

    form_time = ""

    if days:
        form_time += "%d day(s) " % days
    if hours:
        form_time += "%d hour(s) " % hours
    if mins:
        form_time += "%d min(s) " % mins
    if secs:
        form_time += "%d sec(s) " % secs

    return form_time


def get_elastix_entry(elastix_params):
    elastix_entry = ""

    for i, param in enumerate(elastix_params):
        elastix_entry += "--t%d %s " % (i, param)

    return elastix_entry


def run_program(prog, prog_args):
    print("Program: %s" % prog)
    if os.system("%s %s" % (prog, prog_args)):
        sys.exit("Error when executing %s" % prog)


def adapro_segmentation(args):
    prog = "iftMultiNativeAdaProSegmentation"
    prog_args = "--test-image %s --adapro %s --output-dir %s" % (args.test_image, args.adapro, args.output_dir)
    run_program(prog, prog_args)


def atlas_selection(args):
    img_path = pjoin(args.output_dir, "image.nii.gz")
    mask_path = pjoin(args.output_dir, "mask.nii.gz")
    final_train_set = pjoin(args.output_dir, "selected_train_set.csv")
    elastix_entry = get_elastix_entry(args.elastix_params)

    prog = "iftMultiNativeAtlasSelectionByNMI"
    prog_args = "--test-img %s --template %s --train-img-entry %s --num-atlases %d --output-csv %s --test-mask %s %s" %\
                (img_path, args.template, args.train_set_entry, args.num_atlases, final_train_set, mask_path,
                 elastix_entry)
    run_program(prog, prog_args)


def register_symm_objects(args):
    img_path = pjoin(args.output_dir, "image.nii.gz")
    mask_path = pjoin(args.output_dir, "mask.nii.gz")
    elastix_entry = get_elastix_entry(args.elastix_params)

    prog = "iftMultiRegisterSymmObjects"
    prog_args = "--test-image %s --mask %s --pairs-of-symmetric-objects %s --output-dir %s %s" % \
                (img_path, mask_path, args.pairs_of_symmetric_objects, args.output_dir, elastix_entry)
    run_program(prog, prog_args)


def register_template_on_native(args):
    img_path = pjoin(args.output_dir, "image.nii.gz")
    elastix_entry = get_elastix_entry(args.elastix_params)

    prog = "iftMultiRegisterTemplateToNative"
    prog_args = "--test-image %s --template %s --output-dir %s %s" % (img_path, args.template,
                                                                      args.output_dir, elastix_entry)
    run_program(prog, prog_args)


def extract_train_asym_maps(args):
    test_img_path = pjoin(args.output_dir, "image.nii.gz")

    train_set = pjoin(args.output_dir, "selected_train_set.csv")
    if not os.path.exists(train_set):
        train_set = args.train_set_entry

    mask_path = pjoin(args.output_dir, "mask.nii.gz")

    df_native = sorted(glob.glob(pjoin(args.output_dir, "NativeDefFields.*")))
    df_native_entry = ""
    for i in range(len(df_native)):
        df_native_entry += "--d%d %s " % (i, df_native[i])

    df_objs = sorted(glob.glob(pjoin(args.output_dir, "SymmObjectDefFields.*")))
    df_objs_entry = ""
    for i in range(len(df_objs)):
        df_objs_entry += "--r%d %s " % (i, df_objs[i])

    prog = "iftExtractMultiNativeTrainAsymMaps"
    prog_args = "--test-image %s --train-set-entry %s --mask %s --pairs-of-symmetric-objects %s --output-dir " \
                "%s %s %s --use-stdev" % (test_img_path, train_set, mask_path, args.pairs_of_symmetric_objects,
                              args.output_dir, df_native_entry, df_objs_entry)
    run_program(prog, prog_args)


def extract_train_asym_maps_sol2(args):
    test_img_path = pjoin(args.output_dir, "image.nii.gz")

    train_set = pjoin(args.output_dir, "selected_train_set.csv")
    if not os.path.exists(train_set):
        train_set = args.train_set_entry

    mask_path = pjoin(args.output_dir, "mask.nii.gz")

    elastix_entry = get_elastix_entry(args.elastix_params)

    df_objs = sorted(glob.glob(pjoin(args.output_dir, "SymmObjectDefFields.*")))
    df_objs_entry = ""
    for i in range(len(df_objs)):
        df_objs_entry += "--r%d %s " % (i, df_objs[i])

    prog = "iftExtractMultiNativeTrainAsymMaps_Sol2"
    prog_args = "--test-image %s --train-set-entry %s --mask %s --pairs-of-symmetric-objects %s --output-dir " \
                "%s %s %s --use-stdev" % (test_img_path, train_set, mask_path, args.pairs_of_symmetric_objects,
                              args.output_dir, elastix_entry, df_objs_entry)
    run_program(prog, prog_args)


def native_SymmISF(args):
    reg_mov_img_path = pjoin(args.output_dir, "moving_flip_registered_image.nii.gz")
    fix_img_path = pjoin(args.output_dir, "fixed_image.nii.gz")
    fix_mask_path = pjoin(args.output_dir, "fixed_mask.nii.gz")
    symm_super_path = pjoin(args.output_dir, "n_supervoxels_%d" % args.number_of_supervoxels, "symm_supervoxels.nii.gz")
    normal_asymmap_path = pjoin(args.output_dir, "normal_asym_map.nii.gz")
    test_asymmap_path = pjoin(args.output_dir, "asym_map.nii.gz")

    inv_df_objs = sorted(glob.glob(pjoin(args.output_dir, "InverseSymmObjectDefFields.*")))
    inv_df_objs_entry = ""
    for i in range(len(inv_df_objs)):
        inv_df_objs_entry += "--r%d %s " % (i, inv_df_objs[i])

    prog = "iftMultiNativeSymmISF"
    prog_args = "--registered-moving-image %s --fixed-object-image %s --fixed-mask %s --number-of-supervoxels %s " \
                "--output-supervoxels-path %s --alpha %f --beta %f --normal-asymmetry-map %s " \
                "--output-asymmetry-map %s %s" % (reg_mov_img_path, fix_img_path, fix_mask_path,
                                                  args.number_of_supervoxels, symm_super_path, args.alpha, args.beta,
                                                  normal_asymmap_path, test_asymmap_path, inv_df_objs_entry)
    run_program(prog, prog_args)


def extract_haa_feats(args):
    img_path = pjoin(args.output_dir, "image.nii.gz")
    asymmap_path = pjoin(args.output_dir, "asym_map.nii.gz")
    template_on_native = pjoin(args.output_dir, "template_on_native.nii.gz")
    mask_path = pjoin(args.output_dir, "mask.nii.gz")
    train_set_on_native_entry = pjoin(args.output_dir, "train_set_on_native")
    train_asymmap_set_on_native_entry = pjoin(args.output_dir, "train_asym_maps")

    out_dir = pjoin(args.output_dir, "n_supervoxels_%d" % args.number_of_supervoxels)
    symm_super_path = pjoin(out_dir , "symm_supervoxels.nii.gz")

    prog = "iftExtractMultiNativeSupervoxelHAAFeats"
    prog_args = "--test-image %s --test-asymmetry-map %s --template-on-native %s --mask %s " \
                "--pairs-of-symmetric-objects %s --symmetric-supervoxels-path %s --train-set-on-native-entry %s " \
                "--train-asym-maps-on-native-entry %s --output-dir %s --num-of-bins %d" \
                % (img_path, asymmap_path, template_on_native, mask_path, args.pairs_of_symmetric_objects,
                   symm_super_path, train_set_on_native_entry, train_asymmap_set_on_native_entry, out_dir,
                   args.num_of_bins)
    run_program(prog, prog_args)


def classify_asymmetries(args):
    out_dir = pjoin(args.output_dir, "n_supervoxels_%d" % args.number_of_supervoxels)

    super_path = pjoin(out_dir, "supervoxels.nii.gz")
    if not os.path.exists(super_path):
        super_path = pjoin(out_dir, "symm_supervoxels.nii.gz")
    datasets_entry = pjoin(out_dir, "datasets_nbins_%d" % args.num_of_bins)
    out_classification = pjoin(out_dir, "classification_nbins_%d_nu_%f.nii.gz" % (args.num_of_bins, args.nu))

    prog = "python %s" % pjoin(os.environ["NEWIFT_DIR"], "demo", "BrainAsymmetryAnalysis", "NativeSAAD",
                               "classify_asymmetries_by_ocsvm.py")
    prog_args = "%s %s %s --kernel linear --nu %f" % (super_path, datasets_entry, out_classification, args.nu)
    run_program(prog, prog_args)


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    if not args.skip_segmentation:
        adapro_segmentation(args)

    if args.num_atlases > 0 and not args.skip_atlas_selection:
        atlas_selection(args)

    if not args.skip_symm_object_registration:
        register_symm_objects(args)

    if not args.skip_template_registration:
        register_template_on_native(args)

    if not args.skip_train_asym_maps_extraction:
        extract_train_asym_maps(args)
        # extract_train_asym_maps_sol2(args)

    if not args.skip_native_SymmISF:
        native_SymmISF(args)

    if not args.skip_haa_feat_extraction:
        extract_haa_feats(args)

    if not args.skip_classification:
        classify_asymmetries(args)

    train_regs_set_dir = pjoin(args.output_dir, "train_set_on_native")
    if os.path.exists(train_regs_set_dir):
        shutil.rmtree(train_regs_set_dir)

    train_asym_maps_dir = pjoin(args.output_dir, "train_asym_maps")
    if os.path.exists(train_asym_maps_dir):
        shutil.rmtree(train_asym_maps_dir)

    print("\nDone...")
    print("%s" % formatted_time(timeit.default_timer() - t1))


if __name__ == "__main__":
    main()

