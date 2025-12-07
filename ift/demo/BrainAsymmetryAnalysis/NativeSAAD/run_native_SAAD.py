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
- Given a Test Image T, a template R, a set of train control asymmetry maps S, a normal asymmetry map N, and a AdaPro Model P,
SAAD on NATIVE space has the following modules: 
Missing!!!
        '''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-t', '--test-image', type=str, required=True, help='Pathname from the Test Image. ' \
                                                                            'N4, median filter, and MSP alignment '
                                                                            'already done')
    parser.add_argument('-r', '--template', type=str, required=True, help='Template (Reference Image).')
    parser.add_argument('-m', '--template-mask', type=str, required=True, help='Template Mask with the target objects.')
    parser.add_argument('-p', '--adapro', type=str, required=True, help='AdaPro model used for segmentation (*.zip).')
    parser.add_argument('-l', '--pairs-of-symmetric-objects', type=str, required=True,
                        help='Define the pair of labels for registration: The first is registered on to the second.\n'
                             'e.g. 4:3 (object 4 is registered on to 3).')
    parser.add_argument('-i', '--train-set-entry', type=str, required=True,
                        help='Dir. or CSV with the pathname from the Train. Images '
                             '(previously registered on to the template and pre-processed).')
    parser.add_argument('-j', '--train-asym-maps-entry', type=str, required=True,
                        help='Dir. or CSV with the pathname from the Train. Asymmetry Maps '
                             '(previously registered on to the template).')
    parser.add_argument('-s', '--normal-asym-map', type=str, required=True, help='Normal Asymmetry Map.')
    parser.add_argument('-n', '--number-of-supervoxels', type=int, required=True, help='Required number of supervoxels.')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Output directory.')
    parser.add_argument('-x', '--registration', type=str, required=True, choices=['affine', 'nonrigid'], default='nonrigid',
                        help='Type of the registration between test image and template and vice-versa.')
    parser.add_argument('-e', '--object-registration', type=str, required=True, choices=['affine', 'nonrigid'],
                        default='affine', help='Type of the registration between the symmetric objects.')
    parser.add_argument('-a', '--alpha', type=float, required=False, default=0.08, help='Alpha factor of SymmISF. Default: 0.08.')
    parser.add_argument('-b', '--beta', type=float, required=False, default=3, help='Beta factor of SymmISF. Default: 3.')
    parser.add_argument('-d', '--eucl-dist-to-surface', type=float, required=False,
                        help='Euclidean distance between supervoxel\'s geometric center and object\'s surface used '
                             'to filter/remove supervoxels. (Try 5.0)')
    parser.add_argument('-v', '--min-supervoxel-volume', type=int, required=False,
                        help='Minimum volume that supervoxels near from the object\'s surface must have (Try 400).')
    parser.add_argument('-z', '--num-of-bins', type=int, required=False, default=128, help='Number of bins of the histogram. Default: 128.')
    parser.add_argument('--nu', type=float, required=False, default=0.1, help='NU parameter for linear oc-SVM. Default: 0.1')
    parser.add_argument('--skip-segmentation', action='store_true', help='Skip adapro segmentation')
    parser.add_argument('--skip-object-registration', action='store_true',
                        help='Skip Registration between the symmetric objects')
    parser.add_argument('--skip-normal-asym-map-transformation', action='store_true',
                        help='Skip Mapping of the Normal Asymmetric Map to NATIVE space')
    parser.add_argument('--skip-test-registration', action='store_true',
                        help='Skip Registration of the Test Image on to TEMPLATE space')
    parser.add_argument('--skip-native-SymmISF', action='store_true', help='Skip Native SymmISF')
    parser.add_argument('--skip-supervoxels-transformation', action='store_true',
                        help='Skip Mapping of the Supervoxels to TEMPLATE space')
    parser.add_argument('--skip-haa-feat-extraction', action='store_true', help='Skip Extraction of HAA feats')
    parser.add_argument('--skip-classification', action='store_true', help='Skip One-Class Training and Classification')

    return parser


def print_args(args):
    print("---------------------------------")
    print("- Test Image: %s" % args.test_image)
    print("- Template (Reference Image): %s" % args.template)
    print("- Template Mask: %s" % args.template_mask)
    print("- AdaPro Model: %s" % args.adapro)
    print("- Train Set Entry: %s" % args.train_set_entry)
    print("- Train Asymmetric Maps Entry: %s" % args.train_asym_maps_entry)
    print("- Normal Asymmetry Map: %s" % args.normal_asym_map)
    print("- Output Dir: %s" % args.output_dir)
    print("---------------------------------")
    print("- Registration (Test <--> Template): %s" % args.registration)
    print("---------------------------------")
    print("- Moving Object: %d" % int(args.pairs_of_symmetric_objects.split(":")[0]))
    print("- Fixed Object: %d" % int(args.pairs_of_symmetric_objects.split(":")[1]))
    print("- Registration (Fixed Object <--> Moving Object): %s" % args.object_registration)
    print("---------------------------------")
    print("- Number of Supervoxels: %d" % args.number_of_supervoxels)
    print("- Alpha: %f" % args.alpha)
    print("- Beta: %f" % args.beta)
    if args.eucl_dist_to_surface:
        print("- Eucl. Distance to Surface: %f" % args.eucl_dist_to_surface)
    if args.min_supervoxel_volume:
        print("- Min. Supervoxel Volume: %d" % args.min_supervoxel_volume)
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


def get_image_id(img_path):
    return img_path.split("/")[-1].split(".nii.gz")[0]


def get_elastix_entry(registration_type):
    if registration_type == "affine":
        elastix_entry = "--t0 %s" % pjoin(os.environ["NEWIFT_DIR"], "demo", "Registration", "ElastixParamFiles", "Par0000affine.txt")
    else:
        elastix_entry = "--t0 %s --t1 %s" % \
                        (pjoin(os.environ["NEWIFT_DIR"], "demo", "Registration", "ElastixParamFiles", "Par0000affine.txt"),
                         pjoin(os.environ["NEWIFT_DIR"], "demo", "Registration", "ElastixParamFiles", "Par0000bspline.txt"))

    return elastix_entry


def run_program(prog, prog_args):
    print("Program: %s" % prog)
    print("%s %s" % (prog, prog_args))

    if os.system("%s %s" % (prog, prog_args)):
        sys.exit("Error when executing %s" % prog)


def adapro_segmentation(args):
    prog = "iftNativeAdaProSegmentation"
    prog_args = "--test-image %s --adapro %s --output-dir %s" % (args.test_image, args.adapro, args.output_dir)
    run_program(prog, prog_args)


def register_symm_objects(args):
    img_id = get_image_id(args.test_image)

    img_path = pjoin(args.output_dir, "PreProcessedNative", "%s.nii.gz" % img_id)
    mask_path = pjoin(args.output_dir, "Mask", "%s.nii.gz" % img_id)
    elastix_entry = get_elastix_entry(args.object_registration)

    prog = "iftRegisterSymmObjects"
    prog_args = "--test-image %s --mask %s --pairs-of-symmetric-objects %s --output-dir %s %s" % \
                (img_path, mask_path, args.pairs_of_symmetric_objects, args.output_dir, elastix_entry)
    run_program(prog, prog_args)


def transform_norm_asym_map(args):
    img_id = get_image_id(args.test_image)
    out_path = pjoin(args.output_dir, "NormalAsymMap", "%s.nii.gz" % img_id)

    # adapro segmentation applies an affine registration followed by a nonrigd registration from the template to
    # the test image, resulting into two deformation fields (one for affine and other for deformable)
    def_fields = sorted(glob.glob(pjoin(args.output_dir, "TemplateOnNative", "%s*.txt" % img_id)))
    if args.registration == "affine":
        df = def_fields[0]
    else:
        df = def_fields[1]

    prog = "iftTransformImageByTransformix"
    prog_args = "--input-img %s --def-fields-file %s --output-img %s" % (args.normal_asym_map, df, out_path)
    run_program(prog, prog_args)


def register_test_on_template(args):
    elastix_entry = get_elastix_entry(args.registration)

    prog = "iftRegisterTestToTemplate"
    prog_args = "--test-image %s --template %s --output-dir %s %s" % (args.test_image, args.template,
                                                                      args.output_dir, elastix_entry)
    run_program(prog, prog_args)


def native_SymmISF(args):
    img_id = get_image_id(args.test_image)

    reg_mov_img_path = pjoin(args.output_dir, "MovingFlipRegisteredImage", "%s.nii.gz" % img_id)
    fix_img_path = pjoin(args.output_dir, "FixedImage", "%s.nii.gz" % img_id)
    fix_mask_path = pjoin(args.output_dir, "FixedMask", "%s.nii.gz" % img_id)
    symm_super_path = pjoin(args.output_dir, "SymmISF", "%04d" % args.number_of_supervoxels,
                            "SymmSupervoxels", "%s.nii.gz" % img_id)

    normal_asymmap_path = pjoin(args.output_dir, "NormalAsymMap", "%s.nii.gz" % img_id)
    test_asymmap_path = pjoin(args.output_dir, "AsymMap", "%s.nii.gz" % img_id)

    inv_df = sorted(glob.glob(pjoin(args.output_dir, "InverseSymmObjectDefFields", "%s*" % img_id)))[-1]

    prog = "iftNativeSymmISF"
    prog_args = "--registered-moving-image %s --fixed-object-image %s --fixed-mask %s --def-fields-file %s " \
                "--number-of-supervoxels %s --output-supervoxels-path %s --alpha %f --beta %f " \
                "--normal-asymmetry-map %s --output-asymmetry-map %s" % \
                (reg_mov_img_path, fix_img_path, fix_mask_path,inv_df, args.number_of_supervoxels, symm_super_path,
                 args.alpha, args.beta, normal_asymmap_path, test_asymmap_path)

    if args.eucl_dist_to_surface:
        prog_args += " --eucl-dist-to-surface %f" % args.eucl_dist_to_surface
    if args.min_supervoxel_volume:
        prog_args += " --min-supervoxel-volume %d" % args.min_supervoxel_volume
    run_program(prog, prog_args)


def transform_supervoxels_on_template(args):
    img_id = get_image_id(args.test_image)

    symm_super_path = pjoin(args.output_dir, "SymmISF", "%04d" % args.number_of_supervoxels,
                            "SymmSupervoxels", "%s.nii.gz" % img_id)
    out_path = pjoin(args.output_dir, "SymmISF", "%04d" % args.number_of_supervoxels, "SymmSupervoxelsOnTemplate",
                     "%s.nii.gz" % img_id)
    df = sorted(glob.glob(pjoin(args.output_dir, "NativeOnTemplate", "%s*.txt" % img_id)))[-1]

    prog = "iftTransformImageByTransformix"
    prog_args = "--input-img %s --def-fields-file %s --output-img %s" % (symm_super_path, df, out_path)
    run_program(prog, prog_args)


def extract_haa_feats(args):
    img_id = get_image_id(args.test_image)

    img_path = pjoin(args.output_dir, "PreProcessedNative", "%s.nii.gz" % img_id)
    asym_map_path = pjoin(args.output_dir, "AsymMap", "%s.nii.gz" % img_id)
    mask_path = pjoin(args.output_dir, "Mask", "%s.nii.gz" % img_id)

    out_dir = pjoin(args.output_dir, "SymmISF", "%04d" % args.number_of_supervoxels)
    symm_super_path = pjoin(args.output_dir, "SymmISF", "%04d" % args.number_of_supervoxels,
                            "SymmSupervoxels", "%s.nii.gz" % img_id)
    symm_super_template_path = pjoin(args.output_dir, "SymmISF", "%04d" % args.number_of_supervoxels, "SymmSupervoxelsOnTemplate",
                                     "%s.nii.gz" % img_id)
    template_on_native_path = pjoin(args.output_dir, "TemplateOnNative", "%s.nii.gz" % img_id)

    prog = "iftExtractNativeSupervoxelHAAFeats"
    prog_args = "--test-image %s --test-asymmetry-map %s --mask %s --pairs-of-symmetric-objects %s " \
                "--symm-super-path %s --symm-super-on-template-path %s --train-set %s --train-asym-maps %s " \
                "--output-dir %s --num-of-bins %d --template-on-native %s" \
                % (img_path, asym_map_path, mask_path, args.pairs_of_symmetric_objects, symm_super_path,
                   symm_super_template_path, args.train_set_entry, args.train_asym_maps_entry, out_dir,
                   args.num_of_bins, template_on_native_path)
    run_program(prog, prog_args)


def classify_asymmetries(args):
    img_id = get_image_id(args.test_image)

    out_dir = pjoin(args.output_dir, "SymmISF", "%04d" % args.number_of_supervoxels)
    super_path = pjoin(out_dir, "FilteredSymmSupervoxels", "%s.nii.gz" % img_id)

    if not os.path.exists(super_path):
        super_path = pjoin(out_dir, "SymmSupervoxels", "%s.nii.gz" % img_id)

    datasets_entry = pjoin(out_dir, "NumOfBins_%d" % args.num_of_bins, "DataSets", img_id)
    out_classification = pjoin(out_dir, "NumOfBins_%d" % args.num_of_bins, "Classification", "NU_%f" % args.nu,
                               "%s.nii.gz" % img_id)

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

    if not args.skip_object_registration:
        register_symm_objects(args)

    if not args.skip_normal_asym_map_transformation:
        transform_norm_asym_map(args)

    if not args.skip_test_registration:
        register_test_on_template(args)

    if not args.skip_native_SymmISF:
        native_SymmISF(args)

    if not args.skip_supervoxels_transformation:
        transform_supervoxels_on_template(args)

    if not args.skip_haa_feat_extraction:
        extract_haa_feats(args)

    if not args.skip_classification:
        classify_asymmetries(args)


    print("\nDone...")
    print("%s" % formatted_time(timeit.default_timer() - t1))


if __name__ == "__main__":
    main()

