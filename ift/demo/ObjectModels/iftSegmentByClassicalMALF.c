/**
 * @file
 * @brief Segments an Image by Classical MALF.
 * @note See the source code in @ref iftSegmentByClassicalMALF.c
 *
 * @example iftSegmentByClassicalMALF.c
 * @brief Segments an Image by Classical MALF.
 * @author Samuel Martins
 * @date Jan 6, 2017
 */



#include "ift.h"
#include "ift/medical/segm/MALF.h"
#include "ift/medical/registration/Elastix.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **train_label_entry, char **out_img_path);
void iftValidateRequiredArgs(const char *test_img_path, const char *train_label_entry, char *out_img_path);
void iftGetOptionalArgs(  iftDict *args, char **train_img_entry, bool *register_img, char **affine_params_path,
                        char **bspline_params_path);
void iftValidateOptionalArgs(const char *train_img_entry, bool register_img, const char *affine_params_path,
                             const char *bspline_params_path);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path     = NULL;
    char *train_label_entry = NULL;
    char *out_img_path      = NULL;
    // optional args
    char *train_img_entry   = NULL;
    bool register_img         = false;
    char *affine_params_path  = NULL;
    char *bspline_params_path = NULL;

    iftGetRequiredArgs(args, &test_img_path, &train_label_entry, &out_img_path);
    iftGetOptionalArgs(args, &train_img_entry, &register_img, &affine_params_path, &bspline_params_path);


    timer *t1 = iftTic();

    puts("- Reading Testing Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);
    int img_depth      = iftImageDepth(test_img);

    iftFileSet *train_label_set = NULL;

    char *reg_label_tmp_dir = NULL;
    if (register_img) {
        char *def_fields_tmp_dir = iftMakeTempDir("tmp_dir_", NULL, NULL);
        iftRunProgRegisterImageSetByElastix(train_img_entry, test_img_path, img_depth,
                                            affine_params_path, bspline_params_path,
                                            def_fields_tmp_dir);

        char *filename    = iftFilename(test_img_path, iftFileExt(test_img_path));
        reg_label_tmp_dir = iftMakeTempDir(iftConcatStrings(3, "reg_labels_on_", filename, "_"), NULL, NULL);

        iftRunProgTransformImageSetByTransformix(train_label_entry, def_fields_tmp_dir, def_fields_tmp_dir, reg_label_tmp_dir);

        train_label_set = iftLoadFileSetFromDirBySuffix(reg_label_tmp_dir, iftImageExt(test_img), 0);

        iftRemoveDir(def_fields_tmp_dir);
        iftFree(def_fields_tmp_dir);
        iftFree(filename);
    }
    // label mask are already registered
    else {
        train_label_set = iftLoadFileSetFromDirOrCSV(train_label_entry, 0, true);
    }


    puts("- Segmenting by Classical MALF");
    iftImage *seg_img = iftSegmentByClassicalMALF(test_img, train_label_set);
    iftWriteImageByExt(seg_img, out_img_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftDestroyFileSet(&train_label_set);
    iftFree(affine_params_path);
    iftFree(bspline_params_path);
    iftFree(out_img_path);
    if (reg_label_tmp_dir != NULL) {
        iftRemoveDir(reg_label_tmp_dir);
        iftFree(reg_label_tmp_dir);
    }
    iftDestroyImage(&test_img);
    iftDestroyImage(&seg_img);

    return 0;
}




/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segments an image by Classical MALF.\n" \
        "- Classical MALF registers all atlases (img + label img) on test image's space. The label of " \
        "each voxel from segmentated image is the most frequent label (Majority Voting).\n" \
        "- Input atlases might already have to be previously registered on test image's space, " \
        "or it will be in this program by Elastix.\n\n" \
        "--> FOR REGISTRATION, COMPILE THE PROGRAMS:\n" \
        "iftRegisterImageSetByElastix\n" \
        "iftTransformImageSetByTransformix";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image to be segmented."},
        {.short_name = "-l", .long_name = "--train-label-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Train. Label Images or a CSV file with their pathnames. " \
                               "It might already have to be previously registered on test image's space, " \
                                "or it will be in this program"},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output Segmented Image."},
        {.short_name = "-t", .long_name = "--train-img-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Dir with the Train. Images or a CSV file with their pathnames. " \
                                "Only required if the atlases are not registered."},
        {.short_name = "-r", .long_name = "--register-image", .has_arg=false,
         .required=false, .help="Register all train atlases on test image before segmentation by Elastix.\n" \
                                "Compile the programs iftRegisterImageSetByElastix and iftTransformImageSetByTransformix"},
        {.short_name = "-a", .long_name = "--affine-param-file", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Affine (or Rigid or another) Parameter File"},
        {.short_name = "-b", .long_name = "--bspline-param-file", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix BSpline Parameter File (required for Non-Rigid Registration)"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **train_label_entry, char **out_img_path) {
    *test_img_path     = iftGetStrValFromDict("--test-img", args);
    *train_label_entry = iftGetStrValFromDict("--train-label-entry", args);
    *out_img_path      = iftGetStrValFromDict("--output-img", args);
    
    iftValidateRequiredArgs(*test_img_path, *train_label_entry, *out_img_path);

    puts("-----------------------");
    printf("- Test Image: \"%s\"\n", *test_img_path);
    printf("- Train GT Image Entry: \"%s\"\n", *train_label_entry);
    printf("- Output Segmented Image: \"%s\"\n", *out_img_path);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *test_img_path, const char *train_label_entry, char *out_img_path) {
    // INPUT TEST IMAGE
    if (!iftIsImageFile(test_img_path))
        iftError("Invalid Input Test Image: \"%s\"", "iftValidateRequiredArgs", test_img_path);



    // LABEL IMAGE ENTRY
    if (iftFileExists(train_label_entry)) {
        if (!iftEndsWith(train_label_entry, ".csv")) {
            iftError("Train Label Image Set Entry File \"%s\" is not a CSV file... Try *.csv", "iftValidateRequiredArgs", train_label_entry);
        }
    }
    else if (!iftDirExists(train_label_entry))
        iftError("Invalid Pathname for the Label Image Set Entry: \"%s\"...\nTry a valid Directory or a CSV file",
                 "iftValidateRequiredArgs", train_label_entry);

    // OUTPUT SEGMENTED IMAGE
    if (!iftIsImagePathnameValid(out_img_path)) {
        iftError("Invalid Output Grad Image's Pathname: \"%s\"", "iftValidateRequiredArgs", out_img_path);
    }

    char *parent_dir = iftParentDir(out_img_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
}


void iftGetOptionalArgs(  iftDict *args, char **train_img_entry, bool *register_img, char **affine_params_path,
                        char **bspline_params_path) {
    if (iftDictContainKey("--train-img-entry", args, NULL))
        *train_img_entry = iftGetStrValFromDict("--train-img-entry", args);
    else *train_img_entry = NULL;

    *register_img = iftDictContainKey("--register-image", args, NULL);

    if (iftDictContainKey("--affine-param-file", args, NULL))
        *affine_params_path = iftGetStrValFromDict("--affine-param-file", args);

    if (iftDictContainKey("--bspline-param-file", args, NULL))
        *bspline_params_path = iftGetStrValFromDict("--bspline-param-file", args);

    iftValidateOptionalArgs(*train_img_entry, *register_img, *affine_params_path, *bspline_params_path);

    if (*register_img) {
        if (*affine_params_path != NULL)
            printf("- Elastix Affine (or Ridig or another) Param. File: \"%s\"\n", *affine_params_path);
        if (*bspline_params_path != NULL)
            printf("- Elastix BSpline Param. File: \"%s\"\n", *bspline_params_path);
    }
    puts("-----------------------\n");
}


void iftValidateOptionalArgs(const char *train_img_entry, bool register_img, const char *affine_params_path,
                             const char *bspline_params_path) {
    // ELASTIX PARAMS FILES
    if (register_img) {
        if (train_img_entry == NULL)
            iftError("Train. Images are missing for registration", "iftValidateOptionalArgs");
        else {
            if (iftFileExists(train_img_entry)) {
                if (!iftEndsWith(train_img_entry, ".csv")) {
                    iftError("Train Image Set Entry File \"%s\" is not a CSV file... Try *.csv", "iftValidateOptionalArgs", train_img_entry);
                }
            }
            else if (!iftDirExists(train_img_entry))
                iftError("Invalid Pathname for the Image Set Entry: \"%s\"...\nTry a valid Directory or a CSV file",
                         "iftValidateOptionalArgs", train_img_entry);
        }

        if ((affine_params_path == NULL) && (bspline_params_path == NULL))
            iftError("Elastix Affine and BSpline Configuration Files were not passed. Pass one of them at least",
                "iftValidateOptionalArgs");
        else {    
            // Elastix Affine Configuration File
            if ((affine_params_path != NULL) && !iftFileExists(affine_params_path))
                iftError("Elastix Affine (or Rigid or another) Configuration File does not exist or it is a directory!\n--> Pathname: \"%s\"", "iftValidateOptionalArgs", affine_params_path);

            // Elastix BSpline Configuration File
            if ((bspline_params_path != NULL) && !iftFileExists(bspline_params_path))
                iftError("Elastix BSpline Configuration File does not exist or it is a directory!\n--> Pathname: \"%s\"", "iftValidateOptionalArgs", bspline_params_path);
        }
    }
}
/*************************************************************/











