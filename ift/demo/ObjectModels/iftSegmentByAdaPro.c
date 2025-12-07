/**
 * @file
 * @brief Segments an Image on training reference image's space by the Statistical Multi-Object Shape Model SOSM-S.
 * @note See the source code in @ref iftSegmentBySOSM.c
 *
 * @example iftSegmentBySOSM-S.c
 * @brief Segments an Image on training reference image's space by the Statistical Multi-Object Shape Model SOSM-S.
 * @author Samuel Martins
 * @date Dec 28, 2016
 */

#include "ift.h"
#include "ift/medical/brain/PreProcessing.h"
#include "ift/medical/segm/AdaPro.h"
#include "ift/medical/registration/Elastix.h"


typedef enum {NONE, TEMPLATE, NATIVE} iftRegistrationSpace;


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **adapro_path, char **out_mask_path);
void iftValidateRequiredArgs(const char *test_img_path, const char *adapro_path, const char *out_mask_path);
void iftGetOptionalArgs(  iftDict *args, const char *test_img_path, iftRegistrationSpace *registration_space,
                        char **aux_basename, bool *skip_n4, bool *skip_median_filter, bool *skip_msp_alignment,
                        bool *skip_hist_matching, bool *skip_pre_processing);

iftFloatArray *iftParseRadiusEntry(const char *radius_entry,   iftIntArray *labels);
void iftSetNewRadius(  iftDict *args, iftAdaPro *adapro);
void iftWriteAuxFiles(  iftImage *test_img,   iftAdaPro *adapro,   iftPlane *msp,   iftFileSet *def_fields,
                      const char *aux_basename);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path = NULL;
    char *adapro_path   = NULL;
    char *out_mask_path = NULL;
    // optional args
    iftRegistrationSpace registration_space = NONE;
    char *aux_basename = NULL;
    bool skip_n4 = true;
    bool skip_median_filter = true;
    bool skip_msp_alignment = true;
    bool skip_hist_matching = true;
    bool skip_pre_processing = true;
    
    iftGetRequiredArgs(args, &test_img_path, &adapro_path, &out_mask_path);
    iftGetOptionalArgs(args, test_img_path, &registration_space, &aux_basename,
                       &skip_n4, &skip_median_filter, &skip_msp_alignment, &skip_hist_matching, &skip_pre_processing);

    timer *t1 = iftTic();

    iftAdaPro *adapro = iftReadAdaPro(adapro_path);
    iftSetNewRadius(args, adapro);

    puts("\n- Reading Test Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);

    iftPlane *msp = NULL;
    // pre-processing: N4 and/or Median Filter and/or MSP alignment
    if (!skip_pre_processing && (!skip_n4 || !skip_median_filter || !skip_msp_alignment)) {
        iftImage *aux = test_img;
        puts("- N4 Bias Field Correction + Median Filter + MSP Alignment");
        int nbits = iftImageDepth(adapro->template_img);
        test_img = iftBrainMRIPreProcessing(test_img, nbits, NULL, NULL, NULL, NULL, skip_n4, skip_median_filter, skip_msp_alignment, true,
                                            &msp);
        iftDestroyImage(&aux);
    }
    
    // registration
    iftFileSet *def_fields = NULL;
    if (registration_space == NONE)
        iftVerifyImages(test_img, adapro->template_img, "main");
    else {
        char *tmp_basename = iftMakeTempPathname("ElastixParams.", NULL, NULL);
        iftFileSet *elastix_files = iftWriteAdaProElastixParamFiles(adapro, tmp_basename);
        
        if (registration_space == TEMPLATE) {
            puts("- Registering Test Image on TEMPLATE Coordinate Space");
            iftImage *reg_img = iftRegisterImageByElastix(test_img, adapro->template_img, NULL, NULL,
                                                          elastix_files, NULL, &def_fields, NULL);
            iftDestroyImage(&test_img);
            test_img = reg_img;
        }
        else if (registration_space == NATIVE) {
            puts("- Registering AdaPro on NATIVE Test Image's Coordinate Space");
            iftRegisterAdaProOnTestImageByElastix(adapro, test_img, elastix_files, &def_fields);
        }
        
        // cleaning up
        iftFree(tmp_basename);
        iftRemoveFileSet(elastix_files);
        iftDestroyFileSet(&elastix_files);
    }


    // Histogram Matching
    if (!skip_pre_processing && !skip_hist_matching) {
        iftImage *aux = test_img;
        puts("- Histogram Matching");
        iftImage *rough_mask_dilated = iftDilateAdaProRoughSegmentation(adapro);
        test_img = iftBrainMRIPreProcessing(test_img, 0, NULL, rough_mask_dilated, adapro->template_img, rough_mask_dilated, true, true, true, false, NULL);
        iftDestroyImage(&aux);
        iftDestroyImage(&rough_mask_dilated);
    }

    if (aux_basename != NULL)
        iftWriteAuxFiles(test_img, adapro, msp, def_fields, aux_basename);

    puts("- Segmenting the Test Image by AdaPro");
    iftImage *seg_img = iftSegmentByAdaPro(test_img, adapro, aux_basename);
    iftWriteImageByExt(seg_img, out_mask_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftFree(adapro_path);
    iftFree(out_mask_path);
    iftDestroyImage(&test_img);
    iftDestroyAdaPro(&adapro);
    iftDestroyImage(&seg_img);
    iftRemoveFileSet(def_fields);
    iftDestroyFileSet(&def_fields);
    

    return 0;
}




/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[IFT_STR_DEFAULT_SIZE] = \
        "- Segment a test image by Adaptive Probabilistic Atlas AdaPro.\n" \
        "- By default, the segmentation is performed on TEMPLATE coordinate space, assuming that the test image " \
        "is already registered on this space.\n"
        "- Otherwise, choose -r TEMPLATE for registration and delineation on TEMPLATE coordinate space or -r NATIVE for those on NATIVE space.\n\n"
        "- Three pre-processing tasks are applied to the test image:\n" \
        "1) N4 Bias Field Correction\n" \
        "2) Median Filtering\n" \
        "3) (After Registration) Histogram Matching with AdaPro's Template (it is only applied into AdaPro's rough segmentation)\n\n" \
        "- You can skip all pre-processing steps with the option --skip-pre-processing or to select which tasks to skip.\n\n" \
        "- You can pass custom erosion and dilation radius for segmentation instead of using those set in AdaPro model.\n" \
        "- For that, you must assign such parameters for each object of adapro, keeping the order " \
        "of how they were stored, or a single pair for all objects.\n" \
        "Exs:\n" \
        "-e 0 -d 0,2,2 (3 objects)\n" \
        "- Finally, you can pass an output aux dirctory (option -p) to store the resulting AdaPro, pre-processed Test Image, and everything used in delineation.";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image to be segmented. It might have to be previously registered on training " \
                                "template space, or it will be in this program."},
        {.short_name = "-b", .long_name = "--adapro", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="AdaPro model used for segmentation (*.zip)."},
        {.short_name = "-o", .long_name = "--output-segmentation-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output Segmented Mask."},
        {.short_name = "-e", .long_name = "--erosion-radius", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Set of Erosion Radius (one for each object or one overall radius) " \
                                "to find the Object Seeds from the Prob. Atlases (Borders from the Objects' Certain Regions).\n" \
                                 "Ex: -e 0,3,3 or -e 10"},
        {.short_name = "-d", .long_name = "--dilation-radius", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Set of Dilation Radius (one for each object or one overall radius) " \
                                "to find the Bacground Seeds from the Prob. Atlas (Border from the Background's Certain Region).\n" \
                                 "Ex: -d 0,5,5 or -d 10"},
        {.short_name = "-r", .long_name = "--registration-space", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Options: NATIVE or TEMPLATE. Apply registration for segmentation on NATIVE or TEMPLATE space.\nDefault: TEMPLATE"},
        {.short_name = "-p", .long_name = "--aux-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Auxiliary directory used to store the AdaPro model, pre-processed Test Image, and all used in delineation."},
        {.short_name = "", .long_name = "--skip-pre-processing", .has_arg=false,
         .required=false, .help="Skip ALL pre-processing operations."},
        {.short_name = "", .long_name = "--skip-n4", .has_arg=false,
         .required=false, .help="Skip N4 Bias Field Correction."},
        {.short_name = "", .long_name = "--skip-median-filter", .has_arg=false,
         .required=false, .help="Skip Median Filter."},
        {.short_name = "", .long_name = "--skip-msp-alignment", .has_arg=false,
         .required=false, .help="Skip Alignment by Mid-Sagittal Plane."},
        {.short_name = "", .long_name = "--skip-hist-matching", .has_arg=false,
         .required=false, .help="Skip Histogram Matching."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **adapro_path, char **out_mask_path) {
    *test_img_path = iftGetStrValFromDict("--test-image", args);
    *adapro_path    = iftGetStrValFromDict("--adapro", args);
    *out_mask_path  = iftGetStrValFromDict("--output-segmentation-mask", args);

    iftValidateRequiredArgs(*test_img_path, *adapro_path, *out_mask_path);

    puts("-----------------------");
    printf("- Test Image: \"%s\"\n", *test_img_path);
    printf("- AdaPro: \"%s\"\n", *adapro_path);
    printf("- Output Segmented Image: \"%s\"\n", *out_mask_path);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *test_img_path, const char *adapro_path, const char *out_mask_path) {
    if (!iftIsImageFile(test_img_path))
        iftError("Invalid Input Test Image: \"%s\"", "iftValidateRequiredArgs", test_img_path);

    // MODEL
    if (iftFileExists(adapro_path)) {
        if (!iftEndsWith(adapro_path, ".zip")) {
            iftError("Invalid Extension for the AdaPro: \"%s\"... Try *.zip",
                     "iftValidateRequiredArgs", adapro_path);
        }
    }
    else iftError("AdaPro \"%s\" does not exist", "iftValidateRequiredArgs", adapro_path);

    // OUTPUT SEGMENTED IMAGE
    if (!iftIsImagePathnameValid(out_mask_path)) {
        iftError("Invalid Output Image's Pathname: \"%s\"", "iftValidateRequiredArgs", out_mask_path);
    }

    char *parent_dir = iftParentDir(out_mask_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    iftFree(parent_dir);
}


void iftGetOptionalArgs(  iftDict *args, const char *test_img_path, iftRegistrationSpace *registration_space,
                        char **aux_basename, bool *skip_n4, bool *skip_median_filter, bool *skip_msp_alignment,
                        bool *skip_hist_matching, bool *skip_pre_processing) {
    if (iftDictContainKey("--registration-space", args, NULL)) {
        const char *registration_space_str = iftGetConstStrValFromDict("--registration-space", args);

        if (iftCompareStrings(registration_space_str, "TEMPLATE")) {
            *registration_space = TEMPLATE;
            printf("- Registration Space: TEMPLATE\n");
        }
        else if (iftCompareStrings(registration_space_str, "NATIVE")) {
            *registration_space = NATIVE;
            printf("- Registration Space: NATIVE\n");
        }
        else iftError("Invalid registration space: %s. Try NATIVE or TEMPLATE", "iftGetOptionalArgs", registration_space_str);
    }
    else *registration_space = NONE;


    if (iftDictContainKey("--aux-dir", args, NULL)) {
        const char *aux_dir = iftGetConstStrValFromDict("--aux-dir", args);
        char *test_img_basename = iftFilename(test_img_path, iftFileExt(test_img_path));
        *aux_basename = iftJoinPathnames(2, aux_dir, test_img_basename);
        iftFree(test_img_basename);

        printf("- Output Auxiliary Directory: %s\n", aux_dir);
        printf("- Output Auxiliary Basename: %s\n", *aux_basename);
    }


    *skip_pre_processing = iftDictContainKey("--skip-pre-processing", args, NULL);
    *skip_n4 = iftDictContainKey("--skip-n4", args, NULL);
    *skip_median_filter = iftDictContainKey("--skip-median-filter", args, NULL);
    *skip_msp_alignment = iftDictContainKey("--skip-msp-alignment", args, NULL);
    *skip_hist_matching = iftDictContainKey("--skip-hist-matching", args, NULL);
    
    printf("- Skip Pre-Processing: %s\n", iftBoolAsString(*skip_pre_processing));
    printf("- Skip N4 Bias Field Correction: %s\n", iftBoolAsString(*skip_n4));
    printf("- Skip Median Filter: %s\n", iftBoolAsString(*skip_median_filter));
    printf("- Skip MSP Alignment: %s\n", iftBoolAsString(*skip_msp_alignment));
    printf("- Skip Histogram Matching: %s\n", iftBoolAsString(*skip_hist_matching));
    puts("-----------------------");
}


iftFloatArray *iftParseRadiusEntry(const char *radius_entry,   iftIntArray *labels) {
    if (!iftRegexMatch(radius_entry, "^([0-9]+(.[0-9]+)?,?)+$"))
        iftError("Invalid Radius Entry: %s", "iftParseRadiusEntry", radius_entry);

    iftSList *SL = iftSplitString(radius_entry, ",");
    long n_entry_elems = SL->n;
    
    if ((n_entry_elems != 1) && (n_entry_elems != labels->n))
        iftError("Number of radius passed is != Number of Objects from the AdaPro: %d != %d\n" \
                 "Pass a single interval or one for each object\nEntry: %s", "iftParseRadiusEntry",
                 n_entry_elems, labels->n, radius_entry);

    iftFloatArray *radius_arr = iftCreateFloatArray(labels->n);

    for (int i = 0; SL->n != 0; i++)
        radius_arr->val[i] = atof(iftRemoveSListHead(SL));
    iftDestroySList(&SL);

    // only one overall radius was passed, then fill all radius with it
    if (n_entry_elems == 1)
        for (int i = 1; i < labels->n; i++)
            radius_arr->val[i] = radius_arr->val[0];

    return radius_arr;
}


void iftSetNewRadius(  iftDict *args, iftAdaPro *adapro) {
    iftFloatArray *e_radius_arr = NULL, *d_radius_arr = NULL;

    if (iftDictContainKey("--erosion-radius", args, NULL)) {
        e_radius_arr = iftParseRadiusEntry(iftGetConstStrValFromDict("--erosion-radius", args), adapro->labels);

        for (int o = 0; o < adapro->labels->n; o++)
            adapro->obj_models[o]->erosion_radius = e_radius_arr->val[o];
        iftDestroyFloatArray(&e_radius_arr);
    }

    if (iftDictContainKey("--dilation-radius", args, NULL)) {
        d_radius_arr = iftParseRadiusEntry(iftGetConstStrValFromDict("--dilation-radius", args), adapro->labels);

        for (int o = 0; o < adapro->labels->n; o++)
            adapro->obj_models[o]->dilation_radius = d_radius_arr->val[o];
        iftDestroyFloatArray(&d_radius_arr);
    }

    printf("- Erosion Radius: ");
    for (int o = 0; o < adapro->labels->n; o++)
        printf("%f ", adapro->obj_models[o]->erosion_radius);

    printf("\n- Dilation Radius: ");
    for (int o = 0; o < adapro->labels->n; o++)
        printf("%f ", adapro->obj_models[o]->dilation_radius);
    puts("");
    puts("-----------------------");
}


void iftWriteAuxFiles(  iftImage *test_img,   iftAdaPro *adapro,   iftPlane *msp,   iftFileSet *def_fields,
                      const char *aux_basename) {
    puts("\t- Writing Aux Files: Pre-processed Test Image and AdaPro");
    iftWriteImageByExt(test_img, iftConcatStrings(2, aux_basename, ".nii.gz"));
    iftWriteAdaPro(adapro, iftConcatStrings(2, aux_basename, "_adapro.zip"));
    iftWritePlane(msp, "%s_msp.json", aux_basename);
    
    if (def_fields) {
        iftFileSet *new_def_fields = iftCreateFileSet(def_fields->n);
        
        puts("\t- Writing Deformation Fields");
        for (long i = 0; i < def_fields->n; i++) {
            char out_def_field[512];
            sprintf(out_def_field, "%s_DefFields.%ld.txt", aux_basename, i);
            new_def_fields->files[i] = iftCreateFile(out_def_field);
            
            printf("[%ld] %s --> %s\n", i, def_fields->files[i]->path, new_def_fields->files[i]->path);
            iftCopyFromDisk(def_fields->files[i]->path, new_def_fields->files[i]->path);
            iftRemoveFile(def_fields->files[i]->path);
            
            if (i > 0)
                iftSed(new_def_fields->files[i]->path, def_fields->files[i - 1]->path, new_def_fields->files[i - 1]->path);
        }
        puts("");
        
        iftDestroyFileSet(&new_def_fields);
    }
}
/*************************************************************/











