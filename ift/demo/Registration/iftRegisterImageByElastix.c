/**
 * @author Samuel Martins
 * @date Fev 18, 2016
 */

#include "ift.h"
#include "ift/medical/registration/Elastix.h"


/************************** HEADERS **************************/
iftDict *iftGetArguments(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **moving_img_path, char **fixed_img_path, char **out_reg_img_path,
                        iftFileSet **transf_files);
void iftGetOptionalArgs(  iftDict *args, char **moving_mask_path, char **fixed_mask_path);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArguments(argc, argv);

    // required args
    char *moving_img_path    = NULL;
    char *fixed_img_path     = NULL;
    char *out_reg_img_path   = NULL;
    iftFileSet *transf_files = NULL;

    // optional args
    char *moving_mask_path = NULL;
    char *fixed_mask_path = NULL;

    iftGetRequiredArgs(args, &moving_img_path, &fixed_img_path, &out_reg_img_path, &transf_files);
    iftGetOptionalArgs(args, &moving_mask_path, &fixed_mask_path);

    timer *t1 = iftTic();
    
    puts("- Reading Moving Image");
    iftImage *moving_img = iftReadImageByExt(moving_img_path);

    puts("- Reading Fixed Image");
    iftImage *fixed_img = iftReadImageByExt(fixed_img_path);

    iftImage *fixed_mask = NULL;
    if (fixed_mask_path != NULL) {
        puts("- Reading Fixed Mask");
        fixed_mask = iftReadImageByExt(fixed_mask_path);
    }

    iftImage *moving_mask = NULL;
    if (moving_mask_path != NULL) {
        puts("- Reading Fixed Mask");
        moving_mask = iftReadImageByExt(moving_mask_path);
    }

    puts("- Register Image by Elastix");
    char *out_basename = (iftDictContainKey("--no-save-def-fields", args, NULL)) ? NULL : iftBasename(out_reg_img_path);
    iftFileSet *def_fields = NULL;
    iftImage *reg_img = iftRegisterImageByElastix(moving_img, fixed_img, moving_mask, fixed_mask, transf_files, out_basename, &def_fields, NULL);
    
    if (iftDictContainKey("--apply-minmax-normalization-without-outliers", args, NULL)) {
        puts("- Apply MinMax Normalization Without Outliers");
        int max_range = iftMaxImageRange(iftImageDepth(fixed_img));
        iftImage *norm = iftNormalizeWithNoOutliers(reg_img, 0, max_range, 0.98);
        iftDestroyImage(&reg_img);

        reg_img = norm;
    }

    if (!iftDictContainKey("--no-save-reg-img", args, NULL))
        iftWriteImageByExt(reg_img, out_reg_img_path);
    if (iftDictContainKey("--no-save-def-fields", args, NULL)) {
        for (int i = 0; i < def_fields->n; i++)
            iftRemoveFile(def_fields->files[i]->path);
    }

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(moving_img_path);
    iftFree(fixed_img_path);
    iftFree(out_reg_img_path);
    iftDestroyFileSet(&transf_files);
    iftFree(moving_mask_path);
    iftFree(fixed_mask_path);
    iftFree(out_basename);
    iftDestroyImage(&moving_img);
    iftDestroyImage(&fixed_img);
    iftDestroyImage(&moving_mask);
    iftDestroyImage(&fixed_mask);
    iftDestroyImage(&reg_img);


    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program applies Registration from a Moving Image on to a Fixed Image using the Elastix program.\n" \
        "- Elastix parameter files can be found in http://elastix.bigr.nl/wiki/index.php/Parameter_file_database.\n" \
        "- At least ONE Elastix Transformation File must be passed, using the option --t0.\n" \
        "- It is possible to use until 3 Multiple Transformation Files for different registrations.\n" \
        "- For this, we must pass the parameters using --tx, where x is the increment value, starting at 0 until 2,"\
        "representing the order of the execution from the parameter files.\n" \
        "- Examples:\n" \
        "(1) iftRegisterImageByElastix -m moving.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0042rigid.txt\n" \
        "(2) iftRegisterImageByElastix -m moving.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0000affine.txt --t1 Par0000bspline.txt\n" \
        "(3) iftRegisterImageByElastix -m moving.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0042rigid.txt --t1 Par0000affine.txt --t2 Par0000bspline.txt\n";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-m", .long_name = "--moving-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="(Moving) Image to be register on to the Fixed Image"},
        {.short_name = "-f", .long_name = "--fixed-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Reference (Fixed) Image where the Moving Image will be registered"},
        {.short_name = "-o", .long_name = "--output-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Register Image (The resulting Deformation Fields will also have the same basename)."},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Elastix Transformation File for Initial Transform.\n"\
                               "For multiple parameter files (up to 3), use the sequence of options: --t0, --t1, and --t2"},
        {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
        {.short_name = "", .long_name = "--t2", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
        {.short_name = "-v", .long_name = "--moving-label-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Mask image of the Moving Image. It only performs registration on " \
                                "the part of the image that is within the mask."},
        {.short_name = "-x", .long_name = "--fixed-label-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Mask image of the Fixed Image. It only performs registration on " \
                                "the part of the image that is within the mask."},
        {.short_name = "-z", .long_name = "--apply-minmax-normalization-without-outliers", .has_arg=false,
         .required=false, .help="Apply a Min-Max Normalization without outliers (the brightest 2\\% voxels) " \
                                "in the registered image. The normalization range is [0, max_range(fixed image)]"},
        {.short_name = "", .long_name = "--no-save-reg-img", .has_arg=false,
         .required=false, .help="Flag to Not Save the Resulting Registered Image."},
        {.short_name = "", .long_name = "--no-save-def-fields", .has_arg=false,
         .required=false, .help="Flag to Not Save the Deformation Fields."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **moving_img_path, char **fixed_img_path, char **out_reg_img_path,
                        iftFileSet **transf_files) {
    *moving_img_path  = iftGetStrValFromDict("--moving-image", args);
    *fixed_img_path   = iftGetStrValFromDict("--fixed-image", args);
    *out_reg_img_path = iftGetStrValFromDict("--output-image", args);

    // output basename dir
    char *parent_dir = iftParentDir(*out_reg_img_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    free(parent_dir);

    // only incremental values for --tx, starting at 0 until 2, are considered (--t0, --t1, --t2)
    iftSList *SL = iftCreateSList();
    int i = 0;
    char opt[16];
    sprintf(opt, "--t%d", i);
    while (iftDictContainKey(opt, args, NULL)) {
        iftInsertSListIntoTail(SL, iftGetStrValFromDict(opt, args));
        i++;
        sprintf(opt, "--t%d", i);
    }

    *transf_files = iftCreateFileSet(SL->n);
    for (size_t i = 0; i < (*transf_files)->n; i++) {
        char *path = iftRemoveSListHead(SL);
        (*transf_files)->files[i] = iftCreateFile(path);
        iftFree(path);
    }
    iftDestroySList(&SL);

    puts("-----------------------");
    printf("- Moving Image: \"%s\"\n", *moving_img_path);
    printf("- Fixed Image: \"%s\"\n", *fixed_img_path);
    printf("- Output Register Image: \"%s\"\n", *out_reg_img_path);
    printf("- Parameter Files:\n");
    for (size_t i = 0; i < (*transf_files)->n; i++)
        printf("[%lu] %s\n", i, (*transf_files)->files[i]->path);
    puts("-----------------------");
}



void iftGetOptionalArgs(  iftDict *args, char **moving_mask_path, char **fixed_mask_path) {
    if (iftDictContainKey("--moving-label-mask", args, NULL)) {
        *moving_mask_path = iftGetStrValFromDict("--moving-label-mask", args);
        printf("- Moving Mask Image: \"%s\"\n", *moving_mask_path);
    }
    else *moving_mask_path = NULL;
    
    if (iftDictContainKey("--fixed-label-mask", args, NULL)) {
        *fixed_mask_path = iftGetStrValFromDict("--fixed-label-mask", args);
        printf("- Fixed Mask Image: \"%s\"\n", *fixed_mask_path);
    }
    else *fixed_mask_path = NULL;

    puts("-----------------------\n");
}
/*************************************************************/
