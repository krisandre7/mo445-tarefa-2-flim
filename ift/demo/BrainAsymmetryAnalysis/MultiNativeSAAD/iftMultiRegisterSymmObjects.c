#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **mask_path, int *mov_obj, int *fix_obj,
                        char **out_dir, iftFileSet **elastix_params);
void iftGetMasks(  iftImage *mask, int mov_obj, int fix_obj, iftImage **fix_mask, iftImage **mov_flip_mask);
iftFileSet *iftRegisterSymmObjects(  iftImage *img,   iftImage *fix_mask,   iftImage *mov_flip_mask,
                                     iftFileSet *elastix_params, const char *out_dir, iftImage **reg_mov_img,
                                   iftImage **fix_img, iftFileSet **inverse_df_symm_objs);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *img_path = NULL;
    char *mask_path = NULL;
    int mov_obj;
    int fix_obj;
    char *out_dir = NULL;
    iftFileSet *elastix_params = NULL;
    
    iftGetRequiredArgs(args, &img_path, &mask_path, &mov_obj, &fix_obj, &out_dir, &elastix_params);
    

    timer *t1 = iftTic();

    iftImage *img = iftReadImageByExt(img_path);
    iftImage *mask = iftReadImageByExt(mask_path);
    
    iftImage *fix_mask = NULL;
    iftImage *mov_flip_mask = NULL;
    iftGetMasks(mask, mov_obj, fix_obj, &fix_mask, &mov_flip_mask);
    
    iftImage *mov_flip_img_reg = NULL;
    iftImage *fix_img = NULL;
    iftFileSet *inverse_df_symm_objs = NULL;
    iftFileSet *df_symm_objs = iftRegisterSymmObjects(img, fix_mask, mov_flip_mask, elastix_params, out_dir,
                                                      &mov_flip_img_reg, &fix_img, &inverse_df_symm_objs);
    
    iftWriteImageByExt(mov_flip_img_reg, iftJoinPathnames(2, out_dir, "moving_flip_registered_image.nii.gz"));
    iftWriteImageByExt(fix_img, iftJoinPathnames(2, out_dir, "fixed_image.nii.gz"));
    iftWriteImageByExt(fix_mask, iftJoinPathnames(2, out_dir, "fixed_mask.nii.gz"));
    
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&mask);
    iftDestroyImage(&fix_mask);
    iftDestroyImage(&mov_flip_mask);
    iftDestroyImage(&mov_flip_img_reg);
    iftDestroyImage(&fix_img);
    iftDestroyFileSet(&inverse_df_symm_objs);
    iftDestroyFileSet(&df_symm_objs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Register a moving object to a fixed in a test image based on its mask.\n" \
        "- It saves in the output directory: fixed image object, registered flipped moving image object, " \
        "deformation fields and their inverse.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Test Image (already pre-processed)."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Mask with the fixed and moving objects."},
        {.short_name = "-l", .long_name = "--pairs-of-symmetric-objects", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Define the a pair of labels for registration: The first is registered on to the second.\n" \
                                "e.g. 4:3 (object 4 is registered on to 3)."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output directory."},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Elastix Transformation File for Initial Transform.\n"\
                               "For multiple parameter files (up to 2), use the sequence of options: --t0, --t1"},
        {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **mask_path, int *mov_obj, int *fix_obj,
                        char **out_dir, iftFileSet **elastix_params) {
    *img_path = iftGetStrValFromDict("--test-image", args);
    *mask_path = iftGetStrValFromDict("--mask", args);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    if (!iftDirExists(*out_dir))
        iftMakeDir(*out_dir);
    
    const char *symm_objs_entry = iftGetConstStrValFromDict("--pairs-of-symmetric-objects", args);
    
    if (iftRegexMatch(symm_objs_entry, "^[0-9]+:[0-9]+$")) {
        *mov_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 0));
        *fix_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 1));
        
        if (*fix_obj == *mov_obj)
            iftError("First Object label is equal to the second Object Label: %d",
                     "iftGetRequiredArgs", *fix_obj);
    } else
        iftError("Pair of Symmetric Objects \"%s\" does not match with the expected regular expression: [0-9]+:[0-9]+",
                 "iftGetRequiredArgs", symm_objs_entry);
    
    // only incremental values for --dx, starting at 0 until 2, are considered (--t0, --t1, --t2)
    iftSList *SL = iftCreateSList();
    int i = 0;
    char opt[16];
    sprintf(opt, "--t%d", i);
    while (iftDictContainKey(opt, args, NULL)) {
        iftInsertSListIntoTail(SL, iftGetStrValFromDict(opt, args));
        i++;
        sprintf(opt, "--t%d", i);
    }
    
    *elastix_params = iftCreateFileSet(SL->n);
    for (long i = 0; i < (*elastix_params)->n; i++) {
        char *path = iftRemoveSListHead(SL);
        (*elastix_params)->files[i] = iftCreateFile(path);
        iftFree(path);
    }
    iftDestroySList(&SL);
    
    puts("-------------------------");
    printf("- Test Image: %s\n", *img_path);
    printf("- Brain Mask: %s\n", *mask_path);
    printf("- Moving Object: %d\n", *mov_obj);
    printf("- Fixed Object: %d\n", *fix_obj);
    printf("- Output Directory: %s\n", *out_dir);
    puts("-------------------------");
    printf("- Elastix Params:\n");
    for (long i = 0; i < (*elastix_params)->n; i++)
        printf("[%lu] %s\n", i, (*elastix_params)->files[i]->path);
    puts("-------------------------\n");
}


void iftGetMasks(  iftImage *mask, int mov_obj, int fix_obj, iftImage **fix_mask, iftImage **mov_flip_mask) {
    iftImage *mask_flip = iftFlipImage(mask, IFT_AXIS_X);
    *fix_mask = iftExtractObject(mask, fix_obj);
    *mov_flip_mask = iftExtractObject(mask_flip, mov_obj);
    iftDestroyImage(&mask_flip);
}


iftFileSet *iftRegisterSymmObjects(  iftImage *img,   iftImage *fix_mask,   iftImage *mov_flip_mask,
                                     iftFileSet *elastix_params, const char *out_dir, iftImage **reg_mov_img,
                                   iftImage **fix_img, iftFileSet **inverse_df_symm_objs) {
    iftImage *img_flip = iftFlipImage(img, IFT_AXIS_X);
    *fix_img = iftMask(img, fix_mask);
    iftImage *mov_flip_img = iftMask(img_flip, mov_flip_mask);
    
    // register symmetric objetcs
    char *basename = iftJoinPathnames(2, out_dir, "SymmObjectDefFields");
    iftFileSet *df = NULL;
    iftImage *mov_flip_img_reg = iftRegisterImageByElastix(mov_flip_img, *fix_img, NULL, NULL, elastix_params, basename, &df);
    iftImage *mov_flip_img_reg_mask = iftMask(mov_flip_img_reg, fix_mask);
    iftImage *mov_flip_img_reg_mask_hist = iftBrainMRIPreProcessing(mov_flip_img_reg_mask, 12, NULL, *fix_img,
                                                                    NULL, true, false, true, false, NULL);
    *reg_mov_img = iftMask(mov_flip_img_reg_mask_hist, fix_mask);
    
    // find "inverse" registration
    char *basename_inv = iftJoinPathnames(2, out_dir, "InverseSymmObjectDefFields");
    iftImage *fix_img_reg = iftRegisterImageByElastix(*fix_img, mov_flip_img, NULL, NULL, elastix_params, basename_inv, inverse_df_symm_objs);
    
    iftDestroyImage(&img_flip);
    iftDestroyImage(&mov_flip_img);
    iftFree(basename);
    iftDestroyImage(&mov_flip_img_reg);
    iftDestroyImage(&mov_flip_img_reg_mask);
    iftDestroyImage(&mov_flip_img_reg_mask_hist);
    iftFree(basename_inv);
    iftDestroyImage(&fix_img_reg);
    
    return df;
}
/*************************************************************/


