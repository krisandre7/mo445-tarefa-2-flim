#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputArgs(  iftDict *args, char **test_img_path, iftFileSet **train_set, char **mask_path,
                        int *mov_obj, int *fix_obj, char **out_dir, iftFileSet **df_native, iftFileSet **df_symm_objs,
                        bool *use_stdev);
void iftGetMasks(  iftImage *mask, int mov_obj, int fix_obj, iftImage **fix_mask, iftImage **mov_flip_mask);
void iftMapSymmObjects(  iftImage *img,   iftImage *fix_mask,   iftImage *mov_flip_mask,
                         iftFileSet *df, iftImage **reg_mov_img, iftImage **fix_img, const char *filename);
iftFileSet *iftExtractNativeTrainAsymMaps(  iftFileSet *train_set,   iftFileSet *df_native,
                                            iftImage *test_img,   iftImage *mask,
                                            iftImage *fix_mask,   iftImage *mov_flip_mask,
                                            iftFileSet *df_symm_objs, const char *out_dir, bool add_stdev_asymmetries,
                                          iftImage **normal_asym_map_out, iftFileSet **train_set_native_out);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *test_img_path = NULL;
    iftFileSet *train_set = NULL;
    char *mask_path = NULL;
    int mov_obj;
    int fix_obj;
    char *out_dir = NULL;
    iftFileSet *df_native = NULL;
    iftFileSet *df_symm_objs = NULL;
    bool use_stdev = false;
    
    iftGetInputArgs(args, &test_img_path, &train_set, &mask_path, &mov_obj, &fix_obj, &out_dir, &df_native,
                    &df_symm_objs, &use_stdev);

    timer *t1 = iftTic();
    
    iftImage *test_img = iftReadImageByExt(test_img_path);
    iftImage *mask = iftReadImageByExt(mask_path);
    iftImage *fix_mask = NULL;
    iftImage *mov_flip_mask = NULL;
    iftGetMasks(mask, mov_obj, fix_obj, &fix_mask, &mov_flip_mask);
    
    
    iftImage *normal_asym_map = NULL;
    iftFileSet *train_set_native = NULL;
    iftFileSet *train_asym_maps = iftExtractNativeTrainAsymMaps(train_set, df_native, test_img, mask,
                                                                fix_mask, mov_flip_mask, df_symm_objs, out_dir,
                                                                use_stdev, &normal_asym_map, &train_set_native);
    iftWriteImageByExt(normal_asym_map, iftJoinPathnames(2, out_dir, "normal_asym_map.nii.gz"));
    
    

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&train_set);
    iftDestroyFileSet(&df_native);
    iftDestroyFileSet(&df_symm_objs);
    iftDestroyImage(&test_img);
    iftDestroyImage(&mask);
    iftDestroyImage(&fix_mask);
    iftDestroyImage(&mov_flip_mask);
    iftDestroyImage(&normal_asym_map);
    iftDestroyFileSet(&train_set_native);
    iftDestroyFileSet(&train_asym_maps);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Map training images (previously registered on to a same template) on to the NATIVE space, and then " \
        "compute their asymmetry maps and the normal asymmetry map.\n" \
        "- It saves in the output directory: training images on native space, train asymmetry maps, and the normal " \
        "normal asymmetry map.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pre-processed test image."},
        {.short_name = "-i", .long_name = "--train-set-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir. or CSV with the pathname from the Train. Images (previously registered on to the template)."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname of the Mask."},
        {.short_name = "-l", .long_name = "--pairs-of-symmetric-objects", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Define the a pair of labels for registration: The first is registered on to the second.\n" \
                                "e.g. 4:3 (object 4 is registered on to 3)."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output directory."},
        {.short_name = "", .long_name = "--d0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Deformation field to map the training images to NATIVE space.\n"\
                               "For multiple deformation fields (up to 3), use the sequence of options: --d0, --d1"},
        {.short_name = "", .long_name = "--d1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Deformation field to map the training image to NATIVE space."},
        {.short_name = "", .long_name = "--r0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Deformation field to map the Moving Object to the Fixed one.\n"\
                               "For multiple deformation fields (up to 3), use the sequence of options: --r0, --r1"},
        {.short_name = "", .long_name = "--r1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Deformation field to map the Moving Object to the Fixed one."},
        {.short_name = "", .long_name = "--use-stdev", .has_arg=false,
         .required=false, .help="Add/Use the Standard Deviation from the asymmetries into the output asymmetry map."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputArgs(  iftDict *args, char **test_img_path, iftFileSet **train_set, char **mask_path,
                        int *mov_obj, int *fix_obj, char **out_dir, iftFileSet **df_native, iftFileSet **df_symm_objs,
                        bool *use_stdev) {
    *test_img_path = iftGetStrValFromDict("--test-image", args);
    const char *train_img_entry = iftGetConstStrValFromDict("--train-set-entry", args);
    *train_set = iftLoadFileSetFromDirOrCSV(train_img_entry, 0, true);
    *mask_path = iftGetStrValFromDict("--mask", args);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    
    const char *symm_objs_entry = iftGetConstStrValFromDict("--pairs-of-symmetric-objects", args);
    
    if (iftRegexMatch(symm_objs_entry, "^[0-9]+:[0-9]+$")) {
        *mov_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 0));
        *fix_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 1));
        
        if (*fix_obj == *mov_obj)
            iftError("First Object label is equal to the second Object Label: %d",
                     "iftGetInputArgs", *fix_obj);
    } else
        iftError("Pair of Symmetric Objects \"%s\" does not match with the expected regular expression: [0-9]+:[0-9]+",
                 "iftGetInputArgs", symm_objs_entry);
    
    // only incremental values for --dx, starting at 0 until 2, are considered (--d0, --d1, --d2)
    iftSList *SL = iftCreateSList();
    int i = 0;
    char opt[16];
    sprintf(opt, "--d%d", i);
    while (iftDictContainKey(opt, args, NULL)) {
        iftInsertSListIntoTail(SL, iftGetStrValFromDict(opt, args));
        i++;
        sprintf(opt, "--d%d", i);
    }
    
    *df_native = iftCreateFileSet(SL->n);
    for (long i = 0; i < (*df_native)->n; i++) {
        char *path = iftRemoveSListHead(SL);
        (*df_native)->files[i] = iftCreateFile(path);
        iftFree(path);
    }
    iftDestroySList(&SL);
    
    // only incremental values for --rx, starting at 0 until 2, are considered (--r0, --r1, --r2)
    SL = iftCreateSList();
    i = 0;
    sprintf(opt, "--r%d", i);
    while (iftDictContainKey(opt, args, NULL)) {
        iftInsertSListIntoTail(SL, iftGetStrValFromDict(opt, args));
        i++;
        sprintf(opt, "--r%d", i);
    }
    
    *df_symm_objs = iftCreateFileSet(SL->n);
    for (long i = 0; i < (*df_symm_objs)->n; i++) {
        char *path = iftRemoveSListHead(SL);
        (*df_symm_objs)->files[i] = iftCreateFile(path);
        iftFree(path);
    }
    iftDestroySList(&SL);
    
    *use_stdev = iftDictContainKey("--use-stdev", args, NULL);
    
    puts("-------------------------");
    printf("- Test Image: %s\n", *test_img_path);
    printf("- Train Image Entry: %s\n", train_img_entry);
    printf("- Brain Mask: %s\n", *mask_path);
    printf("- Moving Object: %d\n", *mov_obj);
    printf("- Fixed Object: %d\n", *fix_obj);
    printf("- Output Directory: %s\n", *out_dir);
    puts("-------------------------");
    printf("- Def. Fields to map to NATIVE space:\n");
    for (long i = 0; i < (*df_native)->n; i++)
        printf("[%lu] %s\n", i, (*df_native)->files[i]->path);
    printf("- Def. Fields to map the Moving Object to the Fixed one:\n");
    for (long i = 0; i < (*df_symm_objs)->n; i++)
        printf("[%lu] %s\n", i, (*df_symm_objs)->files[i]->path);
    printf("- Use stdev: %s\n", iftBoolAsString(*use_stdev));
    puts("-------------------------\n");
}


void iftGetMasks(  iftImage *mask, int mov_obj, int fix_obj, iftImage **fix_mask, iftImage **mov_flip_mask) {
    iftImage *mask_flip = iftFlipImage(mask, IFT_AXIS_X);
    *fix_mask = iftExtractObject(mask, fix_obj);
    *mov_flip_mask = iftExtractObject(mask_flip, mov_obj);
    iftDestroyImage(&mask_flip);
}


void iftMapSymmObjects(  iftImage *img,   iftImage *fix_mask,   iftImage *mov_flip_mask,
                         iftFileSet *df, iftImage **reg_mov_img, iftImage **fix_img, const char *filename) {
    iftImage *img_flip = iftFlipImage(img, IFT_AXIS_X);
    *fix_img = iftMask(img, fix_mask);
    iftImage *mov_flip_img = iftMask(img_flip, mov_flip_mask);
    
    // register symmetric objetcs
    iftImage *mov_flip_img_reg = iftTransformImageByTransformix(mov_flip_img, df->files[df->n - 1]->path);
    iftImage *mov_flip_img_reg_mask = iftMask(mov_flip_img_reg, fix_mask);
    iftImage *mov_flip_img_reg_mask_hist = iftBrainMRIPreProcessing(mov_flip_img_reg_mask, 12, NULL, *fix_img,
                                                                    NULL, true, false, true, false, NULL);
    *reg_mov_img = iftMask(mov_flip_img_reg_mask_hist, fix_mask);
    
    iftDestroyImage(&img_flip);
    iftDestroyImage(&mov_flip_img);
    iftDestroyImage(&mov_flip_img_reg);
    iftDestroyImage(&mov_flip_img_reg_mask);
    iftDestroyImage(&mov_flip_img_reg_mask_hist);
}


iftFileSet *iftExtractNativeTrainAsymMaps(  iftFileSet *train_set,   iftFileSet *df_native,
                                            iftImage *test_img,   iftImage *mask,
                                            iftImage *fix_mask,   iftImage *mov_flip_mask,
                                            iftFileSet *df_symm_objs, const char *out_dir, bool add_stdev_asymmetries,
                                          iftImage **normal_asym_map_out, iftFileSet **train_set_native_out) {
    iftFileSet *train_asymmap_set = iftCreateFileSet(train_set->n);
    iftFileSet *train_set_native = iftCreateFileSet(train_set->n);
    
    iftImage *test_img_masked = iftMask(test_img, mask);
    iftFImage *mean_asym = iftCreateFImage(mask->xsize, mask->ysize, mask->zsize);
    iftCopyVoxelSize(test_img, mean_asym);
    
    for (long i = 0; i < train_set->n; i++) {
        char *filename = iftFilename(train_set->files[i]->path, NULL);
        puts(filename);
        
        iftImage *aux = NULL;
        iftImage *timg = iftReadImageByExt(train_set->files[i]->path);
        iftImage *timg_native = aux = iftTransformImageByTransformix(timg, df_native->files[df_native->n - 1]->path);
        iftDestroyImage(&timg);
    
        timg_native = iftMask(timg_native, mask);
        iftDestroyImage(&aux);
        aux = timg_native;
        
        timg_native = iftBrainMRIPreProcessing(timg_native, 12, NULL, test_img_masked, NULL, true, false, true, false, NULL);
        iftDestroyImage(&aux);
        aux = timg_native;
    
        timg_native = iftMask(timg_native, mask);
        iftDestroyImage(&aux);
        
        char *timg_native_path = iftJoinPathnames(3, out_dir, "train_set_on_native", filename);
        train_set_native->files[i] = iftCreateFile(timg_native_path);
        iftWriteImageByExt(timg_native, train_set_native->files[i]->path);
        
        iftImage *reg_mov_img = NULL;
        iftImage *fix_img = NULL;
        iftMapSymmObjects(timg_native, fix_mask, mov_flip_mask, df_symm_objs, &reg_mov_img, &fix_img, iftFilename(train_set->files[i]->path, NULL));
        
        iftImage *asym_map = iftAbsSub(reg_mov_img, fix_img);
        char *asym_map_path = iftJoinPathnames(3, out_dir, "train_asym_maps", filename);
        train_asymmap_set->files[i] = iftCreateFile(asym_map_path);
        iftWriteImageByExt(asym_map, train_asymmap_set->files[i]->path);

        #pragma omp parallel for
        for (int p = 0; p < asym_map->n; p++)
            mean_asym->val[p] += (asym_map->val[p] / ((float) train_set->n));
        
        iftFree(filename);
        iftFree(timg_native_path);
        iftDestroyImage(&timg_native);
        iftDestroyImage(&reg_mov_img);
        iftDestroyImage(&fix_img);
        iftDestroyImage(&asym_map);
        iftDestroyImage(&test_img_masked);
        iftFree(asym_map_path);
    }
    
    if (add_stdev_asymmetries) {
        iftFImage *stdev_asym = iftCreateFImage(mean_asym->xsize, mean_asym->ysize, mean_asym->zsize);
        
        for (int i = 0; i < train_asymmap_set->n; i++) {
            iftImage *asymmap_native = iftReadImageByExt(train_asymmap_set->files[i]->path);

            #pragma omp parallel for
            for (int p = 0; p < asymmap_native->n; p++)
                stdev_asym->val[p] += powf(asymmap_native->val[p] - mean_asym->val[p], 2);
            
            iftDestroyImage(&asymmap_native);
        }
        
        // adding stdev asymmetries to the mean asymmetries
        #pragma omp parallel for
        for (int p = 0; p < stdev_asym->n; p++)
            mean_asym->val[p] = mean_asym->val[p] + (sqrtf(stdev_asym->val[p] / train_asymmap_set->n));
        
        iftDestroyFImage(&stdev_asym);
    }
    
    *normal_asym_map_out = iftRoundFImage(mean_asym);
    *train_set_native_out = train_set_native;
    iftDestroyFImage(&mean_asym);
    
    return train_asymmap_set;
}
/*************************************************************/


