#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **mask_path, int *tgt_obj, int *src_obj,
                        char **out_path, iftFileSet **elastix_params);
void iftGetOptionalArgs(  iftDict *args, char **normal_asym_map_path);

void iftRegisterSymmObjects(  iftImage *img,   iftImage *mask, int src_obj, int tgt_obj,
                              iftFileSet *elastix_params, iftImage **src_img_out, iftImage **tgt_img_flip_reg_out);
iftImage *iftNativeAsymMap(  iftImage *img,   iftImage *mask, int src_obj, int tgt_obj,
                             iftFileSet *elastix_params,   iftImage *normal_asym_map);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *img_path = NULL;
    char *mask_path = NULL;
    int tgt_obj;
    int src_obj;
    char *out_path = NULL;
    iftFileSet *elastix_params = NULL;
    // optional args
    char *normal_asym_map_path = NULL;
    
    iftGetRequiredArgs(args, &img_path, &mask_path, &tgt_obj, &src_obj, &out_path, &elastix_params);
    iftGetOptionalArgs(args, &normal_asym_map_path);
    
    timer *t1 = iftTic();
    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *mask = iftReadImageByExt(mask_path);
    iftImage *normal_asym_map = (normal_asym_map_path) ? iftReadImageByExt(normal_asym_map_path) : NULL;
    
    iftImage *asym_map = iftNativeAsymMap(img, mask, src_obj, tgt_obj, elastix_params, normal_asym_map);
    iftWriteImageByExt(asym_map, out_path);
    
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    
    
    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&mask);
    iftDestroyImage(&normal_asym_map);
    iftDestroyImage(&asym_map);
    
    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes the asymmetry map of an Image by registering a pair of rough symmetric objects (target objects).\n" \
        "- A normal asymmetry map (on NATIVE test space) can also passed to attenuate the resulting asymmetry map.";
    
    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Image (already pre-processed)."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Mask with the fixed and moving objects."},
        {.short_name = "-l", .long_name = "--pairs-of-symmetric-objects", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Define the a pair of labels for registration: The first is registered on to the second.\n" \
                            "e.g. 4:3 (object 4 is registered on to 3)."},
        {.short_name = "-o", .long_name = "--out-asym-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Native Asymmetry Map."},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Elastix Transformation File for Initial Transform.\n"\
                           "For multiple parameter files (up to 2), use the sequence of options: --t0, --t1"},
        {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
        {.short_name = "-s", .long_name = "--normal-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the normal asymmetry map (on NATIVE test space) used to attenuate " \
                            "brain asymmetries."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);
    
    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **mask_path, int *tgt_obj, int *src_obj,
                        char **out_path, iftFileSet **elastix_params) {
    *img_path = iftGetStrValFromDict("--image-path", args);
    *mask_path = iftGetStrValFromDict("--mask", args);
    *out_path = iftGetStrValFromDict("--out-asym-map", args);
    
    const char *symm_objs_entry = iftGetConstStrValFromDict("--pairs-of-symmetric-objects", args);
    
    if (iftRegexMatch(symm_objs_entry, "^[0-9]+:[0-9]+$")) {
        *tgt_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 0));
        *src_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 1));
        
        if (*src_obj == *tgt_obj)
            iftError("First Object label is equal to the second Object Label: %d",
                     "iftGetRequiredArgs", *src_obj);
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
    printf("- Image: %s\n", *img_path);
    printf("- Brain Mask: %s\n", *mask_path);
    printf("- Target (Moving) Object: %d\n", *tgt_obj);
    printf("- Source (Fixed) Object: %d\n", *src_obj);
    printf("- Output Asym. Map: %s\n", *out_path);
    puts("-------------------------");
    printf("- Elastix Params:\n");
    for (long i = 0; i < (*elastix_params)->n; i++)
        printf("[%lu] %s\n", i, (*elastix_params)->files[i]->path);
}


void iftGetOptionalArgs(  iftDict *args, char **normal_asym_map_path) {
    if (iftDictContainKey("--normal-asymmetry-map", args, NULL)) {
        *normal_asym_map_path = iftGetStrValFromDict("--normal-asymmetry-map", args);
        printf("- Normal Asymmetry Map (on NATIVE): %s\n", *normal_asym_map_path);
    }
    else *normal_asym_map_path = NULL;
    
    puts("-------------------------\n");
}


void iftRegisterSymmObjects(  iftImage *img,   iftImage *mask, int src_obj, int tgt_obj,
                              iftFileSet *elastix_params, iftImage **src_img_out, iftImage **tgt_img_flip_reg_out) {
    iftImage *src_img = iftObjectMask(img, mask, src_obj);
    iftImage *src_mask = iftExtractObject(mask, src_obj);
    iftImage *tgt_img = iftObjectMask(img, mask, tgt_obj);
    iftImage *tgt_img_flip = iftFlipImage(tgt_img, IFT_AXIS_X);
    
    // register symmetric objetcs
    iftImage *tgt_img_flip_reg = iftRegisterImageByElastix(tgt_img_flip, src_img, NULL, NULL, elastix_params, NULL, NULL);
    iftImage *tgt_img_flip_reg_mask = iftObjectMask(tgt_img_flip_reg, mask, src_obj);
    iftDestroyImage(&tgt_img_flip_reg);
    
    iftImage *tgt_img_flip_reg_hist = iftBrainMRIPreProcessing(tgt_img_flip_reg_mask, 12, NULL, src_img, NULL, true,
                                                               false, true, false, NULL);
    tgt_img_flip_reg = iftObjectMask(tgt_img_flip_reg_hist, mask, src_obj);
    
    if (src_img_out)
        *src_img_out = src_img;
    else iftDestroyImage(&src_img);
    
    if (tgt_img_flip_reg_out)
        *tgt_img_flip_reg_out = tgt_img_flip_reg;
    else iftDestroyImage(&tgt_img_flip_reg);
    
    iftDestroyImage(&src_mask);
    iftDestroyImage(&tgt_img);
    iftDestroyImage(&tgt_img_flip);
    iftDestroyImage(&tgt_img_flip_reg_mask);
    iftDestroyImage(&tgt_img_flip_reg_hist);
}


iftImage *iftNativeAsymMap(  iftImage *img,   iftImage *mask, int src_obj, int tgt_obj,
                             iftFileSet *elastix_params,   iftImage *normal_asym_map) {
    iftImage *src_img = NULL;
    iftImage *tgt_img_flip_reg = NULL;
    
    iftRegisterSymmObjects(img, mask, src_obj, tgt_obj, elastix_params, &src_img, &tgt_img_flip_reg);
    
    iftImage *asym_map = NULL;
    if (normal_asym_map) {
        iftVerifyImages(src_img, normal_asym_map, "iftNativeBrainAsymMap");
    
        asym_map = iftCreateImageFromImage(src_img);

        #pragma omp parallel for
        for (int p = 0; p < asym_map->n; p++)
            asym_map->val[p] = iftMax(0, abs(src_img->val[p] - tgt_img_flip_reg->val[p]) - normal_asym_map->val[p]);
    }
    else asym_map = iftAbsSub(src_img, tgt_img_flip_reg);
    
    iftDestroyImage(&src_img);
    iftDestroyImage(&tgt_img_flip_reg);
    
    return asym_map;
}
/*************************************************************/


