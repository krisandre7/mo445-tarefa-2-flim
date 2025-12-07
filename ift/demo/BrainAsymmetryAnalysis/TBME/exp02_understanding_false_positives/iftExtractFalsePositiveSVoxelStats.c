#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
/*************************************************************/


// flip left hemisphere and sum it to the right one to consider only one side of the brain
// for the correct statistics
// IT ASSUMES THAT THE LESIONS ARE INDEED ASYMMETRIC
iftImage *iftPrepareGT(const char *gt_path) {
    iftImage *gt = iftReadImageByExt(gt_path);
    iftBoundingBox bb = {.begin = {gt->xsize / 2, 0, 0}, .end = {gt->xsize - 1, gt->ysize - 1, gt->zsize - 1}};
    
    iftImage *gt_flip = iftFlipImage(gt, IFT_AXIS_X);
    iftImage *final_gt = iftAdd(gt, gt_flip);
    iftFillBoundingBoxInImage(final_gt, bb, 0);
    
    iftDestroyImage(&gt);
    iftDestroyImage(&gt_flip);
    
    return final_gt;
}


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    timer *t1 = iftTic();
    
    const char *result_dir_path = iftGetConstStrValFromDict("--saad-result-dir", args);
    const char *gt_dir_path = iftGetConstStrValFromDict("--ground-truth-dir", args);
    const char *img_dir_path = iftGetConstStrValFromDict("--image-dir", args);
    const char *right_hem_mask_path = iftGetConstStrValFromDict("--right-brain-mask", args);
    const char *out_path = iftGetConstStrValFromDict("--output-json", args);
    const char *normal_asym_map_path = (iftDictContainKey("--normal-asymmetry-map", args, NULL)) ? iftGetConstStrValFromDict("--normal-asymmetry-map", args) : NULL;
    puts("--------------------------");
    printf("- Resulting Dir: %s\n", result_dir_path);
    printf("- GT Dir: %s\n", gt_dir_path);
    printf("- Right Hemisphere Brain Mask: %s\n", right_hem_mask_path);
    printf("- Output JSON: %s\n", out_path);
    if (normal_asym_map_path)
        printf("- Normal Asymmetry Map: %s", normal_asym_map_path);
    puts("--------------------------\n");
    

    iftDList *precision_list = iftCreateDList();
    iftDList *svoxels_vols_list = iftCreateDList();
    iftDList *edts_list = iftCreateDList();
    iftDList *mean_asym_list = iftCreateDList();
    
    
    iftDir *result_dir = iftLoadDir(result_dir_path, 1);
    iftImage *right_hem_mask = iftReadImageByExt(right_hem_mask_path);
    iftImage *edt_img = iftEuclDistTrans(right_hem_mask, NULL, IFT_INTERIOR, NULL, NULL, NULL);
    iftImage *normal_asym_map = (normal_asym_map_path) ? iftReadImageByExt(normal_asym_map_path) : NULL;
    
    for (int i = 0; i < result_dir->nsubdirs; i++) {
//    for (int i = 0; i < 1; i++) {
        char *img_id = iftFilename(result_dir->subdirs[i]->path, NULL);
        char *img_path = iftJoinPathnames(2, img_dir_path, iftCopyString("%s.nii.gz", img_id));
        char *gt_path = iftJoinPathnames(2, gt_dir_path, iftCopyString("%s.nii.gz", img_id));
        char *result_path = iftJoinPathnames(2, result_dir->subdirs[i]->path, "result.nii.gz");

        printf("\t[%d]\n", i);
        printf("\t- Image: %s\n", img_path);
        printf("\t- GT: %s\n", gt_path);
        printf("\t- Result: %s\n\n", result_path);

        iftImage *img = iftReadImageByExt(img_path);
        iftImage *asym_map = iftBrainAsymMap(img, normal_asym_map);
        
        // merge both hemisphere in gt to provide stats with only one svoxels (not a pair of symmetric ones)
        iftImage *gt_img = iftPrepareGT(gt_path);
        iftBoundingBox bb = {.begin = {.x = gt_img->xsize / 2, .y = 0, .z = 0},
                .end   = {.x = gt_img->xsize - 1, .y = gt_img->ysize - 1, .z = gt_img->zsize - 1}};

        iftImage *result_img = iftReadImageByExt(result_path);
        iftFillBoundingBoxInImage(result_img, bb, 0);

        //        iftWriteImageByExt(gt_img, "out/%s_gt.nii.gz", img_id);
        //        iftWriteImageByExt(svoxels_img, "out/%s_svoxels.nii.gz", img_id);
        //        iftWriteImageByExt(result_img, "out/%s_result.nii.gz", img_id);


        iftIntArray *target_svoxels = iftIntArrayUnique(result_img->val, result_img->n);

        // ignore zero-label
        for (int o = 1; o < target_svoxels->n; o++) {
            iftImage *target_svoxel_img = iftExtractObject(result_img, target_svoxels->val[o]);
            printf("\t\t- Svoxel: %d\n", target_svoxels->val[o]);
    
            float tp = 0, fn = 0, fp = 0, tn = 0;
    
            for (int p = 0; p < target_svoxel_img->n; p++) {
                if (gt_img->val[p]) {
                    if (target_svoxel_img->val[p]) tp++;
                    else fn++;
                } else {
                    if (target_svoxel_img->val[p]) fp++;
                    else tn++;
                }
            }
    
    
            float precision = tp / (tp + fp);
            float svoxel_vol = tp + fp;
            printf("\t\t\t- precision: %f\n", precision);
    
            // false positive voxel
            if (precision <= 0.1) {
                printf("\t\t\t### False Positive Voxel\n");
                iftInsertDListIntoTail(precision_list, precision);
                iftInsertDListIntoTail(svoxels_vols_list, svoxel_vol);
        
                iftVoxel gc = iftGeometricCenterVoxel(target_svoxel_img);
                float edt = sqrtf(iftImgVoxelVal(edt_img, gc));
                iftInsertDListIntoTail(edts_list, edt);
                printf("\t\t\t- edt: %f\n", edt);
        
                float mean_asym = iftMeanValue(asym_map, target_svoxel_img);
                iftInsertDListIntoTail(mean_asym_list, mean_asym);
                printf("\t\t\t- mean_asym: %f\n", mean_asym);
            }
    
            iftDestroyImage(&target_svoxel_img);
        }


        iftFree(img_id);
        iftFree(gt_path);
        iftFree(result_path);
        iftDestroyImage(&img);
        iftDestroyImage(&asym_map);
        iftDestroyImage(&gt_img);
        iftDestroyImage(&result_img);
        iftDestroyIntArray(&target_svoxels);
    }
    
    
    puts("\n- Saving Stats");
    iftDict *stats = iftCreateDict();
    iftInsertIntoDict("precision", iftDListToDblArray(precision_list), stats);
    iftInsertIntoDict("svoxels-volume", iftDListToDblArray(svoxels_vols_list), stats);
    iftInsertIntoDict("edt", iftDListToDblArray(edts_list), stats);
    iftInsertIntoDict("mean-asym", iftDListToDblArray(mean_asym_list), stats);
    iftWriteJson(stats, out_path);
    

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Compute some statistics for Supervoxels and Ground-Truth.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-r", .long_name = "--saad-result-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory with the result for SAAD in the first experiments has the image id: (eg: exps/01_why_good_supervoxels/SAAD_ISBI/split_01)."},
        {.short_name = "-i", .long_name = "--image-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory with the pre-processed testing images. Their filenames must match with the image " \
                                "ids from the result dir."},
        {.short_name = "-g", .long_name = "--ground-truth-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Directory with the ground-truth segmentations. Their filenames must match with the image " \
                                "ids from the result dir."},
        {.short_name = "-m", .long_name = "--right-brain-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Right Brain Hemisphere Mask."},
        {.short_name = "-o", .long_name = "--output-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output JSON file with the stats."},
        {.short_name = "-s", .long_name = "--normal-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the normal asymmetry map used to attenuate the brain asymmetries " \
                                "for the seed initialization for SymmISF."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}
/*************************************************************/


