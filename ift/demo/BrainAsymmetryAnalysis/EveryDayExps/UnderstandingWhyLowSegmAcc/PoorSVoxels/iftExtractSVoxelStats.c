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
    const char *brain_envolpe_path = iftGetConstStrValFromDict("--brain-envelope-mask", args);
    const char *out_path = iftGetConstStrValFromDict("--output-json", args);
    puts("--------------------------");
    printf("- Resulting Dir: %s\n", result_dir_path);
    printf("- GT Dir: %s\n", gt_dir_path);
    printf("- Brain Envelope: %s\n", brain_envolpe_path);
    printf("- Output JSON: %s\n", out_path);
    puts("--------------------------\n");
    

    iftDList *tpr_list[2] = {iftCreateDList(), iftCreateDList()};
    iftDList *precision_list[2] = {iftCreateDList(), iftCreateDList()};
    iftDList *fdr_list[2] = {iftCreateDList(), iftCreateDList()};
    iftDList *svoxels_vols_list[2] = {iftCreateDList(), iftCreateDList()};
    iftDList *edts_list[2] = {iftCreateDList(), iftCreateDList()};
    
    
    iftDir *result_dir = iftLoadDir(result_dir_path, 1);
    iftImage *brain_envelope = iftReadImageByExt(brain_envolpe_path);
    iftImage *edt_img = iftEuclDistTrans(brain_envelope, NULL, IFT_INTERIOR, NULL, NULL, NULL);
    
    for (int i = 0; i < result_dir->nsubdirs; i++) {
//    for (int i = 0; i < 1; i++) {
        char *img_id = iftFilename(result_dir->subdirs[i]->path, NULL);
        char *gt_path = iftJoinPathnames(2, gt_dir_path, iftCopyString("%s.nii.gz", img_id));
        char *svoxels_path = iftJoinPathnames(3, result_dir_path, img_id, "svoxels.nii.gz");
        char *result_path = iftJoinPathnames(3, result_dir_path, img_id, "result.nii.gz");
        
        printf("\t[%d]\n", i);
        printf("\t- GT: %s\n", gt_path);
        printf("\t- SVoxels: %s\n", svoxels_path);
        printf("\t- Result: %s\n\n", result_path);
        
        // merge both hemisphere in gt to provide stats with only one svoxels (not a pair of symmetric ones)
        iftImage *gt_img = iftPrepareGT(gt_path);
        iftBoundingBox bb = {.begin = {.x = gt_img->xsize / 2, .y = 0, .z = 0},
                             .end   = {.x = gt_img->xsize - 1, .y = gt_img->ysize - 1, .z = gt_img->zsize - 1}};
        
        iftImage *svoxels_img = iftReadImageByExt(svoxels_path);
        iftFillBoundingBoxInImage(svoxels_img, bb, 0);
        
        iftImage *result_img = iftReadImageByExt(result_path);
        iftFillBoundingBoxInImage(result_img, bb, 0);
        
//        iftWriteImageByExt(gt_img, "out/%s_gt.nii.gz", img_id);
//        iftWriteImageByExt(svoxels_img, "out/%s_svoxels.nii.gz", img_id);
//        iftWriteImageByExt(result_img, "out/%s_result.nii.gz", img_id);
        

        iftImage *svoxels_intersec_gt = iftMask(svoxels_img, gt_img);
        iftIntArray *target_svoxels = iftIntArrayUnique(svoxels_intersec_gt->val, svoxels_intersec_gt->n);
        
        // ignore zero-label
        for (int o = 1; o < target_svoxels->n; o++) {
            bool was_svoxels_detected = false;
    
            iftImage *target_svoxel_img = iftExtractObject(svoxels_img, target_svoxels->val[o]);
            printf("\t\t- Svoxel: %d\n", target_svoxels->val[o]);
            
            float tp = 0, fn = 0, fp = 0, tn = 0;
            
            for (int p = 0; p < target_svoxel_img->n; p++) {
                if (gt_img->val[p]) {
                    if (target_svoxel_img->val[p]) tp++;
                    else fn++;
                }
                else {
                    if (target_svoxel_img->val[p]) fp++;
                    else tn++;
                }
    
                if (result_img->val[p] == target_svoxels->val[o])
                    was_svoxels_detected = true;
            }
            printf("\t\t\t- was detected: %s\n", iftBoolAsString(was_svoxels_detected));
            
    
            float tpr = tp / (tp + fn);  // true positive rate
            float precision = tp / (tp + fp);
            float fdr = fp / (fp + tp);  // false discovery rate
            float svoxel_vol = tp + fp;
            
            if (was_svoxels_detected && (svoxel_vol <= 1000) && (precision <= 0.15))
                printf("\t\t\t- ANALYZE: %s\n", iftBoolAsString(was_svoxels_detected));
    
            
            int idx = (was_svoxels_detected) ? 1 : 0;
            iftInsertDListIntoTail(tpr_list[idx], tpr);
            iftInsertDListIntoTail(precision_list[idx], precision);
            iftInsertDListIntoTail(fdr_list[idx], fdr);
            iftInsertDListIntoTail(svoxels_vols_list[idx], svoxel_vol);
    
            iftVoxel gc = iftGeometricCenterVoxel(target_svoxel_img);
            float edt = iftImgVoxelVal(edt_img, gc);
            iftInsertDListIntoTail(edts_list[idx], edt);
    
            iftDestroyImage(&target_svoxel_img);
        }
        
        
        iftFree(img_id);
        iftFree(gt_path);
        iftFree(svoxels_path);
        iftFree(result_path);
        iftDestroyImage(&gt_img);
        iftDestroyImage(&svoxels_img);
        iftDestroyImage(&result_img);
        iftDestroyImage(&svoxels_intersec_gt);
        iftDestroyIntArray(&target_svoxels);
    }
    
    
    puts("\n- Saving Stats");
    iftDict *stats_missed = iftCreateDict();
    iftInsertIntoDict("tpr", iftDListToDblArray(tpr_list[0]), stats_missed);
    iftInsertIntoDict("precision", iftDListToDblArray(precision_list[0]), stats_missed);
    iftInsertIntoDict("fdr", iftDListToDblArray(fdr_list[0]), stats_missed);
    iftInsertIntoDict("svoxels-volume", iftDListToDblArray(svoxels_vols_list[0]), stats_missed);
    iftInsertIntoDict("edt", iftDListToDblArray(edts_list[0]), stats_missed);
    
    iftDict *stats_detected = iftCreateDict();
    iftInsertIntoDict("tpr", iftDListToDblArray(tpr_list[1]), stats_detected);
    iftInsertIntoDict("precision", iftDListToDblArray(precision_list[1]), stats_detected);
    iftInsertIntoDict("fdr", iftDListToDblArray(fdr_list[1]), stats_detected);
    iftInsertIntoDict("svoxels-volume", iftDListToDblArray(svoxels_vols_list[1]), stats_detected);
    iftInsertIntoDict("edt", iftDListToDblArray(edts_list[1]), stats_detected);
    
    
    iftDict *stats = iftCreateDict();
    iftInsertIntoDict("missed", stats_missed, stats);
    iftInsertIntoDict("detected", stats_detected, stats);
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
         .required=true, .help="Directory with the result for SAAD. Each subfolder has the image id: (eg: result/000001_000001)."},
        {.short_name = "-g", .long_name = "--ground-truth-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory with the ground-truth segmentations. Their filenames must match with the image " \
                                "ids from the result dir."},
        {.short_name = "-m", .long_name = "--brain-envelope-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Brain Envelope Mask."},
        {.short_name = "-o", .long_name = "--output-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output JSON file with the stats."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}
/*************************************************************/


