#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputArgs(  iftDict *args, char **img_path, char **gt_path, char **symm_svoxels_img_path,
                     float *svoxel_vol_perc, float *patch_lesion_perc, iftIntArray **patch_sizes,
                     char **out_dir, char **aux_dir);

iftIntArray *iftGridSamplingOnSymmetricSVoxelImage(  iftImage *symm_svoxels_img, double svoxel_vol_perc);
iftIntArray *iftGridSamplingOnSymmetricSVoxels(  iftImage *svoxels_img_RH, int svoxel,
                                               iftBoundingBox bb_svoxel_RH, int mid_sagittal_plane,
                                               double svoxel_vol_perc, int svoxel_vol);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path = NULL;
    char *gt_path = NULL;
    char *symm_svoxels_img_path = NULL;
    float svoxel_vol_perc = 0.1;
    float patch_lesion_perc = 0.5;
    iftIntArray *patch_sizes = NULL;
    char *out_dir = NULL;
    char *aux_dir = NULL;

    char *filename = NULL;


    iftGetInputArgs(args, &img_path, &gt_path, &symm_svoxels_img_path,
                    &svoxel_vol_perc, &patch_lesion_perc, &patch_sizes,
                    &out_dir, &aux_dir);

    timer *t1 = iftTic();

    const char *img_ext = iftFileExt(img_path);
    char *img_base = iftFilename(img_path, img_ext);

    iftImage *img = iftReadImageByExt(img_path);
    iftImage *gt_img = iftReadImageByExt(gt_path);
    iftImage *symm_svoxels_img = iftReadImageByExt(symm_svoxels_img_path);

    int max_patch_size = IFT_INFINITY_INT_NEG;
    for (int i = 0; i < patch_sizes->n; i++) {
        max_patch_size = iftMax(max_patch_size, patch_sizes->val[i]);
    }
    int frame_size = max_patch_size / 2;

    iftImage *aux = img;
    img = iftAddFrame(img, frame_size, 0);
    if (aux_dir) {
        filename = iftFilename(img_path, NULL);
        iftWriteImageByExt(img, iftJoinPathnames(3, aux_dir, "imgs", filename));
        iftFree(filename);
    }
    iftDestroyImage(&aux);

    aux = gt_img;
    gt_img = iftAddFrame(gt_img, frame_size, 0);
    if (aux_dir) {
        filename = iftCopyString("%s_lesions%s", img_base, img_ext);
        iftWriteImageByExt(gt_img, iftJoinPathnames(3, aux_dir, "imgs", filename));
        iftFree(filename);
    }
    iftDestroyImage(&aux);

    aux = symm_svoxels_img;
    symm_svoxels_img = iftAddFrame(symm_svoxels_img, frame_size, 0);
    if (aux_dir) {
        filename = iftCopyString("%s_detected_svoxels%s", img_base, img_ext);
        iftWriteImageByExt(symm_svoxels_img, iftJoinPathnames(3, aux_dir, "imgs", filename));
        iftFree(filename);
    }
    iftDestroyImage(&aux);


    iftIntArray *points = iftGridSamplingOnSymmetricSVoxelImage(symm_svoxels_img, svoxel_vol_perc);

    for (int i = 0; i < patch_sizes->n; i++) {
        int patch_vol = patch_sizes->val[i] * patch_sizes->val[i] * patch_sizes->val[i];
        char *patch_subdir = iftCopyString("%dx%dx%d", patch_sizes->val[i], patch_sizes->val[i], patch_sizes->val[i]);

        iftBoundingBoxArray *patches = iftBoundingBoxesAroundVoxels(img, points, patch_sizes->val[i]);

        iftCSV *csv = iftCreateCSV(patches->n, 7);
        iftSetCSVHeader(csv, "x0,y0,z0,x1,y1,z1,class");
        
        iftImage *normal_patch_label_img = iftCreateImageFromImage(img);
        iftImage *abnormal_patch_label_img = iftCreateImageFromImage(img);

        for (int j = 0; j < patches->n; j++) {
            iftBoundingBox patch = patches->val[j];
            if ((patch.end.x - patch.begin.x + 1 != patch_sizes->val[i]) ||
                (patch.end.y - patch.begin.y + 1 != patch_sizes->val[i]) ||
                (patch.end.z - patch.begin.z + 1 != patch_sizes->val[i])) {
                iftPrintBoundingBox(patch);
                iftError("Invalid Patch", "main");
            }

            iftImage *img_roi = iftExtractROI(img, patch);
            iftImage *gt_roi = iftExtractROI(gt_img, patch);

            int n_gt_voxels = gt_roi->n - iftCountBackgroundSpels(gt_roi);

            int patch_class;
            if (n_gt_voxels < (patch_lesion_perc * patch_vol)) {
                patch_class = 1; // normal
                iftFillBoundingBoxInImage(normal_patch_label_img, patch, patch_class);
            }
            else {
                patch_class = 2; // abnormal
                iftFillBoundingBoxInImage(abnormal_patch_label_img, patch, patch_class);
            }

            sprintf(csv->data[j][0], "%d", patch.begin.x);
            sprintf(csv->data[j][1], "%d", patch.begin.y);
            sprintf(csv->data[j][2], "%d", patch.begin.z);
            sprintf(csv->data[j][3], "%d", patch.end.x);
            sprintf(csv->data[j][4], "%d", patch.end.y);
            sprintf(csv->data[j][5], "%d", patch.end.z);
            sprintf(csv->data[j][6], "%d", patch_class);

            char *filename = iftCopyString("%s_%d_%d-%d-%d_%d-%d-%d%s", img_base, patch_class,
                                           patch.begin.x, patch.begin.y, patch.begin.z,
                                           patch.end.x, patch.end.y, patch.end.z,
                                           img_ext);
            char *out_path = iftJoinPathnames(3, out_dir, patch_subdir, filename);
            iftWriteImageByExt(img_roi, out_path);

            iftDestroyImage(&img_roi);
            iftDestroyImage(&gt_roi);
            iftFree(filename);
            iftFree(out_path);
        }
        iftDestroyBoundingBoxArray(&patches);

        if (aux_dir) {
            char *aux_patch_subdir = iftJoinPathnames(2, aux_dir, patch_subdir);
            if (!iftDirExists(aux_patch_subdir)) {
                iftMakeDir(aux_patch_subdir);
            }

            char *filename = iftCopyString("%s_patches.csv", img_base);
            char *out_csv_path = iftJoinPathnames(2, aux_patch_subdir, filename);
            iftWriteCSV(csv, out_csv_path, ';');
            iftFree(filename);
            iftFree(out_csv_path);

            filename = iftCopyString("%s_normal_patches.nii.gz", img_base);
            char *out_normal_patch_label_img_path = iftJoinPathnames(2, aux_patch_subdir, filename);
            iftWriteImageByExt(normal_patch_label_img, out_normal_patch_label_img_path);
            iftFree(out_normal_patch_label_img_path);
            iftFree(filename);

            filename = iftCopyString("%s_abnormal_patches.nii.gz", img_base);
            char *out_abnormal_patch_label_img_path = iftJoinPathnames(2, aux_patch_subdir, filename);
            iftWriteImageByExt(abnormal_patch_label_img, out_abnormal_patch_label_img_path);
            iftFree(out_abnormal_patch_label_img_path);
            iftFree(filename);

            iftFree(aux_patch_subdir);
        }        
        iftDestroyCSV(&csv);
        iftDestroyImage(&normal_patch_label_img);
        iftDestroyImage(&abnormal_patch_label_img);
        iftFree(patch_subdir);
    }


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyIntArray(&patch_sizes);
    iftDestroyImage(&img);
    iftDestroyImage(&gt_img);
    iftDestroyImage(&symm_svoxels_img);
    iftDestroyIntArray(&points);


    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] =
        "Given a label image with symmetrical supervoxels, this program generates multi-scaled 3D cubic patches as follows:\n"
        "(1) For each supervoxel s, uniformly sample k voxels inside s, where k = perc * volume_of_s;\n"
        "(2) For each sampled point p, extract multi-scaled 3D cubic patched centered on p whose sizes are defined by the user;\n"
        "(3) A 3D cubic patch is considered abnormal if at least X\\% of it voxels are lesion (as indicated by the GT)\n\n"
        "Since only one svoxel of the pair of symmetrical svoxels contains the lesion, we sampled patches in both svoxels of the pair separately.\n\n"
        "The output patch images follow the pattern: subjectid_imageid_class_x0-y0-z0_x1-y1-z1.nii.gz, where: \n"
        "class = 01 if normal, and 02 if abnormal, and\n"
        "the 3D patch image is defined by the following points in the input image: begin: x0,y0,z0; end: x1,y1,y2\n"
        "It also saves a CSV file with patch coordinates as well as a label Image where normal patch voxels are 1 and "
        "abornomal ones are 2.\n\n"
        "==> Example of output:\n"
        "Given an image 000010_000001.nii.gz (subjectid_imageid.nii.gz), the sizes 5,15, and the output dir: './patches', the output files will be:\n\n"
        "+ ./patches\n"
        "      + 5x5x5\n"
        "            - 000010_000001_patches.csv\n"
        "            - 000010_000001_patches.nii.gz\n"
        "            - 000010_000001_01_100-50-60_104-54-64.nii.gz (normal)\n"
        "            - 000010_000001_02_200-100-30_204-104-34.nii.gz (abnormal)\n"
        "            - ...\n"
        "      + 15x15x15\n"
        "            - 000010_000001_01_95-45-55_109-59-69.nii.gz (normal)\n"
        "            - 000010_000001_02_195-95-25_209-109-39.nii.gz (abnormal)\n"
        "            - ...\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Image."},
        {.short_name = "-m", .long_name = "--ground-truth-lesion-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label image with the ground-truth segmentations for the lesions."},
        {.short_name = "-l", .long_name = "--symmetrical-supervoxels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label image with the considered simmetrical supervoxels (with respect to the image's MSP)."},
        {.short_name = "-p", .long_name = "--svoxel-volume-perc", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Percentage of Voxels inside each supervoxels that will be sampled. E.g: 0.1"},
        {.short_name = "-g", .long_name = "--patch-lesion-voxel-perc", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Min. Perc. of voxels inside that patch that must be lesion to define it as abnormal. E.g: 0.5"},
        {.short_name = "-s", .long_name = "--patch-sizes", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Sizes of the 3D cubic patches for each Scale. E.g: '5,10,15' will extract 3D patches of 3 " \
                               "sizes for the same sampled points: 5x5x5, 10x10x10, 15x15x15"},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output root directory where the image patches will be saved." \
                               "sizes for the same sampled points: 5x5x5, 10x10x10, 15x15x15"},
        {.short_name = "-x", .long_name = "--aux-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Auxiliary directory to save intermediate files."},
    };

    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


iftIntArray *iftGetPatchSizesFromCmdLine(const char *patch_sizes_str) {

    if (!iftRegexMatch(patch_sizes_str, "^([0-9]+,?)+$")) {
        iftError("Invalid Input Patch Sizes from command line: %s",
                 "iftGetPatchSizesFromCmdLine", patch_sizes_str);
    }

    iftSList *SL = iftSplitString(patch_sizes_str, ",");
    iftIntArray *patch_sizes = iftCreateIntArray(SL->n);

    iftSNode *snode = SL->head;
    for (size_t i = 0; i < SL->n; i++) {
        patch_sizes->val[i] = atoi(snode->elem);
        if (patch_sizes->val[i] <= 0) {
            iftError("Invalid Patch Size: %d <= 0", "iftGetPatchSizesFromCmdLine",
                     patch_sizes->val[i]);
        }
        snode = snode->next;
    }
    iftDestroySList(&SL);

    return patch_sizes;
}


void iftGetInputArgs(  iftDict *args, char **img_path, char **gt_path, char **symm_svoxels_img_path,
                     float *svoxel_vol_perc, float *patch_lesion_perc, iftIntArray **patch_sizes,
                     char **out_dir, char **aux_dir) {
    *img_path = iftGetStrValFromDict("--image", args);
    *gt_path = iftGetStrValFromDict("--ground-truth-lesion-image", args);
    *symm_svoxels_img_path = iftGetStrValFromDict("--symmetrical-supervoxels", args);
    *svoxel_vol_perc = iftGetDblValFromDict("--svoxel-volume-perc", args);
    *patch_lesion_perc = iftGetDblValFromDict("--patch-lesion-voxel-perc", args);
    *out_dir = iftGetStrValFromDict("--output-dir", args);

    if (iftDictContainKey("--aux-dir", args, NULL)) {
        *aux_dir = iftGetStrValFromDict("--aux-dir", args);
        iftMakeDir(*aux_dir);
    }
    else { *aux_dir = NULL; }


    const char *patch_sizes_str = iftGetConstStrValFromDict("--patch-sizes", args);
    *patch_sizes = iftGetPatchSizesFromCmdLine(patch_sizes_str);

    puts("-----------------------");
    printf("- Image: %s\n", *img_path);
    printf("- Lesion GT: %s\n", *gt_path);
    printf("- Symmetrical SVoxels Image: %s\n", *symm_svoxels_img_path);
    printf("- SVoxel Volume Perc. for Sampling: %f\n", *svoxel_vol_perc);
    printf("- Perc. of voxels inside that patch that must be lesion to define it as abnormal: %f\n", *patch_lesion_perc);
    printf("- 3D cubic Patch Sizes: ");
    for (int i = 0; i < (*patch_sizes)->n; i++) {
        printf("%dx%dx%d, ", (*patch_sizes)->val[i], (*patch_sizes)->val[i], (*patch_sizes)->val[i]);
    }
    printf("\n- Output Patch Dir: %s\n", *out_dir);
    puts("-----------------------");
    if (*aux_dir) {
        printf("- Aux Dir: %s\n", *aux_dir);
    }
    puts("-----------------------\n");
}


iftIntArray *iftGridSamplingOnSymmetricSVoxelImage(  iftImage *symm_svoxels_img, double svoxel_vol_perc) {
    int mid_sagittal_plane = symm_svoxels_img->xsize / 2;

    iftBoundingBox bb_LH = {
        .begin = {.x = symm_svoxels_img->xsize / 2, .y = 0, .z = 0},
        .end   = {.x = symm_svoxels_img->xsize - 1, .y = symm_svoxels_img->ysize - 1, .z = symm_svoxels_img->zsize - 1}};

    iftImage *svoxels_img_RH = iftCopyImage(symm_svoxels_img);
    iftFillBoundingBoxInImage(svoxels_img_RH, bb_LH, 0);
    // iftWriteImageByExt(svoxels_img_RH, "tmp/svoxels_img_RH.nii.gz");

    iftIntArray *svoxels = iftGetObjectLabels(svoxels_img_RH);
    iftIntArray *svoxel_vols = iftCountLabelSpels(svoxels_img_RH);
    iftBoundingBox *bb_svoxel_RH_arr = iftMinLabelsBoundingBox(svoxels_img_RH, svoxels, NULL);

    iftIntArray **svoxel_points_arr = iftAlloc(svoxels->n, sizeof(iftIntArray *));

    long n_total_points = 0;

    #pragma omp parallel for reduction(+:n_total_points)
    for (int o = 0; o < svoxels->n; o++) {
        int svoxel_label = svoxels->val[o];
        svoxel_points_arr[o] = iftGridSamplingOnSymmetricSVoxels(svoxels_img_RH, svoxel_label,
                                                                 bb_svoxel_RH_arr[o], mid_sagittal_plane,
                                                                 svoxel_vol_perc, svoxel_vols->val[svoxel_label]);
        n_total_points += svoxel_points_arr[o]->n;
    }

    // merge all points from the svoxels into a single array
    int idx = 0;
    iftIntArray *svoxel_points = iftCreateIntArray(n_total_points);

    for (int o = 0; o < svoxels->n; o++) {
        for (int i = 0; i < svoxel_points_arr[o]->n; i++) {
            svoxel_points->val[idx] = svoxel_points_arr[o]->val[i];
            idx++;
        }
        iftDestroyIntArray(&svoxel_points_arr[o]);
    }


    iftDestroyImage(&svoxels_img_RH);
    iftDestroyIntArray(&svoxels);
    iftDestroyIntArray(&svoxel_vols);
    iftFree(bb_svoxel_RH_arr);
    iftFree(svoxel_points_arr);

    return svoxel_points;
}


iftIntArray *iftGridSamplingOnSymmetricSVoxels(  iftImage *svoxels_img_RH, int svoxel,
                                               iftBoundingBox bb_svoxel_RH, int mid_sagittal_plane,
                                               double svoxel_vol_perc, int svoxel_vol) {
    iftImage *svoxel_img_RH_roi = iftExtractROI(svoxels_img_RH, bb_svoxel_RH);
    iftImage *aux = svoxel_img_RH_roi;
    svoxel_img_RH_roi = iftExtractObject(svoxel_img_RH_roi, svoxel);
    int initial_obj_voxel_idx = iftGetVoxelIndex(svoxel_img_RH_roi, iftGeometricCenterVoxel(svoxel_img_RH_roi));

    int n_samples = iftMax(1, svoxel_vol_perc * svoxel_vol);

    float radius = iftEstimateGridOnMaskSamplingRadius(svoxel_img_RH_roi, initial_obj_voxel_idx, n_samples);
    iftIntArray *points_RH_roi = iftGridSamplingOnMask(svoxel_img_RH_roi, radius, initial_obj_voxel_idx, -1);
    // printf("SVoxel: %d\nVol: %d\nn_samples: %d, radius = %f, n_points = %ld\n\n", svoxel, svoxel_vol, n_samples, radius, points_RH_roi->n);

    iftList *svoxel_points_list = iftCreateList();

    for (int i = 0; i < points_RH_roi->n; i++) {
        int p_roi = points_RH_roi->val[i];
        iftVoxel u_roi = iftGetVoxelCoord(svoxel_img_RH_roi, p_roi);
        iftVoxel u_RH = {
            .x = u_roi.x + bb_svoxel_RH.begin.x,
            .y = u_roi.y + bb_svoxel_RH.begin.y,
            .z = u_roi.z + bb_svoxel_RH.begin.z
        };
        int p_RH = iftGetVoxelIndex(svoxels_img_RH, u_RH);
        iftInsertListIntoTail(svoxel_points_list, p_RH);

        int disp_x = mid_sagittal_plane - u_RH.x;

        iftVoxel u_LH = {.x = mid_sagittal_plane + disp_x,
                         .y = u_RH.y, .z = u_RH.z};
        int p_LH = iftGetVoxelIndex(svoxels_img_RH, u_LH);
        iftInsertListIntoTail(svoxel_points_list, p_LH);
    }

    iftIntArray *svoxel_points = iftListToIntArray(svoxel_points_list);

    iftDestroyImage(&svoxel_img_RH_roi);
    iftDestroyImage(&aux);
    iftDestroyList(&svoxel_points_list);

    return svoxel_points;
}
