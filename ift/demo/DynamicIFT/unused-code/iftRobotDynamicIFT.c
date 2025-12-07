#include "ift.h"
#include "ift/segm/DynamicTrees.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **gt_path, char **out_seeds_path,
                        char **out_label_img_path);
void iftGetOptionalArgs(  iftDict *args, float *alpha, int *max_n_iters, float *convergence_acc,
                        float *markers_radius, float *max_rad_border, char **out_results_file_path);
iftImage *iftSegmentationErrorImage(  iftImage *seg_img,   iftImage *gt);
iftImage *iftDrawMarkersOnImage(  iftImage *img,   iftLabeledSet *seeds);
iftImage *iftRobotIterationMSSkel(  iftImage *grad_img,   iftImage *label_img,   iftImage *gt,
                            float alpha,   iftAdjRel *A,   iftAdjRel *B,   iftAdjRel *C, iftSide side, float *dice, iftLabeledSet **seeds, int *n_seeds_pixels);
iftImage *iftSegmentationRobot(  iftImage *img,   iftImage *gt, float alpha, int max_n_iters,
                               float convergence_acc, float markers_radius, float max_rad_border, iftLabeledSet **out_seeds, float* dice, char* acc_file);
void iftObjectAreaFromPixel(  iftImage* labeled_img, int index, iftImage *area_img_out);
iftImage *iftRobotIterationCenterOfMass(  iftImage *grad_img,   iftImage *label_img,   iftImage *gt,
                                        float alpha,   iftAdjRel *B,   iftAdjRel *C, float *dice, iftLabeledSet **seeds, int *n_seeds_pixels);
iftImage* iftIntelligentMapSegmentation(iftMImage* mimg, iftLabeledSet *seeds, float alpha);
/*************************************************************/

iftImage* iftGraphCutDynamicWeights(iftMImage *mimg, iftLabeledSet *seeds,   iftAdjRel *A, int k, int beta);

#define DEC_CORRECTION 10000.0f

bool IFT_DEBUG = false;

int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *img_path           = NULL;
    char *gt_path            = NULL;
    char *out_label_img_path = NULL;
    char *out_seeds_path     = NULL;

    float alpha;
    int max_n_iters;
    float convergence_acc;
    float markers_radius;
    float max_rad_border;
    char *out_results_file_path = NULL;

    iftGetRequiredArgs(args, &img_path, &gt_path, &out_seeds_path, &out_label_img_path);
    iftGetOptionalArgs(args, &alpha, &max_n_iters, &convergence_acc, &markers_radius, &max_rad_border, &out_results_file_path);
    IFT_DEBUG = iftDictContainKey("--debug-mode", args, NULL);

    timer *t1 = iftTic();

    puts("- Reading Image");
    iftImage *img = iftReadImageByExt(img_path);

    puts("- Reading Ground Truth");
    iftImage *gt = iftReadImageByExt(gt_path);

    /* creating acc_file name */
    char *acc_file_name = iftFilename(img_path, ".ppm");
    char *prefix_acc_file = iftFilename(out_results_file_path, ".csv");
    char *acc_file = iftConcatStrings(5, "./data/", prefix_acc_file, "_", acc_file_name, ".csv");
    printf("%s\n", acc_file);
    /* end */

    puts("- Simulating the User Markers for IFT Delineation");
    iftLabeledSet *out_seeds = NULL;
    float dice = 0.0f;
    iftImage *seg_img = iftSegmentationRobot(img, gt, alpha, max_n_iters, convergence_acc, markers_radius,
                                             max_rad_border, &out_seeds, &dice, acc_file);

    puts("- Writing Resulting Segmentation Image");
    iftImage* aux_mask = iftMask(img, seg_img);
    iftWriteImageByExt(aux_mask, out_label_img_path);
    iftDestroyImage(&aux_mask);

    if (out_seeds_path != NULL) {
        puts("- Writing Resulting Seeds");
        iftWriteSeeds(out_seeds_path, out_seeds, seg_img);
    }
    int n_seeds = iftLabeledSetSize(out_seeds);

    puts("- Writing Accuracy Results");
    FILE* f;
    if(iftFileExists(out_results_file_path)){
        f = fopen(out_results_file_path, "a");
    } else {
        f = fopen(out_results_file_path, "w");
        fprintf(f, "image, n_pixels, accuracy, number_seeds\n");
    }
    fprintf(f, "%s, %5d, %.6f, %4d\n", out_label_img_path, img->n, dice, n_seeds);
    fclose(f);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&gt);
    iftDestroyImage(&seg_img);
    iftDestroyLabeledSet(&out_seeds);

    /* acc_file destroyers */
    iftFree(acc_file_name);
    iftFree(prefix_acc_file);
    iftFree(acc_file);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Generate the markers and Segment a 2D image by using a Robot and the Dynnamic Object Delination.\n" \
        "- Dice similarity is used to evaluate the segmentation accuracy.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input 2D Image."},
        {.short_name = "-g", .long_name = "--ground-truth", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Ground Truth for the Input Image."},
        {.short_name = "-m", .long_name = "--output-generated-markers", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname to save the output generated markers used for delineation."},
        {.short_name = "-o", .long_name = "--output-segmented-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname to save the output segmented image."},
        {.short_name = "-o", .long_name = "--output-segmented-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname to save the output segmented image."},
        {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Alpha for arc weights balance. Default: 0.5"},
        {.short_name = "-n", .long_name = "--max-num-iterations", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Maximum number of Iterations used by the Robot. Default: 20"},
        {.short_name = "-t", .long_name = "--convergence-accuracy", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Dice Convergence Accuracy used to stop the Robot. Default: 0.9"},
        {.short_name = "-r", .long_name = "--markers-adjacency-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Radius used to get a marker as a circle. Default: 1.4"},
        {.short_name = "-R", .long_name = "--markers-distance-radius-border", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Radius used to get a marker as a circle. Default: 3.0"},
        {.short_name = "-N", .long_name = "--name-results-file", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname to file to write results information"},
        {.short_name = "-d", .long_name = "--debug-mode", .has_arg=false,
         .required=false, .help="Debug Mode: Write the resulting images along the robot iteration"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **gt_path, char **out_seeds_path,
                        char **out_label_img_path) {
    *img_path = iftGetStrValFromDict("--input-image", args);
    *gt_path = iftGetStrValFromDict("--ground-truth", args);
    *out_seeds_path = iftGetStrValFromDict("--output-generated-markers", args);
    *out_label_img_path = iftGetStrValFromDict("--output-segmented-image", args);

    puts("-----------------------");
    printf("- Input Image: %s\n", *img_path);
    printf("- Ground Truth: %s\n", *gt_path);
    printf("- Output Generated Markers: %s\n", *out_seeds_path);
    printf("- Output Segmented Image: %s\n", *out_label_img_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *alpha, int *max_n_iters, float *convergence_acc,
                        float *markers_radius, float *max_rad_border, char **out_results_file_path) {
    if (iftDictContainKey("--alpha", args, NULL))
        *alpha = iftGetDblValFromDict("--alpha", args);
    else *alpha = 0.5;

    if (iftDictContainKey("--max-num-iterations", args, NULL))
        *max_n_iters = iftGetLongValFromDict("--max-num-iterations", args);
    else *max_n_iters = 20;

    if (iftDictContainKey("--convergence-accuracy", args, NULL))
        *convergence_acc = iftGetDblValFromDict("--convergence-accuracy", args);
    else *convergence_acc = 0.9;

    if (iftDictContainKey("--markers-adjacency-radius", args, NULL))
        *markers_radius = iftGetDblValFromDict("--markers-adjacency-radius", args);
    else *markers_radius = 1.4;

    if (iftDictContainKey("--markers-distance-radius-border", args, NULL))
        *max_rad_border = iftGetDblValFromDict("--markers-distance-radius-border", args);
    else *max_rad_border = 3.0;

    if(iftDictContainKey("--name-results-file", args, NULL))
        *out_results_file_path = iftGetStrValFromDict("--name-results-file", args);
    else {
        *out_results_file_path = iftCopyString("./results.txt");
    }

    printf("- Alpha: %f\n", *alpha);
    printf("- Max. Number of Iterations: %d\n", *max_n_iters);
    printf("- Convergence Accuracy: %f\n", *convergence_acc);
    printf("- Markers' Adjacency Radius: %f\n", *markers_radius);
    printf("- Markers' Max. Radius from Border: %f\n", *max_rad_border);
    printf("- Results file: %s\n", *out_results_file_path);
    puts("-----------------------\n");
}


iftImage *iftSegmentationErrorImage(  iftImage *seg_img,   iftImage *gt) {
    iftImage *error = iftCreateImageFromImage(seg_img);

    for (int p = 0; p < seg_img->n; p++)
        if (seg_img->val[p] != gt->val[p])
            error->val[p] = gt->val[p] + 1; // increases for that bg errors have label 1

    return error;
}

iftImage *iftDrawMarkersOnImage(  iftImage *img,   iftLabeledSet *seeds) {
    iftDict *label_dict = iftCreateDict();

    int n = 0;
      iftLabeledSet *node = seeds;
    while (node != NULL) {
        if (!iftDictContainKey((int) node->label, label_dict, NULL)) {
            iftInsertIntoDict((int) node->label, n, label_dict);
            n++;
        }
        node = node->next;
    }

    iftColorTable *ctb = iftCreateColorTable(n);

    /* color equal to usis */
    ctb->color[1].val[0] = 101;
    ctb->color[1].val[1] = 181;
    ctb->color[1].val[2] = 225;

    ctb->color[0].val[0] = 121;
    ctb->color[0].val[1] = 194;
    ctb->color[0].val[2] = 70;

    iftImage *draw_img = iftCopyImage(img);
    if (!iftIsColorImage(draw_img))
        iftSetCbCr(draw_img, 128);

    node = seeds;
    while (node != NULL) {
        int p = node->elem;
        int idx = iftGetLongValFromDict((int) node->label, label_dict);
        iftColor YCbCr = ctb->color[idx];

        draw_img->val[p] = YCbCr.val[0];
        draw_img->Cb[p] = YCbCr.val[1];
        draw_img->Cr[p] = YCbCr.val[2];

        node = node->next;
    }


    iftDestroyColorTable(&ctb);
    iftDestroyDict(&label_dict);


    return draw_img;
}

iftImage *iftRobotIterationMSSkel(  iftImage *grad_img,   iftImage *label_img,   iftImage *gt,
                            float alpha,   iftAdjRel *A,   iftAdjRel *B,   iftAdjRel *C, iftSide side, float *dice, iftLabeledSet **seeds, int *n_seeds_pixels) {
    iftImage *relabel_img = NULL, *dist_img;
    iftFImage *msskel = iftMSSkel2D(label_img, A, side, &dist_img, &relabel_img);

    iftIntArray *labels = iftGetObjectLabels(relabel_img);
    iftIntArray *max_vals = iftCreateIntArray(labels->n + 1); // include bg
    iftIntArray *max_pos = iftCreateIntArray(labels->n + 1); // include bg
    iftIntArray *max_area = iftCreateIntArray(labels->n + 1);
    iftIntArray *max_dist = iftCreateIntArray(labels->n + 1);
    max_pos->val[0] = -1; // bg could not have skeleton, then there is no pixel for it

    /* New priority for seeds selection */
    iftImage *area_img = iftCreateImageFromImage(label_img);

    for (int p = 0; p < msskel->n; p++) {
        // bg pixels has object relabeled values after edt computing, then
        // we need to check its original value to guarantee the real label of bg pixels
        int label = (gt->val[p] == 0) ? 0 : relabel_img->val[p];

        if(area_img->val[p] == 0) {
            iftObjectAreaFromPixel(relabel_img, p, area_img);
        }

        if ((area_img->val[p] >= max_area->val[label]) && (msskel->val[p] >= max_vals->val[label]) && (dist_img->val[p] > max_dist->val[label]) ) {
            max_vals->val[label] = msskel->val[p];
            max_pos->val[label]  = p;
            max_area->val[label] = area_img->val[p];
            max_dist->val[label] = dist_img->val[p];
        }
    }
    iftDestroyImage(&area_img);

    iftQuickSort(max_area->val, max_pos->val, 0, max_pos->n - 1, IFT_DECREASING);

    int limit_seeds = 2;
    int qnt_seeds = 0;

    for (int o = 0; o < max_pos->n; o++) {
        if (max_pos->val[o] != -1) {
            iftVoxel u = iftGetVoxelCoord(gt, max_pos->val[o]);

//            /* shift the seed if it's touching a border */
            for (int i = C->n-1; i > 0; i--){
                iftVoxel v = iftGetAdjacentVoxel(C, u, i);
                if(iftValidVoxel(gt, v) && (iftImgVoxelVal(gt, u) != iftImgVoxelVal(gt, v))){
                    u.x -= C->dx[i];
                    u.y -= C->dy[i];
                    u.z -= C->dz[i];
                }
            }

            int in_border = 0;

            /* check if the seed it's touching a border */
            for (int i = 0; i < C->n; i++){
                iftVoxel v = iftGetAdjacentVoxel(C, u, i);
                if(iftValidVoxel(gt, v) && (gt->val[max_pos->val[o]] != iftImgVoxelVal(gt, v))){
                    in_border = 1;
                }
            }

            if(in_border == 0 && qnt_seeds < limit_seeds) {
                qnt_seeds++;
                for (int i = 0; i < B->n; i++) {
                    iftVoxel v = iftGetAdjacentVoxel(B, u, i);

//                    if (iftValidVoxel(gt, v) && (iftImgVoxelVal(gt, u) == iftImgVoxelVal(gt, v))) {
                    if (iftValidVoxel(gt, v)){
                        /* It was adding the same seed multiple times */
                        if(!iftLabeledSetHasElement(*seeds, iftGetVoxelIndex(gt, v))) { /* Temporary fix */
                            iftInsertLabeledSet(seeds, iftGetVoxelIndex(gt, v), gt->val[max_pos->val[o]]);
                            (*n_seeds_pixels)++;
                        }
                    }
                }
            }
        }
    }
    iftAdjRel* D = iftCircular(1.0f);
//    iftAdjRel* D = iftCircular(sqrtf(9.0f));
    // iftAdjRel* D = iftCircular(sqrtf(2.0f));
    iftMImage *mimg = iftImageToMImage(grad_img, LABNorm_CSPACE);
    iftImage *basins = iftMImageBasins(mimg, D);
//    iftImage *seg_img = iftWatershed(basins, D, *seeds, NULL);
//    iftImage *seg_img = iftIntelligentMapSegmentation(iftImageToMImage(grad_img, LABNorm_CSPACE), *seeds, alpha);
//    iftImage *seg_img = iftDynamicObjectDelineation(mimg, *seeds, D, 500);
//    iftImage *seg_img = iftDynamicObjectDelinForest(mimg, *seeds, D, 500);
   iftImage* seg_img = iftGraphCut(basins, *seeds, 90);
    // iftImage* seg_img = iftGraphCutDynamicWeights(mimg, *seeds, A, 500, 90);
    iftDestroyAdjRel(&D);

    iftDestroyMImage(&mimg);
   iftDestroyImage(&basins);

    *dice = iftDiceSimilarity(seg_img, gt);

    iftDestroyImage(&relabel_img);
    iftDestroyImage(&dist_img);
    iftDestroyFImage(&msskel);
    iftDestroyIntArray(&max_vals);
    iftDestroyIntArray(&max_pos);
    iftDestroyIntArray(&max_area);
    iftDestroyIntArray(&max_dist);

    return seg_img;
}

iftImage *iftSegmentationRobot(  iftImage *img,   iftImage *gt, float alpha, int max_n_iters,
                               float convergence_acc, float markers_radius, float max_rad_border, iftLabeledSet **out_seeds, float* dice, char* acc_file) {

    FILE *file = fopen(acc_file, "w");

    iftAdjRel *A = iftCircular(sqrtf(2.0));
    iftAdjRel *B = iftCircular(markers_radius);
    iftAdjRel *C = iftCircular(max_rad_border);
    
    iftLabeledSet *seeds = NULL;

    for(int p = 0; p < gt->n; p++){
        gt->val[p] = (gt->val[p] > 127) ? 1 : 0;
    }

    int new_n_seeds_pixels = 0;
    iftImage *seg_img = iftRobotIterationMSSkel(img, gt, gt, alpha, A, B, C, IFT_BOTH, dice, &seeds, &new_n_seeds_pixels);
    int n_iters       = 1;
    int n_seeds_pixels = 0;
    float last_acc = -1.0f;

    fprintf(file, "iteration, accuracy, labeled_pixels\n");
    /* MSSkel seeds selector */
    // while ((*dice < convergence_acc) && (n_iters < max_n_iters) && (new_n_seeds_pixels != n_seeds_pixels)) {
//    while ((*dice < convergence_acc) && (n_iters < max_n_iters) && (new_n_seeds_pixels != n_seeds_pixels) && (fabsf((*dice) - last_acc) > 0.001f)) {
   while ((n_iters < max_n_iters) && (new_n_seeds_pixels != n_seeds_pixels) && ((fabsf((*dice) - last_acc) >= 0.01f) || (*dice < convergence_acc))) { /* higher than a given threshold and after convergence */
        iftImage *error = iftSegmentationErrorImage(seg_img, gt);
        printf("[%03d] dice = %f\n", n_iters-1, *dice);
        fprintf(file, "%d, %f, %d\n", n_iters, *dice, new_n_seeds_pixels);

        last_acc = *dice;

        if (IFT_DEBUG) {
            iftImage *draw_img = iftDrawMarkersOnImage(img, seeds);
            iftWriteImageByExt(draw_img, "run/draw_img_%d.png", n_iters-1);
            iftWriteImageByExt(iftNormalize(seg_img, 0, 255), "run/seg_img_%d.png", n_iters-1);
            iftWriteImageByExt(iftNormalize(error, 0, 255), "run/error_%d.png", n_iters-1);
            iftDestroyImage(&seg_img);
            iftDestroyImage(&draw_img);
        }

        n_seeds_pixels = new_n_seeds_pixels;
        seg_img = iftRobotIterationMSSkel(img, error, gt, alpha, A, B, C, IFT_INTERIOR, dice, &seeds, &new_n_seeds_pixels);

        if(n_seeds_pixels == new_n_seeds_pixels) { /* If no seed is added we try a alternative method for seeds selecting */
            seg_img = iftRobotIterationCenterOfMass(img, error, gt, alpha, B, C, dice, &seeds, &new_n_seeds_pixels);
        }

        n_iters++;

        iftDestroyImage(&error);
    }

    printf("[%03d] dice = %f\n", n_iters-1, *dice);
    printf("number of seeds pixels: %d\n", n_seeds_pixels);
    fprintf(file, "%d, %f, %d", n_iters, *dice, new_n_seeds_pixels);

    if (IFT_DEBUG) {
        iftWriteImageByExt(iftNormalize(seg_img, 0, 255), "run/seg_img_%d.png", n_iters-1);
        iftImage *draw_img = iftDrawMarkersOnImage(img, seeds);
        iftWriteImageByExt(draw_img, "run/draw_img_%d.png", n_iters-1);
        iftDestroyImage(&draw_img);
    }

    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    iftDestroyAdjRel(&C);

    *out_seeds = seeds;

    for(int p = 0; p < gt->n; p++){
        seg_img->val[p] = (seg_img->val[p] == 1) ? 255 : 0;
    }

    fclose(file);

    return seg_img;
}

void iftObjectAreaFromPixel(  iftImage* labeled_img, int index, iftImage *area_img){
    int area = 0;

    iftFIFO *F = iftCreateFIFO(labeled_img->n);
    iftInsertFIFO(F, index);

    iftAdjRel *A = iftCircular(1.0);

    while(!iftEmptyFIFO(F)){
        int p = iftRemoveFIFO(F);
        area++;
        iftVoxel u = iftGetVoxelCoord(labeled_img, p);

        for(int i = 0; i <A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(labeled_img, v)){
                int q = iftGetVoxelIndex(labeled_img, v);
                if(labeled_img->val[q] == labeled_img->val[p] && F->color[q] == IFT_WHITE)
                    iftInsertFIFO(F, q);
            }
        }
    }

    iftResetFIFO(F);
    iftInsertFIFO(F, index);

    while(!iftEmptyFIFO(F)){
        int p = iftRemoveFIFO(F);
        area_img->val[p] = area;
        iftVoxel u = iftGetVoxelCoord(labeled_img, p);

        for(int i = 0; i <A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(labeled_img, v)){
                int q = iftGetVoxelIndex(labeled_img, v);
                if(labeled_img->val[q] == labeled_img->val[p] && F->color[q] == IFT_WHITE)
                    iftInsertFIFO(F, q);
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyFIFO(&F);
}


iftImage *iftRobotIterationCenterOfMass(  iftImage *grad_img,   iftImage *label_img,   iftImage *gt,
                                  float alpha,   iftAdjRel *B,   iftAdjRel *C, float *dice, iftLabeledSet **seeds, int *n_seeds_pixels) {

    iftIntArray *labels = iftGetObjectLabels(label_img);
    iftIntArray *max_pos = iftCreateIntArray(labels->n + 1);
    iftIntArray *max_area = iftCreateIntArray(labels->n + 1);// include bg
    max_pos->val[0] = -1; // bg could not have skeleton, then there is no pixel for it

    /* New priority for seeds selection */
    iftImage *area_img = iftCreateImageFromImage(label_img);

    for (int p = 0; p < label_img->n; p++) {
        // bg pixels has object relabeled values after edt computing, then
        // we need to check its original value to guarantee the real label of bg pixels
        int label = gt->val[p];

        if(max_pos->val[label] == 0) {
            if (area_img->val[p] == 0) {
                iftObjectAreaFromPixel(label_img, p, area_img);
            }

            if (area_img->val[p] >= max_area->val[label]) {

                double x = 0.0f;
                double y = 0.0f;
                int q;

                for(q = 0; q < label_img->n; q++){
                    if(area_img->val[p] == area_img->val[q]){
                        x += (double) iftGetXCoord(area_img, q);
                        y += (double) iftGetYCoord(area_img, q);
                    }
                }

                iftVoxel v;
                v.x = x / q;
                v.y = y / q;
                v.z = 1;

                max_pos->val[label] = iftGetVoxelIndex(label_img, v);
            }
        }
    }

    for (int o = 0; o < max_pos->n; o++) {
        if (max_pos->val[o] != -1) {
            iftVoxel u = iftGetVoxelCoord(gt, max_pos->val[o]);

//            /* shift the seed if it's touching a border */
            for (int i = C->n-1; i > 0; i--){
                iftVoxel v = iftGetAdjacentVoxel(C, u, i);
                if(iftValidVoxel(gt, v) && (iftImgVoxelVal(gt, u) != iftImgVoxelVal(gt, v))){
                    u.x -= C->dx[i];
                    u.y -= C->dy[i];
                    u.z -= C->dz[i];
                }
            }

            int in_border = 0;

            /* check if the seed it's touching a border */
            for (int i = 0; i < C->n; i++){
                iftVoxel v = iftGetAdjacentVoxel(C, u, i);
                if(iftValidVoxel(gt, v) && (gt->val[max_pos->val[o]] != iftImgVoxelVal(gt, v))){
                    in_border = 1;
                }
            }

            if(in_border == 0) {
                for (int i = 0; i < B->n; i++) {
                    iftVoxel v = iftGetAdjacentVoxel(B, u, i);

//                    if (iftValidVoxel(gt, v) && (iftImgVoxelVal(gt, u) == iftImgVoxelVal(gt, v))) {
                    if (iftValidVoxel(gt, v)){
                        /* It was adding the same seed multiple times */
                        if(!iftLabeledSetHasElement(*seeds, iftGetVoxelIndex(gt, v))) { /* Temporary fix */
                            iftInsertLabeledSet(seeds, iftGetVoxelIndex(gt, v), gt->val[max_pos->val[o]]);
                            (*n_seeds_pixels)++;
                        }
                    }
                }
            }
        }
    }
    iftAdjRel *A = iftCircular(1.0f);
    // iftAdjRel *A = iftCircular(sqrtf(2.0));
//    iftAdjRel *A = iftCircular(sqrtf(9.0f));
    iftMImage *mimg = iftImageToMImage(grad_img, LABNorm_CSPACE);
    iftImage *basins = iftMImageBasins(mimg, A);
//
//    iftImage *seg_img = iftWatershed(basins, A, *seeds, NULL);
//    iftImage *seg_img = iftIntelligentMapSegmentation(iftImageToMImage(grad_img, LABNorm_CSPACE), *seeds, alpha);
//    iftImage *seg_img = iftDynamicObjectDelineation(mimg, *seeds, A, 500);
//    iftImage *seg_img = iftDynamicObjectDelinForest(mimg, *seeds, A, 500);
//    iftImage *smooth_seg_img = iftSmoothRegionsByDiffusion(seg_img, grad_img, 0.1, 5);
   iftImage* seg_img = iftGraphCut(basins, *seeds, 90);
//    iftImage* seg_img = iftGraphCutDynamicWeights(mimg, *seeds, A, 500, 90);
    *dice = iftDiceSimilarity(seg_img, gt);

//    iftDestroyImage(&seg_img);
    iftDestroyAdjRel(&A);
    iftDestroyMImage(&mimg);
    iftDestroyImage(&basins);

    iftDestroyIntArray(&max_pos);
    iftDestroyIntArray(&max_area);

    return seg_img;
}


iftImage* iftIntelligentMapSegmentation(iftMImage* mimg, iftLabeledSet *seeds, float alpha){

    iftAdjRel *A = NULL;

    if (iftIs3DMImage(mimg))
        A = iftSpheric(1.0);
    else A = iftCircular(1.0);

    iftImage *label = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftImage *pathval = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftImage *objmap = iftMEnhanceObject(mimg, seeds, 1);

    int norm_value = iftNormalizationValue(iftMMaximumValue(mimg, -1));

    iftGQueue *Q = iftCreateGQueue(norm_value + 1, mimg->n, pathval->val);

    for(int p = 0; p < mimg->n; p++){
        pathval->val[p] = IFT_INFINITY_INT;
    }

    for(iftLabeledSet *S = seeds; S != NULL; S = S->next){
        int p = S->elem;
        pathval->val[p] = 0;
        label->val[p] = S->label;
        iftInsertGQueue(&Q, p);
    }

    while(!iftEmptyGQueue(Q)){
        int p = iftRemoveGQueue(Q);
        iftVoxel u = iftMGetVoxelCoord(mimg, p);

        for(int i = 1; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(mimg, v)){
                int q = iftGetVoxelIndex(mimg, v);

                int arc_weight;
                if(label->val[p] == 1)
                    arc_weight = alpha * (norm_value - objmap->val[q]) + (1 - alpha) * iftMImageDist(mimg, p, q);
                else arc_weight = alpha * objmap->val[q] - (1 - alpha) * iftMImageDist(mimg, p, q);

                int tmp = iftMax(pathval->val[p], arc_weight);

                if(tmp < pathval->val[q]){
                    if(Q->L.elem[q].color == IFT_GRAY)
                        iftRemoveGQueueElem(Q, q);

                    label->val[q] =label->val[p];
                    pathval->val[q] = tmp;
                    iftInsertGQueue(&Q, q);
                }
            }
        }

    }

    iftDestroyAdjRel(&A);
    iftDestroyImage(&objmap);
    iftDestroyImage(&pathval);
    iftDestroyGQueue(&Q);

    return label;
}
/*************************************************************/


iftImage* iftGraphCutDynamicWeights(iftMImage *mimg, iftLabeledSet *seeds,   iftAdjRel *A, int k, int beta) {

    iftImage *arc_weight = iftDynamicArcWeightsForest(mimg, seeds, A, k);
    iftImage *mask = iftGraphCut(arc_weight, seeds, beta);

    iftDestroyImage(&arc_weight);

    return mask;
}