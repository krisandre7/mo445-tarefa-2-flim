#include "ift.h"
#include "ift/segm/DynamicTrees.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(iftDict *args, char **img_path, char **markers_path, char **out_img_path, char **gt_img_path,
                        char **dataset_name, char **scribble_tag, float *adj_rel_r);
/*************************************************************/

int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *in_dir_path = NULL, *marker_dir_path  = NULL, *out_dir_path = NULL, *gt_dir_path = NULL;
    char *dataset_name = NULL, *scribble_tag = NULL;
    float adj_rel_r = 0.0f;
    FILE *dataset_file = NULL;
    timer *tic = NULL, *toc = NULL;

    iftGetRequiredArgs(args, &in_dir_path, &marker_dir_path, &out_dir_path, &gt_dir_path, &dataset_name,
                       &scribble_tag, &adj_rel_r);

    iftAdjRel *A = iftCircular(adj_rel_r);

    iftFileSet *img_file_set = iftLoadFileSetFromDirBySuffix(in_dir_path, ".ppm", 1);

    char* dataset_path = iftConcatStrings(4, out_dir_path, "/", dataset_name, ".csv");
    dataset_file = fopen(dataset_path, "w");
    fprintf(dataset_file, "image, accuracy, time, scribble, algorithm\n");
    iftFree(dataset_path);

    for (int i = 0; i < img_file_set->n; i++) {

        char* img_path = img_file_set->files[i]->path;
        char* file_name = iftFilename(img_path, ".ppm");
        char* marker_path = iftConcatStrings(4, marker_dir_path, "/", file_name, ".txt");
        char* gt_img_path = iftConcatStrings(4, gt_dir_path, "/", file_name, ".pgm");

        printf("\n- Reading Input Image %d & Creating Auxiliary Images\n", i);

        iftImage *img = iftReadImageByExt(img_path);
        iftImage *gt_image = iftReadImageByExt(gt_img_path);
        iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);

        iftLabeledSet *seeds = iftReadSeeds(img, marker_path);

        #pragma omp parallel for
        for (int p = 0; p < gt_image->n; p++) {
            gt_image->val[p] = (gt_image->val[p] != 0);
        }

        /* variables that are updated at every algorithm run */
        double dice_acc = 0.0;
        float dif_time = 0.0f;
        char dice_and_time[256];
        iftImage* mask = NULL;
        char* this_out_path = NULL;
        char* dataset_row = NULL;
        /* end */

        puts("- Segmenting by Watershed");

        tic = iftTic();

        iftImage *mask_watershed = iftWaterCut(mimg, A, seeds, NULL);

        toc = iftToc();

        this_out_path = iftConcatStrings(4, out_dir_path, "/", file_name, "_gc_max.png");
        mask = iftMask(img, mask_watershed);
        iftWriteImageByExt(mask, this_out_path);

        dice_acc = iftDiceSimilarity(mask_watershed, gt_image);
        dif_time = iftCompTime(tic, toc);

        sprintf(dice_and_time, ", %.6lf, %.6f, ", dice_acc, dif_time);
        dataset_row = iftConcatStrings(4, file_name, dice_and_time, scribble_tag, ", GC max");

        fprintf(dataset_file, "%s\n", dataset_row);

        iftDestroyImage(&mask_watershed);
        iftDestroyImage(&mask);
        iftFree(this_out_path);
        iftFree(dataset_row);

        puts("- Segmenting by Dynamic Set Object Policy");

        tic = iftTic();

        iftImage *mask_obj_policy = iftDynamicSetObjectPolicy(mimg, A, seeds);

        toc = iftToc();

        this_out_path = iftConcatStrings(4, out_dir_path, "/", file_name, "_obj_policy.png");
        mask = iftMask(img, mask_obj_policy);
        iftWriteImageByExt(mask, this_out_path);

        dice_acc = iftDiceSimilarity(mask_obj_policy, gt_image);
        dif_time = iftCompTime(tic, toc);

        sprintf(dice_and_time, ", %.6lf, %.6f, ", dice_acc, dif_time);
        dataset_row = iftConcatStrings(4, file_name, dice_and_time, scribble_tag, ", object policy");

        fprintf(dataset_file, "%s\n", dataset_row);

        iftDestroyImage(&mask_obj_policy);
        iftDestroyImage(&mask);
        iftFree(this_out_path);
        iftFree(dataset_row);

        puts("- Segmenting by GraphCut");

        tic = iftTic();

        iftImage *mask_graphcut = iftGraphCutFromMImage(mimg, seeds, 100);

        toc = iftToc();

        this_out_path = iftConcatStrings(4, out_dir_path, "/", file_name, "_gc_sum.png");
        mask = iftMask(img, mask_graphcut);
        iftWriteImageByExt(mask, this_out_path);

        dice_acc = iftDiceSimilarity(mask_graphcut, gt_image);
        dif_time = iftCompTime(tic, toc);

        sprintf(dice_and_time, ", %.6lf, %.6f, ", dice_acc, dif_time);
        dataset_row = iftConcatStrings(4, file_name, dice_and_time, scribble_tag, ", GC sum");

        fprintf(dataset_file, "%s\n", dataset_row);

        iftDestroyImage(&mask_graphcut);
        iftDestroyImage(&mask);
        iftFree(this_out_path);
        iftFree(dataset_row);

        puts("- Segmenting by Dynamic Set Root Policy");

        tic = iftTic();

        iftImage *mask_root_policy = iftDynamicSetRootPolicy(mimg, A, seeds);

        toc = iftToc();

        this_out_path = iftConcatStrings(4, out_dir_path, "/", file_name, "_root_policy.png");
        mask = iftMask(img, mask_root_policy);
        iftWriteImageByExt(mask, this_out_path);

        dice_acc = iftDiceSimilarity(mask_root_policy, gt_image);
        dif_time = iftCompTime(tic, toc);

        sprintf(dice_and_time, ", %.6lf, %.6f, ", dice_acc, dif_time);
        dataset_row = iftConcatStrings(4, file_name, dice_and_time, scribble_tag, ", root policy");

        fprintf(dataset_file, "%s\n", dataset_row);

        iftDestroyImage(&mask_root_policy);
        iftDestroyImage(&mask);
        iftFree(this_out_path);
        iftFree(dataset_row);

        puts("- Segmenting by Dynamic Set Min Root");

        tic = iftTic();

        iftImage *mask_min_root = iftDynamicSetMinRootPolicy(mimg, A, seeds);

        toc = iftToc();

        this_out_path = iftConcatStrings(4, out_dir_path, "/", file_name, "_min_root.png");
        mask = iftMask(img, mask_min_root);
        iftWriteImageByExt(mask, this_out_path);

        dice_acc = iftDiceSimilarity(mask_min_root, gt_image);
        dif_time = iftCompTime(tic, toc);

        sprintf(dice_and_time, ", %.6lf, %.6f, ", dice_acc, dif_time);
        dataset_row = iftConcatStrings(4, file_name, dice_and_time, scribble_tag, ", min root");

        fprintf(dataset_file, "%s\n", dataset_row);

        iftDestroyImage(&mask_min_root);
        iftDestroyImage(&mask);
        iftFree(this_out_path);
        iftFree(dataset_row);

         /* Destroying Multi Algorithm Variables */
        iftFree(img_path);
        iftFree(file_name);
        iftFree(marker_path);
        iftFree(gt_img_path);

        iftDestroyImage(&img);
        iftDestroyImage(&gt_image);
        iftDestroyMImage(&mimg);

        iftDestroyLabeledSet(&seeds);
        /* end */
    }

    puts("\nDone...");

//    puts("adj");
//    iftDestroyAdjRel(&A);
//    puts("fileset");
//    iftDestroyFileSet(&img_file_set);
//    puts("file close");
//    fclose(dataset_file);
//
//    puts("dict");
//    iftDestroyDict(&args);
//    puts("in_path");
//    iftFree(in_dir_path);
//    puts("mkr path");
//    iftFree(marker_dir_path);
//    puts("out path");
//    iftFree(out_dir_path);
//    puts("gt path");
//    iftFree(gt_dir_path);
//    puts("dataset name");
//    iftFree(dataset_name);
//    puts("scribble");
//    iftFree(scribble_tag);
//    puts("end");

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segmented an input image by the Improved IFT, which uses a new schema to estimate the arc-weights.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Input Images Directory or Single 2D Image."},
        {.short_name = "-m", .long_name = "--marker-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Input Marker(s) Directory."},
        {.short_name = "-g", .long_name = "--ground-truth-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Input Ground Truth(s) Directory."},
        {.short_name = "-o", .long_name = "--output-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Image(s) Directory."},
        {.short_name = "-d", .long_name = "--dataset-file", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Dataset File Name"},
        {.short_name = "-s", .long_name = "--scribble-tag", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Scribble Identification for Dataset."},
        {.short_name = "-a", .long_name = "--adjacency-relation", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                .required=true, .help="Adjacency Relation."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict *args, char **img_path, char **markers_path, char **out_img_path, char **gt_img_path,
                        char **dataset_name, char **scribble_tag, float *adj_rel_r) {

    *img_path      = iftGetStrValFromDict("--image-path", args);
    *markers_path  = iftGetStrValFromDict("--marker-path", args);
    *gt_img_path   = iftGetStrValFromDict("--ground-truth-path", args);
    *out_img_path  = iftGetStrValFromDict("--output-path", args);
    *dataset_name  = iftGetStrValFromDict("--dataset-file", args);
    *scribble_tag  = iftGetStrValFromDict("--scribble-tag", args);
    *adj_rel_r     = (float) iftGetDblValFromDict("--adjacency-relation", args);

    if (!iftDirExists(*out_img_path))
        iftMakeDir(*out_img_path);

    puts("-----------------------");
    printf("- Input Image Path: \"%s\"\n", *img_path);
    printf("- Markers: \"%s\"\n", *markers_path);
    printf("- Ground Truth Image: \"%s\"\n", *gt_img_path);
    printf("- Output Image Path: \"%s\"\n", *out_img_path);
    printf("- Dataset File Name: \"%s\"\n", *dataset_name);
    printf("- Scribble Dataset Tag: \"%s\"\n", *scribble_tag);
    printf("- Adjacency Relation Radius: \"%.3f\"\n", *adj_rel_r);
    puts("-----------------------\n");
}
/*************************************************************/


void iftImageToLabel(char * directory) {
    iftFileSet* fileSet = iftLoadFileSetFromDirBySuffix(directory, ".png", 0);

    for (int i = 0; i < fileSet->n; i++) {
        char *file_path = fileSet->files[i]->path;

        iftImage* markers = iftReadImageByExt(file_path);

        iftLabeledSet *seeds = NULL;

        for (int p = 0; p < markers->n; p++) {
            if (markers->val[p] == 232) {
                iftInsertLabeledSet(&seeds, p, 1);
            } else if (markers->val[p] == 65) {
                iftInsertLabeledSet(&seeds, p, 0);
            }
        }

        char *base_name = iftRemoveSuffix(file_path, ".png");

        char *marker_path = iftConcatStrings(2, base_name, ".txt");

        iftWriteSeeds2D(marker_path, seeds, markers);

        iftFree(base_name);
        iftFree(marker_path);
        iftDestroyImage(&markers);
        iftDestroyLabeledSet(&seeds);
    }

    iftDestroyFileSet(&fileSet);
}

