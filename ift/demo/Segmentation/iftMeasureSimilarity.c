/**
 * @file
 * @brief This program computes the Similarity/Distance between two Label Images to given Objects.
 * @note See the source code in @ref iftMeasureSimilarity.c
 *
 * @example iftMeasureSimilarity.c
 * @brief Computes the Similarity between two Label Images.
 * @author Samuel Martins
 * @date Jul 14, 2016
 */

#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **label_img_path, char **gt_path, char **metric);
void iftValidateRequiredArgs(const char *label_img_path, const char *gt_path, const char *metric);
iftIntArray *iftGetObjLabels(  iftDict *args,   iftImage *gt);
void iftAppendResultsInFiles(  iftIntArray *labels,   iftDblArray *results, const char *base,
                             const char *label_img_filename);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *label_img_path = NULL;
    char *gt_path        = NULL;
    char *metric         = NULL;

    iftGetRequiredArgs(args, &label_img_path, &gt_path, &metric);

    timer *t1 = iftTic();

    iftImage *label_img = iftReadImageByExt(label_img_path);
    iftImage *gt        = iftReadImageByExt(gt_path);
    
    // Computes the Similarity
    iftIntArray *labels  = iftGetObjLabels(args, gt);
    iftDblArray *results = NULL;

    // consider all labels as a single label
    if (labels == NULL) {
        puts("\n- Considering All Labels as a Single One\n");

        results = iftCreateDblArray(1);
        for (int p = 0; p < label_img->n; p++) {
            label_img->val[p] = (label_img->val[p] >= 1);
            gt->val[p]        = (gt->val[p] >= 1);
        }

        if (iftCompareStrings(metric, "DICE"))
            results->val[0] = iftDiceSimilarity(label_img, gt);
        else if (iftCompareStrings(metric, "ASSD"))
            results->val[0] = iftASSD(label_img, gt);

        puts("******** SIMILARITY *******");
        printf("* Single Object = %lf\n", results->val[0]);
        puts("***************************\n");
    }
    else {
        results    = iftCreateDblArray(labels->n);
        double avg = 0.0;

        puts("******** SIMILARITY *******");
        for (size_t o = 0; o < labels->n; o++) {
            iftImage *obj_img    = iftExtractObject(label_img, labels->val[o]);
            iftImage *gt_obj_img = iftExtractObject(gt, labels->val[o]);

            if (iftCompareStrings(metric, "DICE"))
                results->val[o] = iftDiceSimilarity(obj_img, gt_obj_img);
            else if (iftCompareStrings(metric, "ASSD"))
                results->val[o] = iftASSD(obj_img, gt_obj_img);

            avg += results->val[o];
            printf("* Label %d = %lf\n", labels->val[o], results->val[o]);

            iftDestroyImage(&obj_img);
            iftDestroyImage(&gt_obj_img);
        }
        avg /= iftMax(1, labels->n);
        
        printf("***\n* Average %lf\n", avg);
        puts("***************************\n");
    }

    // appends the resulting similarities into the text files with the passed basename 
    if (iftDictContainKey("--out-basename", args, NULL)) {
        const char *base = iftGetConstStrValFromDict("--out-basename", args);
        char *parent_dir = iftParentDir(base);
        if (!iftDirExists(parent_dir))
            iftMakeDir(parent_dir);
        iftFree(parent_dir);

        char *label_img_filename = iftFilename(label_img_path, NULL);
        
        puts("- Appending results into Output Files");
        iftAppendResultsInFiles(labels, results, base, label_img_filename);

        iftFree(label_img_filename);
    }
    

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(label_img_path);
    iftFree(gt_path);
    iftFree(metric);
    iftDestroyImage(&label_img);
    iftDestroyImage(&gt);
    iftDestroyIntArray(&labels);
    iftDestroyDblArray(&results);
    
    return 0;
}



iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program computes the Similarity/Distance between two Label Images to given Objects.\n" \
        "- You can pass the target objects separating them by colons (ex: 1:3:5).\n" \
        "- If BIN is passed for the Target Object, it measures the accuracy of background and object, i.e., " \
        "all objects (labels != 0) are considered as a single object.\n" \
        "- If nothing is passed, the program gets all objects from the Input Label Image\n\n" \
        "* ps: If an output text file basename is passed, the program appends the similarity of each object " \
        "and the average similarity of all in files with such basename.\n" \
        "- Ex: labels: 1:3 - output basename: ./accs\n" \
        "- Resulting files: ./accs_1.csv, ./accs_3.csv, ./accs_avg.csv";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--label-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label Input Image."},
        {.short_name = "-g", .long_name = "--ground-truth", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Ground Truth Image."},
        {.short_name = "-m", .long_name = "--image-metric", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Image Metric: [DICE, ASSD]."},
        {.short_name = "-j", .long_name = "--obj-labels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Target Objects. Ex: 1:2:5 (3 objects)\nUse BIN to consider all objects " \
                                "as a single one.\nDefault: Uses all labels from the Label Image"},
        {.short_name = "-o", .long_name = "--out-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Output text file basename where the similarities will be saved."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **label_img_path, char **gt_path, char **metric) {
    *label_img_path        = iftGetStrValFromDict("--label-img", args);
    *gt_path               = iftGetStrValFromDict("--ground-truth", args);
    const char *metric_aux = iftGetConstStrValFromDict("--image-metric", args);

    iftValidateRequiredArgs(*label_img_path, *gt_path, metric_aux);

    *metric = iftUpperString(metric_aux);

    puts("-----------------------");
    printf("- Label Image: \"%s\"\n", *label_img_path);
    printf("- Ground Truth: \"%s\"\n", *gt_path);
    printf("- Metric: \"%s\"\n", *metric);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *label_img_path, const char *gt_path, const char *metric) {
    if (!iftIsImageFile(label_img_path))
        iftError("Invalid Label Image: \"%s\"", "iftValidateRequiredArgs", label_img_path);

    if (!iftIsImageFile(gt_path))
        iftError("Invalid Grount Truth: \"%s\"", "iftValidateRequiredArgs", gt_path);

    char *metric_upper = iftUpperString(metric);
    if (!iftCompareStrings(metric_upper, "DICE") && !iftCompareStrings(metric_upper, "ASSD"))
        iftError("Invalid Metric: \"%s\"... Try [DICE, ASSD]", "iftValidateRequiredArgs", metric);
    iftFree(metric_upper);
}


iftIntArray *iftGetObjLabels(  iftDict *args,   iftImage *gt) {
    iftIntArray *labels = NULL;
    if (iftDictContainKey("--obj-labels", args, NULL)) {
        const char *labels_str = iftGetConstStrValFromDict("--obj-labels", args);

        if (iftCompareStrings(labels_str, "BIN")) {
            return NULL;
        }
        else {
            iftSList *SL = iftSplitString(labels_str, ":");
            labels       = iftCreateIntArray(SL->n);
            
            iftSNode *snode = SL->head;
            for (size_t o = 0; o < SL->n; o++) {
                if (!iftRegexMatch(snode->elem, "^[0-9]+$"))
                    iftError("Invalid Targer Label: %s\n", "iftGetOptionalArgs", snode->elem);
                labels->val[o] = atoi(snode->elem);
                snode = snode->next;
            }
            iftDestroySList(&SL);
        }
    }
    else {
        labels = iftGetObjectLabels(gt);
    }

    printf("- Labels: [");
    for (size_t o = 0; o < labels->n-1; o++)
        printf("%d, ", labels->val[o]);
    printf("%d]\n", labels->val[labels->n-1]);
    puts("-----------------------\n");


    return labels;
}


void iftAppendResultsInFiles(  iftIntArray *labels,   iftDblArray *results, const char *base,
                             const char *label_img_filename) {
    char out_path[IFT_STR_DEFAULT_SIZE];
    FILE *fp = NULL;

    int n_labels = (labels == NULL) ? 1 : labels->n;
    double avg = 0.0;

    // all labels as a single object
    if (labels == NULL) {
        sprintf(out_path, "%s_bin.csv", base);
        printf("  - %s\n", out_path);

        fp = fopen(out_path, "a+");
        fprintf(fp, "%s;%lf\n", label_img_filename, results->val[0]);
        fclose(fp);

        avg += results->val[0];
    }
    // appends one result per object
    else {
        for (size_t o = 0; o < labels->n; o++) {
            sprintf(out_path, "%s_obj%d.csv", base, labels->val[o]);
            printf("  - %s\n", out_path);
            
            fp = fopen(out_path, "a+");
            fprintf(fp, "%s;%lf\n", label_img_filename, results->val[o]);
            fclose(fp);

            avg += results->val[o];
        }
    }

    avg /= iftMax(1, n_labels);

    // appends the average result of all objects
    sprintf(out_path, "%s_avg.csv", base);
    printf("  - %s\n", out_path);

    fp = fopen(out_path, "a+");
    fprintf(fp, "%s;%lf\n", label_img_filename, avg);
    fclose(fp);
}












