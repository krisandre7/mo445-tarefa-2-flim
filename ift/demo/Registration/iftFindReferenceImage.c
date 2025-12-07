#include "ift.h"
#include "ift/medical/registration/Elastix.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **label_img_set, char **metric, char **out_json_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    iftFileSet *label_img_set = NULL;
    char *metric              = NULL;
    char *out_json_path       = NULL;

    iftGetRequiredArgs(args, &label_img_set, &metric, &out_json_path);

    timer *t1       = iftTic();

    char *most_sim_img_path = NULL;

    if (iftCompareStrings(metric, "RANDOM")) {
        puts("- Random Selection");
        iftRandomSeed(time(NULL));
        int i     = iftRandomInteger(0, label_img_set->n - 1);
        most_sim_img_path = iftCopyString(label_img_set->files[i]->path);
    }
    else {
        char *tmp_dir = iftMakeTempDir("tmpdir_", NULL, NULL);

        puts("- Centralizing Labeled Image");
        iftFileSet *centralized_label_img_set = iftCentralizeLabelImages(label_img_set, tmp_dir);

        iftImage *label_img = iftReadImageByExt(centralized_label_img_set->files[0]->path);
        int n_objs = iftMaximumValue(label_img);
        iftDestroyImage(&label_img);

        iftMatrix *score_matrix = NULL;
        iftMetricType metric_type = IFT_SIMILARITY_FUNCTION;
        printf("n_objs = %d\n", n_objs);

        if (iftCompareStrings(metric, "DICE")) {
            puts("- Computing the Score Matrix by DICE");
            score_matrix = iftComputeScoreMatrixByDice(centralized_label_img_set, n_objs);
            metric_type = IFT_SIMILARITY_FUNCTION;
        }
        else if (iftCompareStrings(metric, "ASSD")) {
            puts("- Computing the Score Matrix by ASSD");
            score_matrix = iftComputeScoreMatrixByASSD(centralized_label_img_set, n_objs);
            metric_type = IFT_DISTANCE_FUNCTION;
        }

        puts("- Computing the Most Similar Image");
        int idx = iftBestSampleScoreSum(score_matrix, metric_type, NULL);
        most_sim_img_path = iftCopyString(label_img_set->files[idx]->path);


        iftDestroyFileSet(&centralized_label_img_set);
        iftFree(tmp_dir);
        iftRemoveDir(tmp_dir);
        iftDestroyMatrix(&score_matrix);
    }

    printf("\n--> Most Similar Image: %s\n\n", most_sim_img_path);

    // Save the Json
    iftDict *json = iftCreateDict();
    iftInsertIntoDict("reference-image", most_sim_img_path, json);
    iftInsertIntoDict("metric", metric, json);
    iftWriteJson(json, out_json_path);
    iftDestroyDict(&json);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&label_img_set);
    iftFree(metric);
    iftFree(out_json_path);
    iftFree(most_sim_img_path);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Finds the Reference Image (Template) from a set of labeled images.\n" \
        "- All object voxels, i.e. voxels with values different of 0, are considered.\n" \
        "- The Reference Image is the most similar one to all other, according to a given similarity/distance function.\n" \
        "- All labeled images are centralized by the geometric center from of the union of all objects.\n" \
        "- There is also the option to choose a RANDOM Reference Image.\n" \
        "- An output Json File stores the resulting Reference Image pathname and the used Image Metric.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--label-image-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory of the Label Images or a CSV file with their labeled images"},
        {.short_name = "-m", .long_name = "--metric", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Image Metric Function: [RANDOM, ASSD, DICE]"},
        {.short_name = "-o", .long_name = "--output-json-pathname", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Json file which will used to store the Reference Image Pathname"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **label_img_set, char **metric, char **out_json_path) {
    const char *label_img_entry = iftGetConstStrValFromDict("--label-image-entry", args);
    *label_img_set              = iftLoadFileSetFromDirOrCSV(label_img_entry, 0, true);
    *metric                     = iftUpperString(iftGetConstStrValFromDict("--metric", args));
    *out_json_path              = iftGetStrValFromDict("--output-json-pathname", args);

    if ((!iftCompareStrings(*metric, "RANDOM")) && (!iftCompareStrings(*metric, "DICE")) &&
        (!iftCompareStrings(*metric, "ASSD")))
        iftError("Invalid Metric: %s", "iftGetRequiredArgs", *metric);

    if (!iftEndsWith(*out_json_path, ".json"))
        iftError("Output Json File is not a .json: %s", "iftGetRequiredArgs", iftFileExt(*out_json_path));

    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Labeled Image Entry: %s\n", label_img_entry);
    printf("- Metric: %s\n", *metric);
    printf("- Output Json File: %s\n", *out_json_path);
    puts("-----------------------\n");
}
/*************************************************************/


