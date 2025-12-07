#include "ift.h"

#include "iftReadWriteBBSizes.c" // file with the function iftReadBBSizes and iftWriteBBSizes


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, char **out_json_path);
iftLabelBoundingBoxArray *iftFindLargestBBs(  iftFileSet *target_bb_set);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *target_bb_set    = NULL;
    char *out_json_path          = NULL;

    iftGetRequiredArgs(args, &target_bb_set, &out_json_path);


    timer *t1 = iftTic();
    iftLabelBoundingBoxArray *lbbs = iftFindLargestBBs(target_bb_set);

    iftIntMatrix *size_mat = iftCreateIntMatrix(3, lbbs->labels->n);

    for (int o = 0; o < lbbs->labels->n; o++) {
        iftIntArray *sizes = iftBoundingBoxSize(lbbs->bbs[o]);
        iftMatrixElem(size_mat, 0, o) = sizes->val[0];
        iftMatrixElem(size_mat, 1, o) = sizes->val[1];
        iftMatrixElem(size_mat, 2, o) = sizes->val[2];
        iftDestroyIntArray(&sizes);
    }

    iftWriteBBSizes(size_mat, lbbs->labels, out_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyLabelBoundingBoxArray(&lbbs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find the largest label bounding boxes from a set of ones.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--target-object-bb-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Labeled Bounding Boxes of the Target objects of interest."},
        {.short_name = "-o", .long_name = "--output-bb-size-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output json file that stores the largest label bb."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, char **out_json_path) {
    const char *target_bb_entry = iftGetConstStrValFromDict("--target-object-bb-entry", args);
    *out_json_path              = iftGetStrValFromDict("--output-bb-size-json", args);

    *target_bb_set = iftLoadFileSetFromDirOrCSV(target_bb_entry, 0, true);

    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Target Object Bounding Box Entry: \"%s\"\n", target_bb_entry);
    printf("- Output (json) Size BB: \"%s\"\n", *out_json_path);
    puts("-----------------------\n");
}


iftLabelBoundingBoxArray *iftFindLargestBBs(  iftFileSet *target_bb_set) {
    iftLabelBoundingBoxArray *lbbs         = iftReadLabelBoundingBoxArray(target_bb_set->files[0]->path);
    iftLabelBoundingBoxArray *largest_lbbs = iftCreateLabelBoundingBoxArray(lbbs->labels->n);
    iftCopyIntArray(largest_lbbs->labels->val, lbbs->labels->val, lbbs->labels->n);
    iftDestroyLabelBoundingBoxArray(&lbbs);
    
    iftIntArray *n_largest_bbs_voxels = iftCreateIntArray(largest_lbbs->labels->n); // all values are 0

    for (int i = 0; i < target_bb_set->n; i++) {
        lbbs = iftReadLabelBoundingBoxArray(target_bb_set->files[i]->path);

        for (int o = 0; o < lbbs->labels->n; o++) {
            int n_bb_voxels    = iftBoundingBoxBoxVolume(lbbs->bbs[o]);

            // swap the current largest bbs for a largest one
            if (n_bb_voxels > n_largest_bbs_voxels->val[o]) {
                printf("object: %d\n", lbbs->labels->val[o]);
                printf("n_bb_voxels = %d > %d\n", n_bb_voxels, n_largest_bbs_voxels->val[o]);
                puts("current");
                iftPrintBoundingBox(largest_lbbs->bbs[o]);
                puts("new");
                iftPrintBoundingBox(lbbs->bbs[o]);
                printf("%s\n\n", target_bb_set->files[i]->path);

                n_largest_bbs_voxels->val[o] = n_bb_voxels;
                largest_lbbs->bbs[o].begin.x = lbbs->bbs[o].begin.x;
                largest_lbbs->bbs[o].begin.y = lbbs->bbs[o].begin.y;
                largest_lbbs->bbs[o].begin.z = lbbs->bbs[o].begin.z;
                largest_lbbs->bbs[o].end.x   = lbbs->bbs[o].end.x;
                largest_lbbs->bbs[o].end.y   = lbbs->bbs[o].end.y;
                largest_lbbs->bbs[o].end.z   = lbbs->bbs[o].end.z;
            }
        }
        iftDestroyLabelBoundingBoxArray(&lbbs);
    }
    iftDestroyIntArray(&n_largest_bbs_voxels);

    return largest_lbbs;
}
/*************************************************************/


