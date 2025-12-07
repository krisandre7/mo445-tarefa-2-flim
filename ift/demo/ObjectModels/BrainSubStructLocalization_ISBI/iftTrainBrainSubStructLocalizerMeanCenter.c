#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, char **out_json_path);
void iftWriteLocalizer(  iftImageDomain *bb_domains,   iftVoxel *disp_vecs,   iftIntArray *labels,
                      const char *out_json_path);
iftLabelBoundingBoxArray *iftFindLargestBBsMeanCenters(  iftFileSet *target_bb_set);
iftBoundingBox *iftFindSearchRegions(  iftFileSet *target_bb_set,   iftIntArray *labels);
void iftWriteLocalizerMeanCenter(  iftLabelBoundingBoxArray *lbbs, iftBoundingBox *search_regions, const char *out_json_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *target_bb_set    = NULL;
    char *out_json_path          = NULL;

    iftGetRequiredArgs(args, &target_bb_set, &out_json_path);

    timer *t1 = iftTic();

    iftIntArray *labels            = NULL;
    iftLabelBoundingBoxArray *lbbs = iftFindLargestBBsMeanCenters(target_bb_set);
    iftBoundingBox *search_regions = iftFindSearchRegions(target_bb_set, lbbs->labels);
    iftWriteLocalizerMeanCenter(lbbs, search_regions, out_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&target_bb_set);
    iftFree(out_json_path);
    iftDestroyIntArray(&labels);
    iftFree(lbbs);
    iftFree(search_regions);


    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program builds a brain substructure localizer from a set of Labeled Bounding Boxes " \
        "of the target objects of interest (e.g. hippocampus).\n" \
        "- The size of the Regions of Interest (Bounding Box) for each object will be the size of the " \
        "largest bounding boxes from the Training Set.\n" \
        "- Note that those Bounding Boxes can be already scaled.\n" \
        "- The position of the final Regions of Interest will be the mean center from all bounding boxes.\n" \
        "- It computes a search region for the bounding box localization by finding the box which contains " \
        "all Target Object Bounding Box Centers.\n" \
        "- PS1: The provided Bounding Boxes can be already scaled and translated, in order to simulate a user interaction.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--target-object-bb-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Labeled Bounding Boxes of the Target objects of interest."},
        {.short_name = "-o", .long_name = "--output-localizer", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output json file that stores the localizer information."},
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
    *out_json_path              = iftGetStrValFromDict("--output-localizer", args);

    *target_bb_set = iftLoadFileSetFromDirOrCSV(target_bb_entry, 0, true);

    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Target Object Bounding Box Entry: \"%s\"\n", target_bb_entry);
    printf("- Output (json) Localizer: \"%s\"\n", *out_json_path);
    puts("-----------------------\n");
}


// finds the largest bounding box from the set, for each object, and centralize them into the mean center
// from all label bounding boxes
// it suposes that all json files has the same number of labeled bounding boxes stored of the same way
iftLabelBoundingBoxArray *iftFindLargestBBsMeanCenters(  iftFileSet *target_bb_set) {
    iftLabelBoundingBoxArray *lbbs         = iftReadLabelBoundingBoxArray(target_bb_set->files[0]->path);
    iftLabelBoundingBoxArray *largest_lbbs = iftCreateLabelBoundingBoxArray(lbbs->labels->n);
    iftCopyIntArray(largest_lbbs->labels->val, lbbs->labels->val, lbbs->labels->n);
    iftDestroyLabelBoundingBoxArray(&lbbs);
    
    iftDict *labels_dict_idxs         = iftIntArrayToDict(largest_lbbs->labels->val, largest_lbbs->labels->n);
    iftIntArray *n_largest_bbs_voxels = iftCreateIntArray(largest_lbbs->labels->n); // all values are 0

    iftVoxel *mean_bbs_centers = iftAlloc(largest_lbbs->labels->n, sizeof(iftVoxel)); // all values are 0


    for (int i = 0; i < target_bb_set->n; i++) {
        lbbs = iftReadLabelBoundingBoxArray(target_bb_set->files[i]->path);

        for (int o = 0; o < lbbs->labels->n; o++) {
            int label_idx      = iftGetLongValFromDict(lbbs->labels->val[o], labels_dict_idxs);
            int n_bb_voxels    = iftBoundingBoxBoxVolume(lbbs->bbs[o]);
            iftVoxel bb_center = iftBoundingBoxCenterVoxel(lbbs->bbs[o]);
            mean_bbs_centers[label_idx].x += bb_center.x;
            mean_bbs_centers[label_idx].y += bb_center.y;
            mean_bbs_centers[label_idx].z += bb_center.z;


            // swap the current largest bbs for a largest one
            if (n_bb_voxels > n_largest_bbs_voxels->val[label_idx]) {
                // printf("n_bb_voxels = %d > %d\n", n_bb_voxels, n_largest_bbs_voxels->val[o]);
                // puts("current");
                // iftPrintBoundingBox(largest_lbbs->bbs[label_idx]);
                // puts("new");
                // iftPrintBoundingBox(lbbs->bbs[o]);
                // puts("");

                n_largest_bbs_voxels->val[o] = n_bb_voxels;
                largest_lbbs->bbs[label_idx].begin.x = lbbs->bbs[o].begin.x;
                largest_lbbs->bbs[label_idx].begin.y = lbbs->bbs[o].begin.y;
                largest_lbbs->bbs[label_idx].begin.z = lbbs->bbs[o].begin.z;
                largest_lbbs->bbs[label_idx].end.x   = lbbs->bbs[o].end.x;
                largest_lbbs->bbs[label_idx].end.y   = lbbs->bbs[o].end.y;
                largest_lbbs->bbs[label_idx].end.z   = lbbs->bbs[o].end.z;
            }
        }
        iftDestroyLabelBoundingBoxArray(&lbbs);
    }
    iftDestroyIntArray(&n_largest_bbs_voxels);
    iftDestroyDict(&labels_dict_idxs);


    for (int o = 0; o < largest_lbbs->labels->n; o++) {
        mean_bbs_centers[o].x /= iftRound(1.0 * target_bb_set->n);
        mean_bbs_centers[o].y /= iftRound(1.0 * target_bb_set->n);
        mean_bbs_centers[o].z /= iftRound(1.0 * target_bb_set->n);
        // printf("mean_bbs_centers[%d].x = %d\n", o, mean_bbs_centers[o].x);
        // printf("mean_bbs_centers[%d].y = %d\n", o, mean_bbs_centers[o].y);
        // printf("mean_bbs_centers[%d].z = %d\n", o, mean_bbs_centers[o].z);

        largest_lbbs->bbs[o] = iftCentralizeBoundingBox(largest_lbbs->bbs[o], mean_bbs_centers[o]);
    }
    iftFree(mean_bbs_centers);

    return largest_lbbs;
}



iftBoundingBox *iftFindSearchRegions(  iftFileSet *target_bb_set,   iftIntArray *labels) {
    iftDict *labels_dict_idxs = iftIntArrayToDict(labels->val, labels->n);


    iftBoundingBox *search_regions = iftAlloc(labels->n, sizeof(iftBoundingBox));
    for (int o = 0; o < labels->n; o++) {
        search_regions[o].begin.x = search_regions[o].begin.y = search_regions[o].begin.z = IFT_INFINITY_INT;
        search_regions[o].end.x   = search_regions[o].end.y   = search_regions[o].end.z   = -1;
    }


    for (int i = 0; i < target_bb_set->n; i++) {
        iftLabelBoundingBoxArray *lbbs = iftReadLabelBoundingBoxArray(target_bb_set->files[i]->path);

        for (int o = 0; o < lbbs->labels->n; o++) {
            int label_idx      = iftGetLongValFromDict(lbbs->labels->val[o], labels_dict_idxs);
            iftVoxel bb_center = iftBoundingBoxCenterVoxel(lbbs->bbs[o]);

            printf("Object: %d\n", lbbs->labels->val[o]);
            puts("search region");
            iftPrintBoundingBox(search_regions[label_idx]);
            printf("bb_center: (%d, %d, %d)\n", bb_center.x, bb_center.y, bb_center.z);
            puts("");


            if (bb_center.x < search_regions[label_idx].begin.x)
                search_regions[label_idx].begin.x = bb_center.x;
            if (bb_center.y < search_regions[label_idx].begin.y)
                search_regions[label_idx].begin.y = bb_center.y;
            if (bb_center.z < search_regions[label_idx].begin.z)
                search_regions[label_idx].begin.z = bb_center.z;

            if (bb_center.x > search_regions[label_idx].end.x)
                search_regions[label_idx].end.x = bb_center.x;
            if (bb_center.y > search_regions[label_idx].end.y)
                search_regions[label_idx].end.y = bb_center.y;
            if (bb_center.z > search_regions[label_idx].end.z)
                search_regions[label_idx].end.z = bb_center.z;
        }

        iftDestroyLabelBoundingBoxArray(&lbbs);
    }
    iftDestroyDict(&labels_dict_idxs);

    return search_regions;
}




void iftWriteLocalizerMeanCenter(  iftLabelBoundingBoxArray *lbbs, iftBoundingBox *search_regions,
                                 const char *out_json_path) {
    iftDict *json        = iftCreateDict();
    json->erase_elements = true;

    iftIntArray *labels = iftCreateIntArray(lbbs->labels->n);
    iftCopyIntArray(labels->val, lbbs->labels->val, lbbs->labels->n);
    iftInsertIntoDict("labels", labels, json);

    for (int o = 0; o < labels->n; o++) {
        char key[512];

        sprintf(key, "data:%d:bounding-box:begin", labels->val[o]);
        iftIntArray *begin = iftCreateIntArray(3);
        begin->val[0] = lbbs->bbs[o].begin.x;
        begin->val[1] = lbbs->bbs[o].begin.y;
        begin->val[2] = lbbs->bbs[o].begin.z;
        iftInsertIntoDict(key, begin, json);

        sprintf(key, "data:%d:bounding-box:end", labels->val[o]);
        iftIntArray *end = iftCreateIntArray(3);
        end->val[0] = lbbs->bbs[o].end.x;
        end->val[1] = lbbs->bbs[o].end.y;
        end->val[2] = lbbs->bbs[o].end.z;
        iftInsertIntoDict(key, end, json);

        sprintf(key, "data:%d:search-region-size", labels->val[o]);
        iftIntArray *search_region_sizes = iftBoundingBoxSize(search_regions[o]);
        iftInsertIntoDict(key, search_region_sizes, json);
    }
    iftWriteJson(json, out_json_path);

    iftDestroyDict(&json);
}
/*************************************************************/


