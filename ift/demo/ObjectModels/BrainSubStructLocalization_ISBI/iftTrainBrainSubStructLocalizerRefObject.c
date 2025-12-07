#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, iftDict **ref_obj_dict, char **out_json_path);
void iftWriteLocalizer(  iftImageDomain *bb_domains,   iftVoxel *disp_vecs,   iftIntArray *labels,
                      const char *out_json_path);
iftLabelBoundingBoxArray *iftFindLargestBBs(  iftFileSet *target_bb_set);
iftBoundingBox *iftFindSearchRegion(  iftFileSet *target_bb_set,   iftDict *ref_obj_dict,
                                      iftIntArray *labels, iftVoxel ref_voxel, iftVoxel **out_disp_vecs);
void iftWriteLocalizerRefObject(  iftLabelBoundingBoxArray *lbbs,   iftVoxel *disp_vecs,
                                  iftBoundingBox *search_regions, const char *out_json_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *target_bb_set = NULL;
    iftDict *ref_obj_dict     = NULL; // dict with the reference image paths indexed by their filenames
    char *out_json_path       = NULL;

    iftGetRequiredArgs(args, &target_bb_set, &ref_obj_dict, &out_json_path);

    timer *t1 = iftTic();

    puts("Finding Largest BB");
    iftLabelBoundingBoxArray *lbbs = iftFindLargestBBs(target_bb_set);

    puts("Finding Search Region");
    // We choose any image just to use its reference object to find the search region and the displacement vector
    char *ref_obj_img_path = ref_obj_dict->table[ref_obj_dict->firstKey].val.str_val;
    printf("ref_obj_img_path = %s\n", ref_obj_img_path);
    iftImage *ref_obj_img = iftReadImageByExt(ref_obj_img_path);

    iftVoxel *disp_vecs = NULL;
    iftBoundingBox *search_regions = iftFindSearchRegion(target_bb_set, ref_obj_dict, lbbs->labels,
                                            iftBoundingBoxCenterVoxel(iftMinBoundingBox(ref_obj_img, NULL)),
                                            &disp_vecs);

    puts("Writing Localizer");
    iftWriteLocalizerRefObject(lbbs, disp_vecs, search_regions, out_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&target_bb_set);
    iftFree(out_json_path);
    iftFree(lbbs);
    iftDestroyImage(&ref_obj_img);
    iftFree(search_regions);


    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program builds a brain substructure localizer from a set of labeled bounding boxes " \
        "of the target objects of interest (e.g. hippocampus), and their registered Label Images with " \
        "Reference Object (e.g. cerebellum).\n" \
        "- The size of the Regions of Interest (Bounding Box) for each object will be the size of the " \
        "largest bounding boxes from the Training Set.\n" \
        "- Note that those Bounding Boxes can be already scaled.\n" \
        "- Displacement vectors, between the reference object center and the target object, is figured out for "\
        "each image.\n" \
        "- Then, we apply all displacement vector from the reference object's center of an arbitrary image, " \
        "which will provide the estimated bounding box center for the target object.\n" \
        "- The search region then corresponds to the min bounding box which contains all estimated centers, " \
        "and the resulting displacement vector is the one from the center of the considered reference image to the center " \
        "of the search region.\n\n" \
        "- PS1: The provided Bounding Boxes can be already scaled and translated, in order to simulate a user interaction.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--target-object-bb-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Labeled Bounding Boxes of the Target objects of interest."},
        {.short_name = "-r", .long_name = "--registered-reference-object-imgs-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Training. Labeled Images with the Reference Object."},
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


void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, iftDict **ref_obj_dict, char **out_json_path) {
    const char *target_bb_entry = iftGetConstStrValFromDict("--target-object-bb-entry", args);
    const char *ref_obj_entry   = iftGetConstStrValFromDict("--registered-reference-object-imgs-entry", args);

    *out_json_path              = iftGetStrValFromDict("--output-localizer", args);

    *target_bb_set = iftLoadFileSetFromDirOrCSV(target_bb_entry, 0, true);
    iftFileSet *ref_obj_set = iftLoadFileSetFromDirOrCSV(ref_obj_entry, 0, true);
    *ref_obj_dict           = iftFileSetToDict(ref_obj_set);
    iftDestroyFileSet(&ref_obj_set);


    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Target Object Bounding Box Entry: \"%s\"\n", target_bb_entry);
    printf("- Reference Object Images Entry: \"%s\"\n", ref_obj_entry);
    printf("- Output (json) Localizer: \"%s\"\n", *out_json_path);
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



iftBoundingBox *iftFindSearchRegion(  iftFileSet *target_bb_set,   iftDict *ref_obj_dict,
                                      iftIntArray *labels, iftVoxel ref_voxel, iftVoxel **out_disp_vecs) {
    iftBoundingBox *search_regions = iftAlloc(labels->n, sizeof(iftBoundingBox));

    for (int o = 0; o < labels->n; o++) {
        search_regions[o].begin.x = search_regions[o].begin.y = search_regions[o].begin.z = IFT_INFINITY_INT;
        search_regions[o].end.x   = search_regions[o].end.y   = search_regions[o].end.z   = -1;
    }

    for (int i = 0; i < target_bb_set->n; i++) {
        const char *target_img_path = target_bb_set->files[i]->path;
        char *fkey                  = iftFilename(target_img_path, iftFileExt(target_img_path));
        char *ref_obj_path          = iftGetStrValFromDict(fkey, ref_obj_dict);
        iftFree(fkey);
        printf("[%d]\n", i);
        puts(target_img_path);
        puts(ref_obj_path);
        // printf("target_img_path: %s\n", target_img_path);
        // printf("ref_obj_path: %s\n", ref_obj_path);

        iftImage *ref_obj_img          = iftReadImageByExt(ref_obj_path);
        iftVoxel ref_obj_bb_center     = iftBoundingBoxCenterVoxel(iftMinObjectBoundingBox(ref_obj_img, 1, NULL));
        iftLabelBoundingBoxArray *lbbs = iftReadLabelBoundingBoxArray(target_bb_set->files[i]->path);

        for (int o = 0; o < labels->n; o++) {
            iftVoxel target_bb_center = iftBoundingBoxCenterVoxel(lbbs->bbs[o]);
            iftVoxel disp = iftVectorSub(target_bb_center, ref_obj_bb_center);

            iftVoxel target_final_center = iftVectorSum(ref_voxel, disp);

            if (target_final_center.x < search_regions[o].begin.x)
                search_regions[o].begin.x = target_final_center.x;
            if (target_final_center.y < search_regions[o].begin.y)
                search_regions[o].begin.y = target_final_center.y;
            if (target_final_center.z < search_regions[o].begin.z)
                search_regions[o].begin.z = target_final_center.z;
            if (target_final_center.x > search_regions[o].end.x)
                search_regions[o].end.x = target_final_center.x;
            if (target_final_center.y > search_regions[o].end.y)
                search_regions[o].end.y = target_final_center.y;
            if (target_final_center.z > search_regions[o].end.z)
                search_regions[o].end.z = target_final_center.z;
        }

        iftDestroyImage(&ref_obj_img);
        iftFree(ref_obj_path);
        iftDestroyLabelBoundingBoxArray(&lbbs);
    }


    iftVoxel *disp_vecs = iftAlloc(labels->n, sizeof(iftVoxel));
    for (int o = 0; o < labels->n; o++) {
        iftVoxel search_region_center = iftBoundingBoxCenterVoxel(search_regions[o]);
        iftVoxel disp                  = iftVectorSub(search_region_center, ref_voxel);
        disp_vecs[o] = disp;
        
        printf("Object: %d\n", labels->val[o]);
        puts("search_regions");
        iftPrintBoundingBox(search_regions[o]);
        printf("disp: (%d, %d, %d)\n", disp.x, disp.y, disp.z);
    }

    *out_disp_vecs = disp_vecs;

    return search_regions;
}


void iftWriteLocalizerRefObject(  iftLabelBoundingBoxArray *lbbs,   iftVoxel *disp_vecs,
                                  iftBoundingBox *search_regions, const char *out_json_path) {
    iftDict *json        = iftCreateDict();
    json->erase_elements = true;

    iftIntArray *labels = iftCreateIntArray(lbbs->labels->n);
    iftCopyIntArray(labels->val, lbbs->labels->val, lbbs->labels->n);
    iftInsertIntoDict("labels", labels, json);

    for (int o = 0; o < labels->n; o++) {
        char key[512];

        sprintf(key, "data:%d:bounding-box-size", labels->val[o]);
        iftIntArray *bb_size = iftBoundingBoxSize(lbbs->bbs[o]);
        iftInsertIntoDict(key, bb_size, json);

        sprintf(key, "data:%d:disp-vector", labels->val[o]);
        iftIntArray *disp_vec_arr = iftCreateIntArray(3);
        disp_vec_arr->val[0] = disp_vecs[o].x;
        disp_vec_arr->val[1] = disp_vecs[o].y;
        disp_vec_arr->val[2] = disp_vecs[o].z;
        iftInsertIntoDict(key, disp_vec_arr, json);

        sprintf(key, "data:%d:search-region-size", labels->val[o]);
        iftIntArray *search_region_sizes = iftBoundingBoxSize(search_regions[o]);
        iftInsertIntoDict(key, search_region_sizes, json);
    }
    iftWriteJson(json, out_json_path);

    iftDestroyDict(&json);
}









