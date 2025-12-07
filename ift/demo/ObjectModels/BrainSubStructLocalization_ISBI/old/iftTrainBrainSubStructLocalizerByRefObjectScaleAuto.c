#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, iftFileSet **ref_obj_set,
                        char **ref_img_target_bb_path, char **out_json_path);
iftVoxel *iftFindMeanDispVector(  iftFileSet *target_bb_set,   iftFileSet *ref_obj_set,
                                  iftIntArray *labels, iftVoxel **enlarging_factors);
void iftWriteLocalizer(  iftImageDomain *bb_domains,   iftVoxel *disp_vecs,   iftIntArray *labels,
                      const char *out_json_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *target_bb_set    = NULL;
    iftFileSet *ref_obj_set        = NULL; // dict with the reference image paths indexed by their filenames
    char *ref_img_target_bb_path = NULL;
    char *out_json_path          = NULL;

    iftGetRequiredArgs(args, &target_bb_set, &ref_obj_set, &ref_img_target_bb_path, &out_json_path);

    timer *t1 = iftTic();

    puts("- Reading Target Object Bounding Boxes");
    iftIntArray *labels               = NULL;
    iftBoundingBox *ref_img_target_bb = iftReadLabelBoundingBox(ref_img_target_bb_path, &labels);

    puts("- Finding the Mean Displacement Vector");
    iftVoxel *enlarging_factors;
    iftVoxel *mean_disps = iftFindMeanDispVector(target_bb_set, ref_obj_set, labels, &enlarging_factors);
    
    puts("- Finding the Output Bounding Box Sizes");
    iftImageDomain *bb_domains = iftAlloc(labels->n, sizeof(iftImageDomain));
    for (int o = 0; o < labels->n; o++) {
        bb_domains[o].xsize = ref_img_target_bb[o].end.x - ref_img_target_bb[o].begin.x + 1 + enlarging_factors[o].x;
        bb_domains[o].ysize = ref_img_target_bb[o].end.y - ref_img_target_bb[o].begin.y + 1 + enlarging_factors[o].y;
        bb_domains[o].zsize = ref_img_target_bb[o].end.z - ref_img_target_bb[o].begin.z + 1 + enlarging_factors[o].z;

        printf("Object: %d\n", labels->val[o]);
        printf("Output BB Sizes: (%d, %d, %d)\n\n", bb_domains[o].xsize, bb_domains[o].ysize, bb_domains[o].zsize);
    }
    iftFree(ref_img_target_bb);

    puts("\n- Saving the Output Localizer (Json)");
    iftWriteLocalizer(bb_domains, mean_disps, labels, out_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&target_bb_set);
    iftDestroyFileSet(&ref_obj_set);
    iftFree(ref_img_target_bb_path);
    iftFree(out_json_path);
    iftDestroyIntArray(&labels);
    iftFree(mean_disps);
    iftFree(bb_domains);
    iftFree(enlarging_factors);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program builds a brain substructure localizer from a set of Labeled Bounding Boxes " \
        "of the target objects of interest (e.g. hippocampus), and their registered Label Images with " \
        "Reference Object (e.g. cerebellum).\n" \
        "- The size of the Regions of Interest (Bounding Box) for each object will be the size of the " \
        "bounding boxes from the Reference Image (Standard Space), which is also provided.\n" \
        "- Note that those Bounding Boxes will be enlarged by adding factors in each coordinate.\n" \
        "- PS1: The provided Bounding Boxes can be already scaled and translated, in order to simulate a user interaction.\n" \
        "- PS2: The pairs of Labeled Bounding Box Json File and the Reference Object Image must have the same filename.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--target-object-bb-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Labeled Bounding Boxes of the Target objects of interest."},
        {.short_name = "-r", .long_name = "--registered-reference-object-imgs-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Training. Labeled Images with the Reference Object."},
        {.short_name = "-p", .long_name = "--reference-image-bb", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Json Path with the Labeled Bounding Boxes from the Reference Image (Standard Space)."},
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


void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, iftFileSet **ref_obj_set,
                        char **ref_img_target_bb_path, char **out_json_path) {
    const char *target_bb_entry = iftGetConstStrValFromDict("--target-object-bb-entry", args);
    const char *ref_obj_entry   = iftGetConstStrValFromDict("--registered-reference-object-imgs-entry", args);
    *ref_img_target_bb_path     = iftGetStrValFromDict("--reference-image-bb", args);
    *out_json_path              = iftGetStrValFromDict("--output-localizer", args);


    *target_bb_set = iftLoadFileSetFromDirOrCSV(target_bb_entry, 0, true);

    *ref_obj_set = iftLoadFileSetFromDirOrCSV(ref_obj_entry, 0, true);

    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Target Object Bounding Box Entry: \"%s\"\n", target_bb_entry);
    printf("- Reference Object Images Entry: \"%s\"\n", ref_obj_entry);
    printf("- Bounding Box File from the Reference Images: \"%s\"\n", *ref_img_target_bb_path);
    printf("- Output (json) Localizer: \"%s\"\n", *out_json_path);
    puts("-----------------------");
}


iftVoxel *iftFindMeanDispVector(  iftFileSet *target_bb_set,   iftFileSet *ref_obj_set,
                                  iftIntArray *labels, iftVoxel **enlarging_factors) {
    iftVoxel *mean_disps  = iftAlloc(labels->n, sizeof(iftVoxel)); // all values are zero

    // "search regions", one for each target object, which contain all centers from the target object bounding boxes
    iftBoundingBox *search_regions = iftAlloc(labels->n, sizeof(iftBoundingBox));
    for (int o = 0; o < labels->n; o++) {
        search_regions[o].begin.x = search_regions[o].begin.y = search_regions[o].begin.z = IFT_INFINITY_INT;
        search_regions[o].end.x   = search_regions[o].end.y   = search_regions[o].end.z   = -1;
    }


    // finds the search region
    for (int i = 0; i < target_bb_set->n; i++) {
        const char *target_bb_path = target_bb_set->files[i]->path;
        iftBoundingBox *target_bbs = iftReadLabelBoundingBox(target_bb_path, NULL);
        
        for (int o = 0; o < labels->n; o++) {
            iftVoxel target_bb_center = iftBoundingBoxCenterVoxel(target_bbs[o]);

            if (target_bb_center.x < search_regions[o].begin.x)
                search_regions[o].begin.x = target_bb_center.x;
            if (target_bb_center.y < search_regions[o].begin.y)
                search_regions[o].begin.y = target_bb_center.y;
            if (target_bb_center.z < search_regions[o].begin.z)
                search_regions[o].begin.z = target_bb_center.z;
            if (target_bb_center.x > search_regions[o].end.x)
                search_regions[o].end.x = target_bb_center.x;
            if (target_bb_center.y > search_regions[o].end.y)
                search_regions[o].end.y = target_bb_center.y;
            if (target_bb_center.z > search_regions[o].end.z)
                search_regions[o].end.z = target_bb_center.z;
        }

        iftFree(target_bbs);
    }

    // finds the mean disp vector to the center of the search region
    for (int i = 0; i < ref_obj_set->n; i++) {
        const char *ref_obj_img_path = ref_obj_set->files[i]->path;
        iftImage *ref_obj_img        = iftReadImageByExt(ref_obj_img_path);
        iftBoundingBox mbb_ref_obj   = iftMinBoundingBox(ref_obj_img, NULL); // Ref. Object Image must be binary (only one object)
        iftVoxel mbb_ref_obj_center  = iftBoundingBoxCenterVoxel(mbb_ref_obj);

        for (int o = 0; o < labels->n; o++) {
            iftVoxel search_region_obj_center = iftBoundingBoxCenterVoxel(search_regions[o]);
            iftVoxel disp  = iftVectorSub(search_region_obj_center, mbb_ref_obj_center);
            mean_disps[o].x += disp.x;
            mean_disps[o].y += disp.y;
            mean_disps[o].z += disp.z;
        }

        iftDestroyImage(&ref_obj_img);
    }

    // averaging the displacement vector
    puts("\n- Resulting Bounding Box, Mean Disp Vector, and Enlarging Factors");
    *enlarging_factors = iftAlloc(labels->n, sizeof(iftVoxel));
    for (int o = 0; o < labels->n; o++) {
        mean_disps[o].x = iftRound(mean_disps[o].x / (target_bb_set->n*1.0));
        mean_disps[o].y = iftRound(mean_disps[o].y / (target_bb_set->n*1.0));
        mean_disps[o].z = iftRound(mean_disps[o].z / (target_bb_set->n*1.0));

        printf("mean_disps[%d] = (%d, %d, %d)\n", o, mean_disps[o].x, mean_disps[o].y, mean_disps[o].z);

        (*enlarging_factors)[o].x = search_regions[o].end.x - search_regions[o].begin.x + 1;
        (*enlarging_factors)[o].y = search_regions[o].end.y - search_regions[o].begin.y + 1;
        (*enlarging_factors)[o].z = search_regions[o].end.z - search_regions[o].begin.z + 1;

        printf("enlarging_factors[%d] = (%d, %d, %d)\n\n", o, (*enlarging_factors)[o].x, (*enlarging_factors)[o].y, (*enlarging_factors)[o].z);
    }

    iftFree(search_regions);

    return mean_disps;
}



void iftWriteLocalizer(  iftImageDomain *bb_domains,   iftVoxel *disp_vecs,   iftIntArray *labels,
                      const char *out_json_path) {
    iftJson *json = iftCreateJsonRoot();

    // saving labels
    iftAddIntArrayToJson(json, "labels", labels);

    iftAddJDictToJson(json, "localizer", iftCreateJDict());

    for (int o = 0; o < labels->n; o++) {
        char dict_key[128];
        sprintf(dict_key, "localizer:%d", labels->val[o]);
        iftAddJDictToJson(json, dict_key, iftCreateJDict());
        
        char key[128];
        sprintf(key, "%s:ROI-sizes", dict_key);
        iftIntArray *ROI_sizes = iftCreateIntArray(3);
        ROI_sizes->val[0] = bb_domains[o].xsize;
        ROI_sizes->val[1] = bb_domains[o].ysize;
        ROI_sizes->val[2] = bb_domains[o].zsize;
        iftAddIntArrayToJson(json, key, ROI_sizes);
        iftDestroyIntArray(&ROI_sizes);

        sprintf(key, "%s:displacement-vector", dict_key);
        iftIntArray *disp_vecs_arr = iftCreateIntArray(3);
        disp_vecs_arr->val[0] = disp_vecs[o].x;
        disp_vecs_arr->val[1] = disp_vecs[o].y;
        disp_vecs_arr->val[2] = disp_vecs[o].z;
        iftAddIntArrayToJson(json, key, disp_vecs_arr);
        iftDestroyIntArray(&disp_vecs_arr);
    }

    iftWriteJson(json, out_json_path);
    iftDestroyJson(&json);
}
/*************************************************************/


