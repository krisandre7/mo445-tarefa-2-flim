#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, iftDict **ref_obj_dict,
                        char **ref_img_target_bb_path, char **out_json_path);
iftVoxel *iftFindMeanDispVector(  iftFileSet *target_bb_set,   iftDict *ref_obj_dict, iftIntArray *labels);
void iftWriteLocalizer(  iftImageDomain *bb_domains,   iftVoxel *disp_vecs,   iftIntArray *labels,
                      const char *out_json_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *target_bb_set    = NULL;
    iftDict *ref_obj_dict        = NULL; // dict with the reference image paths indexed by their filenames
    char *ref_img_target_bb_path = NULL;
    char *out_json_path          = NULL;

    iftGetRequiredArgs(args, &target_bb_set, &ref_obj_dict, &ref_img_target_bb_path, &out_json_path);

    timer *t1 = iftTic();

    puts("- Finding the Mean Displacement Vector");
    iftIntArray *labels               = NULL;
    iftBoundingBox *ref_img_target_bb = iftReadLabelBoundingBox(ref_img_target_bb_path, &labels);
    iftImageDomain *bb_domains = iftAlloc(labels->n, sizeof(iftImageDomain));
    for (int o = 0; o < labels->n; o++) {
        bb_domains[o].xsize = ref_img_target_bb[o].end.x - ref_img_target_bb[o].begin.x + 1;
        bb_domains[o].ysize = ref_img_target_bb[o].end.y - ref_img_target_bb[o].begin.y + 1;
        bb_domains[o].zsize = ref_img_target_bb[o].end.z - ref_img_target_bb[o].begin.z + 1;

        printf("Object: %d\n", labels->val[o]);
        printf("Output BB Sizes: (%d, %d, %d)\n\n", bb_domains[o].xsize, bb_domains[o].ysize, bb_domains[o].zsize);
    }


    puts("- Finding the Output Bounding Box Sizes");
    iftVoxel *mean_disps = iftFindMeanDispVector(target_bb_set, ref_obj_dict, labels);
    

    puts("\n- Saving the Output Localizer (Json)");
    iftWriteLocalizer(bb_domains, mean_disps, labels, out_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&target_bb_set);
    iftDestroyDict(&ref_obj_dict);
    iftFree(ref_img_target_bb_path);
    iftFree(out_json_path);
    iftDestroyIntArray(&labels);
    iftFree(mean_disps);
    iftFree(bb_domains);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program builds a brain substructure localizer from a set of Labeled Bounding Boxes " \
        "of the target objects of interest (e.g. hippocampus), and their registered Label Images with " \
        "Reference Object (e.g. cerebellum).\n" \
        "- The size of the Regions of Interest (Bounding Box) for each object will be the size of the " \
        "bounding boxes from the Reference Image (Standard Space), which is also provided." \
        "Note that those Bounding Boxes can be already scaled.\n\n"
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


void iftGetRequiredArgs(  iftDict *args, iftFileSet **target_bb_set, iftDict **ref_obj_dict,
                        char **ref_img_target_bb_path, char **out_json_path) {
    const char *target_bb_entry = iftGetConstStrValFromDict("--target-object-bb-entry", args);
    const char *ref_obj_entry   = iftGetConstStrValFromDict("--registered-reference-object-imgs-entry", args);
    *ref_img_target_bb_path     = iftGetStrValFromDict("--reference-image-bb", args);
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
    printf("- Bounding Box File from the Reference Images: \"%s\"\n", *ref_img_target_bb_path);
    printf("- Output (json) Localizer: \"%s\"\n", *out_json_path);
    puts("-----------------------");
}


iftVoxel *iftFindMeanDispVector(  iftFileSet *target_bb_set,   iftDict *ref_obj_dict, iftIntArray *labels) {
    iftVoxel *mean_disps  = iftAlloc(labels->n, sizeof(iftVoxel)); // all values are zero


    for (int i = 0; i < target_bb_set->n; i++) {
        const char *target_bb_path   = target_bb_set->files[i]->path;
        char *fkey                   = iftFilename(target_bb_path, iftFileExt(target_bb_path));
        const char *ref_obj_img_path = iftGetConstStrValFromDict(fkey, ref_obj_dict);
        iftFree(fkey);
        printf("\n[%d]\ntarget: %s\nreference obj: %s\n", i, target_bb_path, ref_obj_img_path);

        iftBoundingBox *target_bbs = iftReadLabelBoundingBox(target_bb_path, NULL);
        iftImage *ref_obj_img      = iftReadImageByExt(ref_obj_img_path);
        iftBoundingBox mbb_ref_obj = iftMinBoundingBox(ref_obj_img, NULL); // Ref. Object Image must be binary (only one object)
        iftVoxel mbb_ref_obj_center = iftBoundingBoxCenterVoxel(mbb_ref_obj);

        // for each object
        for (int o = 0; o < labels->n; o++) {
            iftVoxel target_bb_center = iftBoundingBoxCenterVoxel(target_bbs[o]);
            iftVoxel disp  = iftVectorSub(target_bb_center, mbb_ref_obj_center);
            mean_disps[o].x += disp.x;
            mean_disps[o].y += disp.y;
            mean_disps[o].z += disp.z;

            printf("target_bbs[%d].begin: (%d, %d, %d)\n", o, target_bbs[o].begin.x, target_bbs[o].begin.y, target_bbs[o].begin.z);
            printf("target_bbs[%d].end: (%d, %d, %d)\n", o, target_bbs[o].end.x, target_bbs[o].end.y, target_bbs[o].end.z);
            printf("mbb_ref_obj.begin: (%d, %d, %d)\n", mbb_ref_obj.begin.x, mbb_ref_obj.begin.y, mbb_ref_obj.begin.z);
            printf("mbb_ref_obj.end: (%d, %d, %d)\n", mbb_ref_obj.end.x, mbb_ref_obj.end.y, mbb_ref_obj.end.z);
            printf("disp = (%d, %d, %d)\n", disp.x, disp.y, disp.z);
            printf("mean_disps[%d] = (%d, %d, %d)\n\n", o, mean_disps[o].x, mean_disps[o].y, mean_disps[o].z);
        }

        iftDestroyImage(&ref_obj_img);
        iftFree(target_bbs);
    }

    // averaging the displacement vector
    puts("\n- Resulting Bounding Box and Mean Disp Vector");
    for (int o = 0; o < labels->n; o++) {
        mean_disps[o].x = iftRound(mean_disps[o].x / (target_bb_set->n*1.0));
        mean_disps[o].y = iftRound(mean_disps[o].y / (target_bb_set->n*1.0));
        mean_disps[o].z = iftRound(mean_disps[o].z / (target_bb_set->n*1.0));

        printf("mean_disps[%d] = (%d, %d, %d)\n\n", o, mean_disps[o].x, mean_disps[o].y, mean_disps[o].z);
    }

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


