#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **ref_bb_json_path, char **model_path, char **out_bbs_path);
iftBoundingBox *iftFindBrainSubStructsByProbAtlas(  iftBoundingBox *ref_bbs,   iftIntArray *ref_bb_labels,
                                                    iftMObjModel *model);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *ref_bb_json_path = NULL;
    char *model_path       = NULL;
    char *out_bbs_path     = NULL;

    iftGetRequiredArgs(args, &ref_bb_json_path, &model_path, &out_bbs_path);

    timer *t1 = iftTic();

    puts("- Reading Reference Image's Bounding Boxes");
    iftIntArray *ref_bb_labels = NULL;
    iftBoundingBox *ref_bbs    = iftReadLabelBoundingBox(ref_bb_json_path, &ref_bb_labels);

    puts("- Reading Object Model");
    iftMObjModel *model = iftReadSOSM(model_path);

    puts("- Finding the Output ROIs");
    iftBoundingBox *out_bbs = iftFindBrainSubStructsByProbAtlas(ref_bbs, ref_bb_labels, model);

    puts("- Writing the Output ROIs");
    iftWriteLabelBoundingBox(out_bbs, ref_bb_labels, out_bbs_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(ref_bb_json_path);
    iftFree(model_path);
    iftFree(out_bbs_path);
    iftFree(ref_bbs);
    iftDestroyIntArray(&ref_bb_labels);
    iftDestroyMObjModel(&model);
    iftFree(out_bbs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find Brain Substructures by placing the ROIs (Bounding Boxes drawn by user) from the center of the probabilistic " \
        "atlas.\n" \
        "PS: Both ROIs and Prob. Atlas must have the same objects.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-r", .long_name = "--reference-image-bb", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Json Path with the Labeled Bounding Boxes from the Reference Image (Standard Space)."},
        {.short_name = "-m", .long_name = "--model", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Object Shape Model used of the Target Objects (*.zip)."},
        {.short_name = "-o", .long_name = "--output-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Label Image with the ROIs of the Brain Substructues."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **ref_bb_json_path, char **model_path, char **out_bbs_path) {
    *ref_bb_json_path = iftGetStrValFromDict("--reference-image-bb", args);
    *model_path       = iftGetStrValFromDict("--model", args);
    *out_bbs_path     = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_bbs_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Bounding Box File from the Reference Images: \"%s\"\n", *ref_bb_json_path);
    printf("- Object Shape Model: \"%s\"\n", *model_path);
    printf("- Output ROIs Json: \"%s\"\n", *out_bbs_path);
    puts("-----------------------\n");
}


iftBoundingBox *iftFindBrainSubStructsByProbAtlas(  iftBoundingBox *ref_bbs,   iftIntArray *ref_bb_labels,
                                                    iftMObjModel *model) {
    iftBoundingBox *out_bbs    = iftAlloc(ref_bb_labels->n, sizeof(iftBoundingBox));
    iftDict *model_labels_dict = iftIntArrayToDict(ref_bb_labels);
    
    iftVoxel ref_voxel = model->finder->func(model->ref_img, model->finder->params);
    
    for (int o = 0; o < ref_bb_labels->n; o++) {
        int obj_idx = iftGetLongValFromDict(ref_bb_labels->val[o], model_labels_dict);
        printf("Object: %d\n", ref_bb_labels->val[o]);
        printf("Object Index Model: %d\n", obj_idx);

        // to obtain the model center on reference image space, we move the reference voxel from the reference image
        // by using the displacement vector from the prob. atlas
        // Remember that the prob. atlas has its own domain, it is a crop, ie., it is not on the reference domain
        iftVoxel model_center_on_ref_space = iftVectorSum(ref_voxel, model->models[obj_idx]->disp_vec);


        iftVoxel bb_center = iftBoundingBoxCenterVoxel(ref_bbs[o]);
        // centralize the bounding box on model center
        iftVoxel disp = iftVectorSub(model_center_on_ref_space, bb_center);
        out_bbs[o].begin.x = ref_bbs[o].begin.x + disp.x;
        out_bbs[o].begin.y = ref_bbs[o].begin.y + disp.y;
        out_bbs[o].begin.z = ref_bbs[o].begin.z + disp.z;
        out_bbs[o].end.x = ref_bbs[o].end.x + disp.x;
        out_bbs[o].end.y = ref_bbs[o].end.y + disp.y;
        out_bbs[o].end.z = ref_bbs[o].end.z + disp.z;
    }

    iftDestroyDict(&model_labels_dict);

    return out_bbs;
}
/*************************************************************/


















