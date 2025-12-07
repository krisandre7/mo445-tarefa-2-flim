#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **reg_label_set, char **out_localizer_path);
void iftGetOptionalArgs(  iftDict *args, const char *label_img_path, iftIntArray **labels);
iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path);
iftImage *iftFindLocalizer(  iftFileSet *reg_label_set,   iftIntArray *labels);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory arguments
    iftFileSet *reg_label_set = NULL;
    char *out_localizer_path  = NULL;
    // optional args
    iftIntArray *labels = NULL;

    iftGetRequiredArgs(args, &reg_label_set, &out_localizer_path);
    iftGetOptionalArgs(args, reg_label_set->files[0]->path, &labels);

    timer *t1 = iftTic();

    printf(" - Finding Localizer\n");
    iftImage *localizer = iftFindLocalizer(reg_label_set, labels);

    printf(" - Writing Localizer\n");
    iftWriteImageByExt(localizer, out_localizer_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&reg_label_set);
    iftDestroyIntArray(&labels);
    iftDestroyImage(&localizer);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Train a localizer of brain substructures from a set of registered label images with the " \
        "required objects. The localizer is a label image with a bounding boxes to localize each required objects.\n" \
        "- It is founded by getting the bounding boxes from the union of all registered labeled images.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--registered-label-imgs-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Labeled Images with the target objects of interest."},
        {.short_name = "-o", .long_name = "--output-localizer", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Localizer (Label Image) path."},
        {.short_name = "-j", .long_name = "--obj-labels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Target Objects. Ex: 1:2:5 (3 objects)\nDefault: All labels from the First Image"},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **reg_label_set, char **out_localizer_path) {
    const char *reg_label_entry = iftGetConstStrValFromDict("--registered-label-imgs-entry", args);
    *out_localizer_path         = iftGetStrValFromDict("--output-localizer", args);

    *reg_label_set = iftLoadFileSetFromDirOrCSV(reg_label_entry, 0, true);

    char *parent_dir = iftParentDir(*out_localizer_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Registered Label Images Entry: \"%s\"\n", reg_label_entry);
    printf("- Output Localizer: \"%s\"\n", *out_localizer_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, const char *label_img_path, iftIntArray **labels) {
    *labels = iftGetLabelArrayFromCmdLine(args, label_img_path);

    printf("- Objects: [%d", (*labels)->val[0]);
    for (int i = 1; i < (*labels)->n; i++)
        printf(", %d", (*labels)->val[i]);
    printf("]\n");
    puts("-----------------------\n");
}


iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path) {
    iftIntArray *labels = NULL;

    if (iftDictContainKey("--obj-labels", args, NULL)) {
        iftSList *SL = iftSplitString(iftGetConstStrValFromDict("--obj-labels", args), ":");
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
    else {
        iftImage *label_img = iftReadImageByExt(label_path);
        labels              = iftGetObjectLabels(label_img);
        iftDestroyImage(&label_img);
    }

    return labels;
}


iftImage *iftFindLocalizer(  iftFileSet *reg_label_set,   iftIntArray *labels) {
    iftImage *label_img = iftReadImageByExt(reg_label_set->files[0]->path);
    iftImage *localizer = iftCreateImageFromImage(label_img);
    iftDestroyImage(&label_img);
    
    iftBoundingBox **mbbs = iftAllMinLabelsBoundingBox(reg_label_set, labels, NULL);

    for (int o = 0; o < labels->n; o++) {
        iftBoundingBox max_bb = iftMaxBoundingBox(mbbs[o], reg_label_set->n);

        printf("Obj: %d\n", labels->val[o]);
        printf("max_bb.begin: (%d, %d, %d)\n", max_bb.begin.x, max_bb.begin.y, max_bb.begin.z);
        printf("max_bb.end: (%d, %d, %d)\n\n", max_bb.end.x, max_bb.end.y, max_bb.end.z);

        for (int z = max_bb.begin.z; z <= max_bb.end.z; z++)
            for (int y = max_bb.begin.y; y <= max_bb.end.y; y++)
                for (int x = max_bb.begin.x; x <= max_bb.end.x; x++)
                    iftImgVal(localizer, x, y, z) = labels->val[o];

        iftFree(mbbs[o]);
    }

    iftFree(mbbs);

    return localizer;
}
/*************************************************************/


