#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **train_set, char **template_path,
                        char **template_label_path, char **out_bank_path);
void iftGetOptionalArgs(  iftDict *args, const char *label_path, float *scale, iftIntArray **labels,
                        bool *patch_size_power_of_two);
iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *train_set     = NULL;
    char *template_path       = NULL;
    char *template_label_path = NULL;
    char *out_bank_path       = NULL;
    float scale;
    iftIntArray *labels       = NULL;
    bool patch_size_power_of_two = false;

    iftGetRequiredArgs(args, &train_set, &template_path, &template_label_path, &out_bank_path);
    iftGetOptionalArgs(args, train_set->files[0]->path, &scale, &labels, &patch_size_power_of_two);

    timer *t1 = iftTic();

    puts("- Getting the Patches from Training Set");
    iftBoundingBox **patches = iftAllMinLabelsBoundingBox(train_set, labels, NULL);

    puts("- Reading Template Image");
    iftImage *template_img = iftReadImageByExt(template_path);

    puts("- Reading Template Label Image");
    iftImage *template_label_img = iftReadImageByExt(template_label_path);
    iftBoundingBox *template_patches = iftMinLabelsBoundingBox(template_label_img, labels, NULL);

    puts("- Scaling Bounding Boxes");
    if (!iftAlmostZero(scale - 1.0))
        for (int o = 0; o < labels->n; o++)
            for (int i = 0; i < train_set->n; i++)
                patches[o][i] = iftScaleBoundingBox(patches[o][i], scale);

    puts("- Training Bank of Patch Localizers");
    iftPatchLocalizerBank *bank = iftTrainPatchLocalizerBank(patches, template_img, template_patches,
                                                             labels->n, train_set->n, patch_size_power_of_two);

    // puts("- Saving Image");
    // iftImage *out = iftCreateImageFromImage(iftReadImageByExt(train_set->files[0]->path));
    // iftImage *tmp = iftCreateImageFromImage(template_img);
    // for (int o = 0; o < labels->n; o++) {
    //     iftFillBoundingBoxInImage(out, bank->localizers[o].patch, labels->val[o]);
    //     iftFillBoundingBoxInImage(out, bank->localizers[o].search_region, 2+labels->val[o]);

    //     iftFillBoundingBoxInImage(tmp, bank->localizers[o].template_patch, labels->val[o]);
    // }
    // iftWriteImageByExt(tmp, "tmp/template_img.hdr");
    // iftWriteImageByExt(out, "tmp/out.hdr");

    puts("- Writing Bank");
    iftWritePatchLocalizerBank(bank, out_bank_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    for (int o = 0; o < labels->n; o++)
        iftFree(patches[o]);
    iftFree(patches);
    iftDestroyDict(&args);
    iftDestroyFileSet(&train_set);
    iftFree(out_bank_path);
    iftDestroyIntArray(&labels);
    iftDestroyPatchLocalizerBank(&bank);
    iftDestroyImage(&template_img);
    iftDestroyImage(&template_label_img);
    iftFree(template_patches);


    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Train a patch localizer from the Minimum Bounding Boxes (patch) from a set of training labeled images.\n" \
        "- Firstly, find the patch for each object from each label image. Such patches can be scaled, in order to guarantee " \
        "a given margin to overcome localization errors.\n" \
        "- All resulting patches from the localizer will have the same size (in order to apply some Deep Learning techniques later), " \
        "which corresponds to largest patch from the training set.\n" \
        "- The position of the resulting patches will be the mean center from all patches from training set.\n" \
        "- A search region is also computed for the patch localizer by finding the box which contains " \
        "all centers from the training patches.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--training-label-image-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="CSV or directory with the Training Label Image Set with required target objects of interest."},
        {.short_name = "-r", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template (Reference Image) used to register the Input Label Images."},
        {.short_name = "-l", .long_name = "--template-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label Image of the Template (Reference Image) with the target objects."},
        {.short_name = "-o", .long_name = "--output-localizer", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output json file that stores the localizer information (*.zip)."},
        {.short_name = "-s", .long_name = "--scaling-factor", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Scaling Factor to scale the found patches from training set.\nDefault: 1.0"},
        {.short_name = "-j", .long_name = "--obj-labels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Target Objects. Ex: 1,2,5 (3 objects)\nDefault: All labels from the First Image"},
        {.short_name = "", .long_name = "--patch-size-power-of-two", .has_arg=false,
         .required=false, .help="Approximate the localizer patch size to values proprotional to the power of two"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **train_set, char **template_path,
                        char **template_label_path, char **out_bank_path) {
    const char *train_set_entry = iftGetConstStrValFromDict("--training-label-image-entry", args);
    *template_path              = iftGetStrValFromDict("--template", args);
    *template_label_path        = iftGetStrValFromDict("--template-label-image", args);
    *out_bank_path              = iftGetStrValFromDict("--output-localizer", args);

    if (!iftEndsWith(*out_bank_path, ".zip"))
        iftError("Invalid Output Localizer Path: %s\nTry *.zip", *out_bank_path);

    *train_set = iftLoadFileSetFromDirOrCSV(train_set_entry, 0, true);

    char *parent_dir = iftParentDir(*out_bank_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Training Labeled Set: \"%s\"\n", train_set_entry);
    printf("- Template: \"%s\"\n", *template_path);
    printf("- Template Label Image: \"%s\"\n", *template_label_path);
    printf("- Output (json) Localizer: \"%s\"\n", *out_bank_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, const char *label_path, float *scale, iftIntArray **labels,
                        bool *patch_size_power_of_two) {
    if (iftDictContainKey("--scaling-factor", args, NULL))
        *scale = iftGetDblValFromDict("--scaling-factor", args);
    else *scale = 1.0;

    *labels = iftGetLabelArrayFromCmdLine(args, label_path);
    
    printf("- Objects: [%d", (*labels)->val[0]);
    for (int i = 1; i < (*labels)->n; i++)
        printf(", %d", (*labels)->val[i]);
    puts("]");

    *patch_size_power_of_two = iftDictContainKey("--patch-size-power-of-two", args, NULL);

    printf("- Scaling Factor: %f\n", *scale);
    printf("- Patch Size in the Porwe of Two: %s\n", iftBoolAsString(*patch_size_power_of_two));
    puts("-----------------------\n");
}


iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path) {
    iftIntArray *labels = NULL;

    if (iftDictContainKey("--obj-labels", args, NULL)) {
        const char *labels_str = iftGetConstStrValFromDict("--obj-labels", args);

        if (!iftRegexMatch(labels_str, "^([0-9]+,?)+$"))
            iftError("Invalid Labels: %s", "iftGetLabelArrayFromCmdLine", labels_str);
        
        iftSList *SL = iftSplitString(labels_str, ",");
        labels       = iftCreateIntArray(SL->n);
        
        iftSNode *snode = SL->head;
        for (size_t o = 0; o < SL->n; o++) {
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
/*************************************************************/


