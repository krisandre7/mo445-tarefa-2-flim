/**
 * @file
 * @brief Builds a Statistical Multi-Object Shape Model (without Texture Classifier(s)) from a set of Registered Segmentation Masks.
 * @note See the source code in @ref iftBuildSOSM.c
 *
 * @example iftBuildSOSM.c
 * @brief Builds a Statistical Multi-Object Shape Model (without Texture Classifier(s)) from a set of Registered Segmentation Masks.
 * @author Samuel Martins
 * @date Dec 15, 2016
 */



#include "ift.h"
#include "ift/medical/segm/SOSM-S.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **atlas_set, char **template_path, char **out_sosm_path);
void iftValidateRequiredArgs(const char *label_entry, const char *template_path, const char *out_sosm_path);
void iftGetOptionalArgs(  iftDict *args, const char *label_path, iftIntArray **labels);
iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path);
/*************************************************************/

int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    // mandatory args
    iftFileSet *atlas_set = NULL;
    char *template_path   = NULL;
    char *out_sosm_path   = NULL;
    // optional args
    iftIntArray *labels = NULL;

    iftGetRequiredArgs(args, &atlas_set, &template_path, &out_sosm_path);
    iftGetOptionalArgs(args, atlas_set->files[0]->path, &labels);

    timer *t1 = iftTic();

    puts("- Reading Reference Image");
    iftImage *ref_img = iftReadImageByExt(template_path);

    puts("- Training SOSM-S");
    iftSOSMS *sosms = iftTrainSOSMS(atlas_set, ref_img, labels);
    puts("- Writing SOSM-S");
    iftWriteSOSMS(sosms, out_sosm_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&atlas_set);
    iftFree(out_sosm_path);
    iftDestroyImage(&ref_img);
    iftDestroySOSMS(&sosms);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program builds a SOSM-S Model [1] " \
        "from a set of Registered Segmentation Masks.\n" \
        "- If a given Target Object is not passed, all Labels from the First Image will be considered " \
        "for the construction of the Models.\n\n" \
        "[1] Phellan, 2016 - Medical physics - Medical image segmentation via atlases and fuzzy object models";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--atlas-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Atlas (Label Images) already registered or a CSV file with their pathnames."},
        {.short_name = "-r", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template (Reference Image) used to register the Input Label Images."},
        {.short_name = "-o", .long_name = "--output-sosms", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Zip File to store the Statistical Multi-Object Shape Model"},
        {.short_name = "-j", .long_name = "--labels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Target Objects. Ex: 1:2:5 (3 objects)\nDefault: All labels from the First Image"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **atlas_set, char **template_path, char **out_sosm_path) {
    const char *label_entry = iftGetConstStrValFromDict("--atlas-entry", args);
    *template_path          = iftGetStrValFromDict("--template", args);
    *out_sosm_path          = iftGetStrValFromDict("--output-sosms", args);

    iftValidateRequiredArgs(label_entry, *template_path, *out_sosm_path);

    *atlas_set = iftLoadFileSetFromDirOrCSV(label_entry, 0, true);

    puts("-----------------------");
    printf("- Atlas Entry: \"%s\"\n", label_entry);
    printf("- Template: \"%s\"\n", *template_path);
    printf("- Output SOSM-S Model: \"%s\"\n", *out_sosm_path);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *label_entry, const char *template_path, const char *out_sosm_path) {
    // LABEL IMAGE ENTRY
    if (iftFileExists(label_entry)) {
        if (!iftEndsWith(label_entry, ".csv")) {
            iftError("The Label Image Set Entry File \"%s\" is not a CSV file... Try *.csv", "iftValidateRequiredArgs", label_entry);
        }
    }
    else if (!iftDirExists(label_entry))
        iftError("Invalid Pathname for the Label Image Set Entry: \"%s\"...\nTry a valid Directory or a CSV file",
                 "iftValidateRequiredArgs", label_entry);

    // REFERENCE IMAGE
    if (!iftIsImageFile(template_path))
        iftError("Invalid Reference Image: \"%s\"", "iftValidateRequiredArgs", template_path);

    // OUT MODEL PATH
    if (!iftEndsWith(out_sosm_path, ".zip"))
        iftError("Invalid Zip File Extension: \"%s\". Try *.zip", "iftValidateRequiredArgs", iftFileExt(out_sosm_path));

    char *parent_dir = iftParentDir(out_sosm_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    free(parent_dir);
}


void iftGetOptionalArgs(  iftDict *args, const char *label_path, iftIntArray **labels) {
    // TARGET LABELS
    *labels = iftGetLabelArrayFromCmdLine(args, label_path);
    
    printf("- Objects: [%d", (*labels)->val[0]);
    for (int i = 1; i < (*labels)->n; i++)
        printf(", %d", (*labels)->val[i]);
    printf("]\n");
    puts("-----------------------\n");
}


iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path) {
    iftIntArray *labels = NULL;

    if (iftDictContainKey("--labels", args, NULL)) {
        iftSList *SL = iftSplitString(iftGetConstStrValFromDict("--labels", args), ":");
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
/*************************************************************/



















