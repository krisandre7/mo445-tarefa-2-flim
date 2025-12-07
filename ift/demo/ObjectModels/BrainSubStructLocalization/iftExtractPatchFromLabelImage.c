#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **label_img_path, char **out_basename);
void iftGetOptionalArgs(  iftDict *args, iftSize *out_patch_size);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path = NULL;
    char *label_img_path = NULL;
    char *out_basename = NULL;
    iftSize out_patch_size = {0, 0, 0};

    iftGetRequiredArgs(args, &img_path, &label_img_path, &out_basename);
    iftGetOptionalArgs(args, &out_patch_size);

    iftBoundingBox out_base_patch;
    out_base_patch.begin.x = out_base_patch.begin.y = out_base_patch.begin.z = 0;
    out_base_patch.end.x = out_patch_size.x - 1;
    out_base_patch.end.y = out_patch_size.y - 1;
    out_base_patch.end.z = out_patch_size.z - 1;

    timer *t1 = iftTic();

    puts("- Reading Image");
    iftImage *img = iftReadImageByExt(img_path);

    puts("- Reading Image");
    iftImage *label_img = iftReadImageByExt(label_img_path);
    iftIntArray *labels = iftGetObjectLabels(label_img);

    puts("- Getting Patches");
    iftBoundingBox *out_patches = iftMinLabelsBoundingBox(label_img, labels, NULL);

    puts("- Saving Found Patches");
    for (int i = 0; i < labels->n; i++) {
        if (iftDictContainKey("--output-patch-size", args, NULL)) {
            iftVoxel center = iftBoundingBoxCenterVoxel(out_patches[i]);
            out_patches[i] = iftCentralizeBoundingBox(out_base_patch, center);
        }

        if (!iftDictContainKey("--skip-patch-saving", args, NULL))
            iftWriteBoundingBox(out_patches[i], "%s_patch%d.json", out_basename, i);

        if (!iftDictContainKey("--skip-patch-image-saving", args, NULL)) {
            iftImage *patch_label_img = iftExtractROI(img, out_patches[i]);
            iftWriteImageByExt(patch_label_img, "%s_patch%d%s", out_basename, i, iftFileExt(label_img_path));
            iftDestroyImage(&patch_label_img);
        }
        // iftFillBoundingBoxInImage(patch_mask, out_patches->bbs[i], i+1);
    }
    // iftWriteImageByExt(patch_mask, "%s_patch_mask%s", out_basename, iftFileExt(label_img_path));

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&label_img);
    iftFree(label_img_path);
    iftFree(out_basename);
    iftDestroyIntArray(&labels);
    iftFree(out_patches);
    iftDestroyImage(&img);
    iftDestroyImage(&label_img);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find and Extract patches (for each target object) from a label image.\n" \
        "- A fixed size of the output patches can be passed. In this case, a patch with this size " \
        "is centralized in each found patch." \
        "- The outputs are saved with the basename passed, being append a suffix for each type of file.\n" \
        "- Eg: 2 objects and output basename 'out/patch'\n" \
        "- out/patch_0.json\n" \
        "- out/patch_0.hdr\n" \
        "- out/patch_1.json\n" \
        "- out/patch_1.hdr";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Image."},
        {.short_name = "-l", .long_name = "--input-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Label Image."},
        {.short_name = "-o", .long_name = "--output-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Basename to save the found patches (.json) and the extracted image patches."},
        {.short_name = "-s", .long_name = "--output-patch-size", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="A fixed size for the output patches. Eg: 10,20,30"},
        {.short_name = "", .long_name = "--skip-patch-saving", .has_arg=false,
         .required=false, .help="Skip the saving of the Output Patch Files (*.json)."},
        {.short_name = "", .long_name = "--skip-patch-image-saving", .has_arg=false,
         .required=false, .help="Skip the saving of the Extracted Patch Images."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetRequiredArgs(  iftDict *args, char **img_path, char **label_img_path, char **out_basename) {
    *img_path       = iftGetStrValFromDict("--input-image", args);
    *label_img_path = iftGetStrValFromDict("--input-label-image", args);
    *out_basename   = iftGetStrValFromDict("--output-basename", args);

    char *parent_dir = iftParentDir(*out_basename);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Image: %s\n", *img_path);
    printf("- Label Image: %s\n", *label_img_path);
    printf("- Output Basename: %s\n", *out_basename);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, iftSize *out_patch_size) {
    if (iftDictContainKey("--output-patch-size", args, NULL)) {
        char *size_str = iftGetStrValFromDict("--output-patch-size", args);

        if (!iftRegexMatch(size_str, "^[0-9]+,[0-9]+,[0-9]+$"))
            iftError("Invalid Size: %s", "iftGetOptionalArgs", size_str);

        iftSList *SL = iftSplitString(size_str, ",");
        (*out_patch_size).x = atoi(iftRemoveSListHead(SL));
        (*out_patch_size).y = atoi(iftRemoveSListHead(SL));
        (*out_patch_size).z = atoi(iftRemoveSListHead(SL));
        printf("- Output Patch Size: %d, %d, %d\n", (*out_patch_size).x, (*out_patch_size).y, (*out_patch_size).z);
    }
    puts("-----------------------\n");
}
/*************************************************************/







