#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **bank_path, char **out_basename);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path = NULL;
    char *bank_path = NULL;
    char *out_basename = NULL;

    iftGetRequiredArgs(args, &img_path, &bank_path, &out_basename);

    timer *t1 = iftTic();

    puts("- Reading Image");
    iftImage *img = iftReadImageByExt(img_path);

    puts("- Reading Patch Localizer Bank");
    iftPatchLocalizerBank *bank = iftReadPatchLocalizerBank(bank_path);

    puts("- Finding Patches");
    iftBoundingBoxArray *out_patches = iftFindPatchesByPatchLocalizerBank(img, bank);

    // iftImage *patch_mask = iftCreateImageFromImage(img);

    puts("- Saving Found Patches");
    for (int i = 0; i < bank->n; i++) {
        if (!iftDictContainKey("--skip-patch-saving", args, NULL))
            iftWriteBoundingBox(out_patches->val[i], "%s_patch_%04d.csv", out_basename, i+1);

        if (!iftDictContainKey("--skip-patch-image-saving", args, NULL)) {
            iftImage *patch_img = iftExtractROI(img, out_patches->val[i]);
            iftWriteImageByExt(patch_img, "%s_patch_%04d%s", out_basename, i+1, iftFileExt(img_path));
            iftDestroyImage(&patch_img);
        }
        // iftFillBoundingBoxInImage(patch_mask, out_patches->val[i], i+1);
    }
    // iftWriteImageByExt(patch_mask, "%s_patch_mask%s", out_basename, iftFileExt(img_path));

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyPatchLocalizerBank(&bank);
    iftFree(out_patches);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find and Extract patches from an image by a Patch Localizer.\n" \
        "- The localizer places the patches on the image and then applies a local search " \
        "by checking the Normalized Mutual Information between the patch and the true patch " \
        "on template.\n" \
        "- The outputs are saved with the basename passed, being append a suffix for each type of file.\n" \
        "- Eg: 2 patches and output basename 'out/patch'\n" \
        "- out/patch_0001.csv\n" \
        "- out/patch_0001.nii.gz\n" \
        "- out/patch_0002.csv\n" \
        "- out/patch_0002.nii.gz\n\n" \
        "PS: The image must have in the same space from the template of the bank.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Image."},
        {.short_name = "-b", .long_name = "--patch-localizer-bank", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Patch Localizer Bank."},
        {.short_name = "-o", .long_name = "--output-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Basename to save the found patches (.csv) and the extracted image patches."},
        {.short_name = "", .long_name = "--skip-patch-saving", .has_arg=false,
         .required=false, .help="Skip the saving of the Output Patch Files (*.csv)."},
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

void iftGetRequiredArgs(  iftDict *args, char **img_path, char **bank_path, char **out_basename) {
    *img_path     = iftGetStrValFromDict("--input-image", args);
    *bank_path    = iftGetStrValFromDict("--patch-localizer-bank", args);
    *out_basename = iftGetStrValFromDict("--output-basename", args);

    char *parent_dir = iftParentDir(*out_basename);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Image: %s\n", *img_path);
    printf("- Patch Localizer Bank: %s\n", *bank_path);
    printf("- Output Basename: %s\n", *out_basename);
    puts("-----------------------\n");
}
/*************************************************************/


