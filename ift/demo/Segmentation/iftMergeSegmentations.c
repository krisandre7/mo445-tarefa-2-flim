/**
 * @file
 * @brief Merge a set of Segmented (Label) Images into a single one.
 * 
 * @example iftMergeSegmentations.c
 * @brief Merge a set of Segmented (Label) Images into a single one.
 * @author Samuel Martins
 * @date Jul 11, 2016
 */

#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, char **out_img_path);
void iftValidateRequiredArgs(const char *img_entry, const char *out_img_path);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // madatory args
    iftFileSet *img_set = NULL;
    char *out_img_path  = NULL;

    iftGetRequiredArgs(args, &img_set, &out_img_path);

    timer *t1 = iftTic();

    printf("- [0] = %s\n", img_set->files[0]->path);
    iftImage *out_img = iftReadImageByExt(img_set->files[0]->path);

    for (size_t i = 1; i < img_set->n; i++) {
        printf("- [%lu] = %s\n", i, img_set->files[i]->path);
        iftImage *img   = iftReadImageByExt(img_set->files[i]->path);

        #pragma omp parallel
        for (int p = 0; p < img->n; p++) {
            if (img->val[p] > 0) {
                out_img->val[p] = img->val[p];
            }
        }

        iftDestroyImage(&img);
    }

    puts("\nWriting Merged Image...");
    iftWriteImageByExt(out_img, out_img_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftDestroyImage(&out_img);
    iftFree(out_img_path);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "Merge a set of Segmented (Label) Images into a single one.";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--image-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Dir with the Segmented (Label) Images to be merged or a CSV file with their pathnames."},
            {.short_name = "-o", .long_name = "--output-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Output Merged Image Pathname."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, char **out_img_path) {
    const char *img_entry = iftGetConstStrValFromDict("--image-entry", args);
    *out_img_path         = iftGetStrValFromDict("--output-image", args);

    iftValidateRequiredArgs(img_entry, *out_img_path);

    *img_set = iftLoadFileSetFromDirOrCSV(img_entry, 0, false);
    iftValidateFileSet(*img_set);

    puts("-----------------------");
    printf("- Image Entry: \"%s\"\n", img_entry);
    printf("- Output Merged Image: \"%s\"\n", *out_img_path);
    puts("-----------------------\n");
}


void iftValidateRequiredArgs(const char *img_entry, const char *out_img_path) {
    // IMAGE ENTRY
    if (iftFileExists(img_entry)) {
        if (!iftEndsWith(img_entry, ".csv")) {
            iftError("Image Set Entry File \"%s\" is not a CSV file... Try *.csv", "iftValidateRequiredArgs", img_entry);
        }
    }
    else if (!iftDirExists(img_entry))
        iftError("Invalid Pathname for the Label Image Set Entry: \"%s\"...\nTry a valid Directory or a CSV file",
                 "iftValidateRequiredArgs", img_entry);

    if (!iftIsImagePathnameValid(out_img_path))
        iftError("Invalid Output Image Pathname: \"%s\"", "iftValidateRequiredArgs", out_img_path);

    char *parent_dir = iftParentDir(out_img_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);
}



