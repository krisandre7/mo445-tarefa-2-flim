#include "ift.h"

iftDict *iftGetArguments(int argc, const char **argv);

void iftInvertMaskLabelingOverridingTheOriginalOnes(  iftFileSet *mask_files);

int main(int argc, const char **argv) {
    iftDict *args = NULL;

    iftFileSet *mask_files = NULL;

    /*--------------------------------------------------------*/
    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/


    args = iftGetArguments(argc, argv);

    char *mask_dir = iftGetStrValFromDict("--input-dataset-mask-folder", args);

    mask_files = iftLoadFileSetFromDirBySuffix(mask_dir, "png");

    iftInvertMaskLabelingOverridingTheOriginalOnes(mask_files);

    iftDestroyFileSet(&mask_files);
    iftDestroyDict(&args);
    free(mask_dir);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);
}

void iftInvertMaskLabelingOverridingTheOriginalOnes(  iftFileSet *mask_files) {
    for(size_t i = 0; i < mask_files->n; i++) {
        fprintf(stderr,"Fixing labeling issues for image %s\n", mask_files->files[i]->path);
        iftImage *tmp_mask = iftReadImageByExt(mask_files->files[i]->path);

        iftImage *mask = iftThreshold(tmp_mask, 0, iftMaximumValue(tmp_mask)/2, 255);

        iftWriteImageByExt(mask, mask_files->files[i]->path);

        iftDestroyImage(&tmp_mask);
        iftDestroyImage(&mask);
    }
}



iftDict *iftGetArguments(int argc, const char **argv) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-m", .long_name = "--input-dataset-mask-folder", .has_arg=true, .required=false, .arg_type = IFT_STR_TYPE, .help="Input dataset folder with masks"},
    };

    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[IFT_STR_DEFAULT_SIZE] = "This program fixes the original labeling performed with ImageJ by inverting the labels (disease pixels are assigned 1, instead of 0).\n";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}
