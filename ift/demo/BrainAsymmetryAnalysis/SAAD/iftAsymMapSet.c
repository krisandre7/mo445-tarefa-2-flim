#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, iftFileSet **img_set, char **out_dir, char **mask_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    iftFileSet *img_set = NULL;
    char *out_dir = NULL;
    char *mask_path = NULL;

    iftGetInputs(args, &img_set, &out_dir, &mask_path);
    
    timer *t1 = iftTic();
    
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;

    #pragma omp parallel for
    for (long i = 0; i < img_set->n; i++) {
        char *filename = iftFilename(img_set->files[i]->path, NULL);
        char *out_path = iftJoinPathnames(2, out_dir, filename);
        printf("[%ld/%ld]\n%s\n%s\n\n", i, img_set->n - 1, img_set->files[i]->path, out_path);

        iftImage *img = iftReadImageByExt(img_set->files[i]->path);
        iftImage *asym_map = iftBrainAsymMap(img, NULL);
        if (mask) {
            iftImage *aux = asym_map;
            asym_map = iftMask(asym_map, mask);
            iftDestroyImage(&aux);
        }
        iftWriteImageByExt(asym_map, out_path);
        
        iftFree(filename);
        iftFree(out_path);
        iftDestroyImage(&img);
        iftDestroyImage(&asym_map);
    }


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftDestroyImage(&mask);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes the voxel-wise absolute asymmetries between the brain right and left sides according to its Mid-Sagittal Plane (MSP) " \
        "for each brain image from a set.\n" \
        "- The function assumes that the image's MSP is its central sagittal slice (xslice).\n" \
        "- The absolute asymmetries of each brain image is saved in the output directory with the same image's filename.\n" \
        "- If a mask is passed, the resulting asymmetry maps will be only considered inside it.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-set-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="CSV or directory with the brain images."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the resulting image with the asymmetries."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname with the mask of the target objects."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, iftFileSet **img_set, char **out_dir, char **mask_path) {
    const char *img_path = iftGetStrValFromDict("--image-set-entry", args);
    *img_set = iftLoadFileSetFromDirOrCSV(img_path, 0, true);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    
    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    puts("--------------------");
    printf("- Brain Image Set Entry: %s\n", img_path);
    printf("- Output Directory: %s\n", *out_dir);
    puts("--------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    puts("--------------------\n");
}
/*************************************************************/


