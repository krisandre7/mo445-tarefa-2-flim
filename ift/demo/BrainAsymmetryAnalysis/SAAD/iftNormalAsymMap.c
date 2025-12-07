#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, iftFileSet **img_set, char **out_map_path, char **mask_path, bool *use_stdev);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    iftFileSet *img_set = NULL;
    char *out_map_path = NULL;
    char *mask_path = NULL;
    bool use_stdev = false;

    iftGetInputs(args, &img_set, &out_map_path, &mask_path, &use_stdev);
    
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;

    timer *t1 = iftTic();
    
    puts("- Computing Mean Brain Asymmetries");
    iftImage *mean_asym = iftMeanBrainAsymMap(img_set, use_stdev);
    if (mask) {
        iftImage *aux = mean_asym;
        mean_asym = iftMask(mean_asym, mask);
        iftDestroyImage(&aux);
    }
    
    puts("- Writing Mean Brain Asymmetries");
    iftWriteImageByExt(mean_asym, out_map_path);
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftDestroyImage(&mean_asym);
    iftDestroyImage(&mask);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes an absolute asymmetry map for a set of (normal) control images.\n" \
        "- For each image, it computes the absolute asymmetries between the brain sides based on its Mid-Sagittal Plane (MSP).\n" \
        "- The normal asymmetry map consists of the mean absolute asymmetries from the image set. Optionally, it can " \
        "add the standard deviation from the absolute asymmetries.\n" \
        "- PS: All images must have the same domain, voxel sizes, orientation, and their MSPs must be the central sagittal slice.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory of CSV with the image pathnames."},
        {.short_name = "-o", .long_name = "--output-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output image pathname to save the asymmetry map."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
        .required=false, .help="Pathname with the mask of the target objects."},
        {.short_name = "", .long_name = "--use-stdev", .has_arg=false,
         .required=false, .help="Add/Use the Standard Deviation from the asymmetries into the output asymmetry map."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, iftFileSet **img_set, char **out_map_path, char **mask_path, bool *use_stdev) {
    const char *img_entry = iftGetConstStrValFromDict("--image-set", args);
    *img_set = iftLoadFileSetFromDirOrCSV(img_entry, 0, true);
    *out_map_path = iftGetStrValFromDict("--output-asymmetry-map", args);
    
    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    *use_stdev = iftDictContainKey("--use-stdev", args, NULL);

    puts("-----------------------");
    printf("- Image Entry: %s\n", img_entry);
    printf("- Output Asymmetry Map: %s\n", *out_map_path);
    puts("-----------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    printf("- Add/Use Standard Deviation Asymmetries: %s\n", iftBoolAsString(*use_stdev));
    puts("-----------------------\n");
}
/*************************************************************/


