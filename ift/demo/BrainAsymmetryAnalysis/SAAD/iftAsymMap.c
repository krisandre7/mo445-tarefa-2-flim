#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, char **img_path, char **out_path, char **mask_path,
                  char **normal_asym_map_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path = NULL;
    char *out_path = NULL;
    char *mask_path = NULL;
    char *normal_asym_map_path = NULL;


    iftGetInputs(args, &img_path, &out_path, &mask_path, &normal_asym_map_path);
    
    timer *t1 = iftTic();
    
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;
    iftImage *normal_asym_map = (normal_asym_map_path) ? iftReadImageByExt(normal_asym_map_path) : NULL;

    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *asym_map = iftBrainAsymMap(img, normal_asym_map);
    if (mask) {
        iftImage *aux = asym_map;
        asym_map = iftMask(asym_map, mask);
        iftDestroyImage(&aux);
    }
    iftWriteImageByExt(asym_map, out_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&asym_map);
    iftDestroyImage(&mask);
    iftDestroyImage(&normal_asym_map);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes the voxel-wise absolute asymmetries between the brain right and left sides according to its Mid-Sagittal Plane (MSP).\n" \
        "- The function assumes that the image's MSP is its central sagittal slice (xslice).\n" \
        "- If a mask is passed, the resulting asymmetry map will be only considered inside it.";;

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Image."},
        {.short_name = "-o", .long_name = "--out-asym-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the resulting image with the asymmetries."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname with the mask of the target objects."},
        {.short_name = "-s", .long_name = "--normal-asym-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the normal asymmetry map used to attenuate the brain asymmetries " \
                                "for the seed initialization."} 
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, char **img_path, char **out_path, char **mask_path,
                  char **normal_asym_map_path) {
    *img_path = iftGetStrValFromDict("--image-path", args);
    *out_path = iftGetStrValFromDict("--out-asym-map", args);
    
    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    if (iftDictContainKey("--normal-asym-map", args, NULL))
        *normal_asym_map_path = iftGetStrValFromDict("--normal-asym-map", args);
    else *normal_asym_map_path = NULL;

    puts("--------------------");
    printf("- Image Path: %s\n", *img_path);
    printf("- Output Assymetry Image Path: %s\n", *out_path);
    puts("--------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    if (*normal_asym_map_path)
        printf("- Normal asymmetry Map: %s\n", *normal_asym_map_path);
    puts("--------------------\n");
}
/*************************************************************/


