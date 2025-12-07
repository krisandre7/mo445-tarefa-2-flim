#include "ift.h"
#include "ift/medical/brain/PreProcessing.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **out_path);
void iftGetOptionalArgs(  iftDict *args, int *nbits, iftImage **mask, iftImage **ref_img, iftImage **ref_mask,
                        char **msp_path, char **out_msp_path, bool *skip_n4, bool *skip_median_filter, bool *skip_msp_alignment, bool *skip_hist_matching);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    // mandatory args
    char *img_path = NULL;
    char *out_path = NULL;
    // optional args
    int nbits = 0;
    iftImage *mask = NULL;
    iftImage *ref_img = NULL;
    iftImage *ref_mask = NULL;
    char *msp_path = NULL;
    char *out_msp_path = NULL;
    bool skip_n4;
    bool skip_median_filter;
    bool skip_msp_alignment;
    bool skip_hist_matching;
    
    iftGetRequiredArgs(args, &img_path, &out_path);
    iftGetOptionalArgs(args, &nbits, &mask, &ref_img, &ref_mask, &msp_path, &out_msp_path, &skip_n4, &skip_median_filter,
                       &skip_msp_alignment, &skip_hist_matching);
    

    timer *t1 = iftTic();
    
    puts("- Reading Input Image");
    iftImage *img = iftReadImageByExt(img_path);

    printf("- Applying Default MR Brain Image Pre-Processing\n");
    iftPlane *msp = (!skip_msp_alignment && msp_path) ? iftReadPlane(msp_path) : NULL;
    iftPlane *msp_out = NULL;
    
    iftImage *out_img = iftBrainMRIPreProcessing(img, nbits, msp, mask, ref_img, ref_mask, skip_n4, skip_median_filter,
                                                 skip_msp_alignment, skip_hist_matching, &msp_out);
    
    printf("- Writing Output Image\n");
    iftWriteImageByExt(out_img, out_path);

    if (out_msp_path) {
        printf("- Writing MSP\n");
        iftWritePlane(msp_out, out_msp_path);
    }
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(img_path);
    iftFree(out_path);
    iftDestroyImage(&img);
    iftDestroyImage(&out_img);
    iftDestroyImage(&mask);
    iftDestroyImage(&ref_img);
    iftDestroyImage(&ref_mask);
    iftDestroyPlane(&msp);
    iftDestroyPlane(&msp_out);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Apply the Default Pre-Processing in a Brain Image.\n" \
        "- The pre-processing consists of the following ordered steps:\n" \
        "1 - Bias Field Correction by N4;\n" \
        "2 - Median Filter using a Spherical Adjacency of Radius 1;\n" \
        "3 - Normalization within [0, 2ˆnbits - 1] (nbits is passed);\n" \
        "4 - Extraction and alignment with the Mid-Sagittal Plain (MSP). A MSP previously extracted can be passed;\n" \
        "5 - Histogram Matching with a given Reference Image (binary masks can be passed to define the regions for Hist. Matching)\n\n" \
        
        "Obs: If a valid MSP path is passed, the program uses it to align the image. Otherwise, the program extracts the MSP.\n" \
        "Obs2: If an output path is passed to save the MSP, the program saves the MSP used for the alignment.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Brain Image to be pre-processed."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the output pre-processed brain image."},
        {.short_name = "-b", .long_name = "--num-bits", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Number of bits for Normalization (eg., 12). Ignore this option for no normalization."},
        {.short_name = "-mi", .long_name = "--binary-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Binary mask of the Input Image used for Histogram Matching."}, 
        {.short_name = "-r", .long_name = "--reference-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Reference Image used for Histogram Matching."},
        {.short_name = "-mr", .long_name = "--reference-binary-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Binary mask of the Reference Image used for Histogram Matching."},
        {.short_name = "-p", .long_name = "--msp", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the Mid-Sagittal Plane (previously extracted) for the alignment."},
        {.short_name = "-q", .long_name = "--output-msp", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname to save the extracted MSP."},
        {.short_name = "", .long_name = "--skip-n4", .has_arg=false,
         .required=false, .help="Skip N4 Bias Field Correction."},
        {.short_name = "", .long_name = "--skip-median-filter", .has_arg=false,
         .required=false, .help="Skip Median Filter."},
        {.short_name = "", .long_name = "--skip-msp-alignment", .has_arg=false,
         .required=false, .help="Skip the extraction and image alignment by its MSP."},
        {.short_name = "", .long_name = "--skip-hist-matching", .has_arg=false,
         .required=false, .help="Skip Histogram Matching."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **out_path) {
    *img_path = iftGetStrValFromDict("--input-img", args);
    *out_path = iftGetStrValFromDict("--output-img", args);

    if (!iftFileExists(*img_path))
        iftError("Input image %s does not exists", "iftGetRequiredArgs", *img_path);

    puts("-----------------------");
    printf("- Input Brain Image: %s\n", *img_path);
    printf("- Output Image: %s\n", *out_path);
    puts("-----------------------");
}

void iftGetOptionalArgs(  iftDict *args, int *nbits, iftImage **mask, iftImage **ref_img, iftImage **ref_mask,
                        char **msp_path, char **out_msp_path, bool *skip_n4, bool *skip_median_filter, bool *skip_msp_alignment, bool *skip_hist_matching) {
    if (iftDictContainKey("--num-bits", args, NULL)) {
        *nbits = iftGetLongValFromDict("--num-bits", args);
        printf("- Number of Bits: %d\n", *nbits);
    }
    else *nbits = 0;
    
    *skip_hist_matching = iftDictContainKey("--skip-hist-matching", args, NULL);
    
    if (!*skip_hist_matching) {
        const char *mask_path = NULL;
        if (iftDictContainKey("--binary-mask", args, NULL)) {
            mask_path = iftGetConstStrValFromDict("--binary-mask", args);
            *mask = iftReadImageByExt(mask_path);
            printf("- Binary Mask Path: %s\n", mask_path);
        }
        else *mask = NULL;

        const char *ref_img_path = NULL;
        if (iftDictContainKey("--reference-image", args, NULL)) {
            ref_img_path = iftGetConstStrValFromDict("--reference-image", args);
            *ref_img = iftReadImageByExt(ref_img_path);
            printf("- Ref. Image Path: %s\n", ref_img_path);
        }
        else iftError("No reference image was passed for histogram matching", "iftGetOptionalArgs");
    
        const char *ref_mask_path = NULL;
        if (iftDictContainKey("--reference-binary-mask", args, NULL)) {
            ref_mask_path = iftGetConstStrValFromDict("--reference-binary-mask", args);
            *ref_mask = iftReadImageByExt(ref_mask_path);
            printf("- Reference Binary Mask Path: %s\n", ref_mask_path);
        }
        else *ref_mask = NULL;
    }
    
    
    *skip_msp_alignment = iftDictContainKey("--skip-msp-alignment", args, NULL);
    if (!*skip_msp_alignment) {
        if (iftDictContainKey("--msp", args, NULL)) {
            *msp_path = iftGetStrValFromDict("--msp", args);
            
            if (!iftEndsWith(*msp_path, ".json"))
                iftError("Invalid MSP path: %s\nTry *.json", "iftGetOptionalArgs", *msp_path);
            
            printf("- MSP used for Alignment: %s\n", *msp_path);
        }
        else {
            *msp_path = NULL;
            printf("- MSP will be extracted\n");
        }
    
        if (iftDictContainKey("--output-msp", args, NULL)) {
            *out_msp_path = iftGetStrValFromDict("--output-msp", args);
        
            if (!iftEndsWith(*out_msp_path, ".json"))
                iftError("Invalid MSP path: %s\nTry *.json", "iftGetOptionalArgs", *out_msp_path);
        
            printf("- Output MSP path: %s\n", *out_msp_path);
        }
        else *out_msp_path = NULL;
    }
    
    *skip_n4 = iftDictContainKey("--skip-n4", args, NULL);
    *skip_median_filter = iftDictContainKey("--skip-median-filter", args, NULL);
    
    printf("- Skip N4: %s\n", iftBoolAsString(*skip_n4));
    printf("- Skip Median Filter: %s\n", iftBoolAsString(*skip_median_filter));
    if (*nbits == 0)
        printf("- Skip Normalization\n");
    printf("- Skip MSP Alignment: %s\n", iftBoolAsString(*skip_msp_alignment));
    printf("- Skip Hist. Matching: %s\n", iftBoolAsString(*skip_hist_matching));
    puts("-----------------------\n");
}

