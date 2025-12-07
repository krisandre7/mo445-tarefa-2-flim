#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **volBrain_label_img_path, char **out_img_path);
void iftGetOptionalArgs(  iftDict *args, float *e_radius, float *d_radius);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path           = NULL;
    char *volBrain_label_img_path = NULL;
    char *out_img_path            = NULL;
    // optional args
    float e_radius; // erosion radius
    float d_radius; // dilation radius


    iftGetRequiredArgs(args, &test_img_path, &volBrain_label_img_path, &out_img_path);
    iftGetOptionalArgs(args, &e_radius, &d_radius);

    timer *t1 = iftTic();

    puts("- Reading Test Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);

    puts("- Reading Test Image");
    iftImage *volBrain_label_img = iftReadImageByExt(volBrain_label_img_path);

    puts("- Getting Labels");
    iftIntArray *labels = iftGetObjectLabels(volBrain_label_img);

    puts("- Getting Seeds");
    iftLabeledSet *all_seeds = NULL;

    for (int o = 0; o < labels->n; o++) {
        printf("Object: %d\n", labels->val[o]);
        iftImage *obj_img    = iftExtractObject(volBrain_label_img, labels->val[o]);
        
        iftSet *bg_seeds     = NULL;
        iftImage *dilate_img = iftDilateBin(obj_img, &bg_seeds, d_radius);
        iftWriteImageByExt(dilate_img, "dilate_%d.hdr", labels->val[o]);
        iftDestroyImage(&dilate_img);

        iftSet *obj_seeds    = NULL;
        iftImage *erode_img = iftErodeBin(obj_img, &obj_seeds, e_radius);
        iftWriteImageByExt(erode_img, "erode_%d.hdr", labels->val[o]);
        iftDestroyImage(&erode_img);

        iftInsertSetIntoLabeledSet(&bg_seeds, 0, &all_seeds);
        iftInsertSetIntoLabeledSet(&obj_seeds, labels->val[o], &all_seeds);

        iftDestroyImage(&obj_img);
    }

    puts("\n- Segmenting by IFT-Watershed");
    iftWriteSeeds("seeds.txt", all_seeds, volBrain_label_img);
    iftImage *grad_img = iftImageBasins(test_img, NULL);
    iftImage *seg_img  = iftWatershed(grad_img, NULL, all_seeds, NULL);
    iftWriteImageByExt(seg_img, out_img_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftFree(volBrain_label_img_path);
    iftFree(out_img_path);
    iftDestroyImage(&test_img);
    iftDestroyImage(&volBrain_label_img);
    iftDestroyIntArray(&labels);
    iftDestroyImage(&grad_img);
    iftDestroyImage(&seg_img);


    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segment the image by IFT using the volBrain segmentation to estimate the seeds.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image to be segmented. It might have to be previously registered on training " \
                                "reference image, or it will be in this program"},
        {.short_name = "-l", .long_name = "--volBrain-segmentation", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="volBrain segmentation to estimate the seeds for IFT"},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output Segmented Image."},
        {.short_name = "-e", .long_name = "--erosion-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Erosion Radius.\n" \
                                 "Default 3.0"},
        {.short_name = "-d", .long_name = "--dilation-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Radius Dilation.\n" \
                                 "Default 0.0"}

    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **volBrain_label_img_path, char **out_img_path) {
    *test_img_path           = iftGetStrValFromDict("--test-img", args);
    *volBrain_label_img_path = iftGetStrValFromDict("--volBrain-segmentation", args);
    *out_img_path            = iftGetStrValFromDict("--output-img", args);

    char *parent_dir = iftParentDir(*out_img_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- MRI Test Image: %s\n", *test_img_path);
    printf("- volBrain Label Image: %s\n", *volBrain_label_img_path);
    printf("- Output Image Path: %s\n", *out_img_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *e_radius, float *d_radius) {
    if (iftDictContainKey("--erosion-radius", args, NULL)) {
        *e_radius = iftGetDblValFromDict("--erosion-radius", args);
    }
    else *e_radius = 3.0f;

    if (iftDictContainKey("--dilation-radius", args, NULL)) {
        *d_radius = iftGetDblValFromDict("--dilation-radius", args);
    }
    else *d_radius = 0.0f;

    printf("- Erosion Radius: %f\n", *e_radius);
    printf("- Dilation Radius: %f\n", *d_radius);
    puts("-----------------------\n");
}
/*************************************************************/

















