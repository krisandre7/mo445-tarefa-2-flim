#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **label_img_path, char **out_json_path);
void iftGetOptionalArgs(  iftDict *args, int *max_disp_mag, double *scale_factor);
iftBoundingBox iftSimulateUserBoundingBox(iftBoundingBox bb, int max_disp_mag, double scale_factor);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *label_img_path = NULL;
    char *out_json_path  = NULL;
    // optional args
    int max_disp_mag;
    double scale_factor;

    iftGetRequiredArgs(args, &label_img_path, &out_json_path);
    iftGetOptionalArgs(args, &max_disp_mag, &scale_factor);

    iftRandomSeed(time(NULL));

    timer *t1 = iftTic();

    puts("- Reading Label Image");
    iftImage *label_img = iftReadImageByExt(label_img_path);

    puts("- Getting the labels");
    iftIntArray *labels  = iftGetObjectLabels(label_img);

    puts("- Getting the Min. Bounding Boxes");
    iftBoundingBox *umbbs = iftMinLabelsBoundingBox(label_img, labels, NULL);

    for (int o = 0; o < labels->n; o++) {
        printf("[%d] Object: %d\n", o, labels->val[o]);
        printf("umbbs.begin: (%d, %d, %d)\n", umbbs[o].begin.x, umbbs[o].begin.y, umbbs[o].begin.z);
        printf("umbbs.end: (%d, %d, %d)\n", umbbs[o].end.x, umbbs[o].end.y, umbbs[o].end.z);

        umbbs[o] = iftSimulateUserBoundingBox(umbbs[o], max_disp_mag, scale_factor);

        printf("umbbs.begin: (%d, %d, %d)\n", umbbs[o].begin.x, umbbs[o].begin.y, umbbs[o].begin.z);
        printf("umbbs.end: (%d, %d, %d)\n\n", umbbs[o].end.x, umbbs[o].end.y, umbbs[o].end.z);
    }

    puts("- Writing the Json File");
    iftWriteLabelBoundingBox(umbbs, labels, out_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(label_img_path);
    iftFree(out_json_path);
    iftDestroyImage(&label_img);
    iftDestroyIntArray(&labels);
    iftFree(umbbs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Simulates the Min Bounding Boxes (for the required objects) drawn by the user by translating ." \
        "and scaling the true MBBs.\n"
        "- A translation vector is created by randomly selecting a displacement for each coordinate which " \
        "can have a maximum and minimum displacement passed by the user.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label Image with required target objects of interest."},
        {.short_name = "-o", .long_name = "--output-mbb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Json Path with the simulated MBBs."},
        {.short_name = "-d", .long_name = "--min-max-displacement-magnitude", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Value which indicates the the magnitude of the max and min displacement for translation.\n" \
                               "E.g: If -d 3, the displacement in each coordinate can be in [-3, 3]\n" \
                               "Default: 3"},
        {.short_name = "-a", .long_name = "--scale-factor", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Factor to scale the true (Bounding Boxes). Default: 1.0"}

    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **label_img_path, char **out_json_path) {
    *label_img_path = iftGetStrValFromDict("--label-image", args);
    *out_json_path  = iftGetStrValFromDict("--output-mbb-json", args);

    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Label Image: \"%s\"\n", *label_img_path);
    printf("- Output MBB Json Path: \"%s\"\n", *out_json_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, int *max_disp_mag, double *scale_factor) {
    if (iftDictContainKey("--min-max-displacement-magnitude", args, NULL))
        *max_disp_mag = iftGetLongValFromDict("--min-max-displacement-magnitude", args);
    else *max_disp_mag = 3;
    
    if (iftDictContainKey("--scale-factor", args, NULL))
        *scale_factor = iftGetDblValFromDict("--scale-factor", args);
    else *scale_factor = 1.0;

    printf("- Min Max Displacement Magnitude: %d\n", *max_disp_mag);
    printf("- Scaling Factor: %lf\n", *scale_factor);
    puts("-----------------------\n");
}


iftBoundingBox iftSimulateUserBoundingBox(iftBoundingBox bb, int max_disp_mag, double scale_factor) {
    max_disp_mag = abs(max_disp_mag);
    int max_disp = max_disp_mag;
    int min_disp = -1 * max_disp_mag;

    iftBoundingBox ubb = iftScaleBoundingBox(bb, scale_factor);

    // Since it is too difficult that the user draws a mbb centralized in the target object,
    // we try to simulate a real situation by enlarging the mbbs by a factor and moving the mbb's center
    // by a random displacement vector.
    iftVector disp_mbb_vec;
    disp_mbb_vec.x = iftRandomInteger(min_disp, max_disp);
    disp_mbb_vec.y = iftRandomInteger(min_disp, max_disp);
    disp_mbb_vec.z = iftRandomInteger(min_disp, max_disp);

    ubb.begin.x += disp_mbb_vec.x;
    ubb.end.x   += disp_mbb_vec.x;
    ubb.begin.y += disp_mbb_vec.y;
    ubb.end.y   += disp_mbb_vec.y;
    ubb.begin.z += disp_mbb_vec.z;
    ubb.end.z   += disp_mbb_vec.z;

    printf("disp_mbb_vec: (%.0f, %.0f, %.0f)\n\n", disp_mbb_vec.x, disp_mbb_vec.y, disp_mbb_vec.z);

    return ubb;
}
/*************************************************************/






