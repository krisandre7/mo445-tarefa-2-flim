#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, char **reg_error_mag_path, char **svoxels_img_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *reg_error_mag_path = NULL;
    char *svoxels_img_path = NULL;

    iftGetInputs(args, &reg_error_mag_path, &svoxels_img_path);

    timer *t1 = iftTic();
    
    iftImage *reg_error_mag = iftReadImageByExt(reg_error_mag_path);
    iftImage *svoxels_img = iftReadImageByExt(svoxels_img_path);

    iftIntArray *svoxels = iftGetObjectLabels(svoxels_img);
    
    int max_label = iftMaximumValue(svoxels_img);
    iftFloatArray *mean_reg_errors = iftCreateFloatArray(max_label + 1);
    iftIntArray *svoxel_vols = iftCreateIntArray(max_label + 1);

    for (int p = 0; p < svoxels_img->n; p++) {
        int label = svoxels_img->val[p];
        mean_reg_errors->val[label] += reg_error_mag->val[p];
        svoxel_vols->val[label]++;
    }

    for (int o = 0; o < svoxels->n; o++) {
        int label = svoxels->val[o];
        mean_reg_errors->val[label] /= svoxel_vols->val[label];

        printf("- %s - SVoxel: %d ==> Mean Reg. Error. Mag: %04.2f - vol %d\n", iftFilename(svoxels_img_path, NULL), label, mean_reg_errors->val[label], svoxel_vols->val[label]);
    }

    // puts("\nDone...");
    // puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&reg_error_mag);
    iftDestroyImage(&svoxels_img);
    iftDestroyFloatArray(&mean_reg_errors);
    iftDestroyIntArray(&svoxel_vols);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Compute the Mean Registration Error inside each supervoxel.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--reg-error-mag", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the registration error magnitude from an image."},
        {.short_name = "-s", .long_name = "--supervoxels-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname with the Supervoxel Mask."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetInputs(  iftDict *args, char **reg_error_mag_path, char **svoxels_img_path) {
    *reg_error_mag_path = iftGetStrValFromDict("--reg-error-mag", args);
    *svoxels_img_path = iftGetStrValFromDict("--supervoxels-path", args);

    // puts("--------------------");
    // printf("- Registration Error Magnitude Path: %s\n", *reg_error_mag_path);
    // printf("- Supervoxel Mask: %s\n", *svoxels_img_path);
    // puts("--------------------\n");
}
/*************************************************************/


