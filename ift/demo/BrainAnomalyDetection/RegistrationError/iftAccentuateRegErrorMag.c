#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, char **reg_mag_error_path, char **mask_path, char **out_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *reg_mag_error_path = NULL;
    char *mask_path = NULL;
    char *out_path = NULL;

    iftGetInputs(args, &reg_mag_error_path, &mask_path, &out_path);

    timer *t1 = iftTic();
    
    iftImage *reg_error_mag = iftReadImageByExt(reg_mag_error_path);
    iftImage *mask = iftReadImageByExt(mask_path);

    puts("- Computing EDT");
    iftImage *edt = iftEuclDistTrans(mask, NULL, IFT_INTERIOR, NULL, NULL, NULL);    
    iftIntArray *max_vals = iftMaximumObjectValues(edt, mask, NULL);

    puts("- Computing Max Distance for each Object");
    iftFImage *edt_norm = iftCreateFImage(edt->xsize, edt->ysize, edt->zsize);
    puts("- Normalizing Distances of Each Object to [0.0, 1.0]");
    for (int p = 0; p < edt->n; p++) {
        int label = mask->val[p];
        if (label) {
            edt_norm->val[p] = edt->val[p] / (max_vals->val[label] * 1.0);
        }
    }

    iftImage *reg_error_mag_acc = iftCreateImageFromImage(reg_error_mag);
    puts("- Attenuating Registration Errors");
    for (int p = 0; p < reg_error_mag->n; p++) {
        float factor = (1 + 0.75 * (1.0 - edt_norm->val[p]));
        reg_error_mag_acc->val[p] = reg_error_mag->val[p] * factor;
    }
    iftWriteImageByExt(reg_error_mag_acc, "tmp/reg_error_mag_linear.nii.gz");

    puts("- Attenuating Registration Errors");
    for (int p = 0; p < reg_error_mag->n; p++) {
        float factor = (1 + 0.75 * powf(edt_norm->val[p] - 1, 2));
        reg_error_mag_acc->val[p] = reg_error_mag->val[p] * factor;
    }
    iftWriteImageByExt(reg_error_mag_acc, "tmp/reg_error_mag_exp2.nii.gz");



    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&reg_error_mag);
    iftDestroyImage(&mask);
    iftDestroyImage(&edt);
    iftDestroyIntArray(&max_vals);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Accentuate the Magnitude of the Registration Error based on the EDT for objects in a segmentation mask.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--reg-error-mag", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the registration error magnitude map."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname with the mask of the target objects."},
        {.short_name = "-o", .long_name = "--out-reg-error-mag-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname for the resulting registration error magnitude."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, char **reg_mag_error_path, char **mask_path, char **out_path) {
    *reg_mag_error_path = iftGetStrValFromDict("--reg-error-mag", args);
    *mask_path = iftGetStrValFromDict("--mask", args);
    *out_path = iftGetStrValFromDict("--out-reg-error-mag-map", args);
    
    puts("--------------------");
    printf("- Reg. Error Mag Map: %s\n", *reg_mag_error_path);
    printf("- Mask: %s\n", *mask_path);
    printf("- Output Path: %s\n", *out_path);
    puts("--------------------\n");
}
/*************************************************************/


