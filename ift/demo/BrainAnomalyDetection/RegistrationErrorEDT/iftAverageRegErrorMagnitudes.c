#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, iftFileSet **reg_error_set, char **out_path,
                  char **mask_path, bool *use_stdev);

iftImage *iftAverageRegErrorMagnitudes(  iftFileSet *reg_error_set, bool use_stdev);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    iftFileSet *reg_error_set = NULL;
    char *out_path = NULL;
    char *mask_path = NULL;
    bool use_stdev = false;

    iftGetInputs(args, &reg_error_set, &out_path, &mask_path, &use_stdev);
    
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;

    timer *t1 = iftTic();

    puts("- Computing Mean Registration Error Magnitude");
    iftImage *mean_reg_error_mag = iftAverageRegErrorMagnitudes(reg_error_set, use_stdev);


    if (mask) {
        puts("****************************************");
        iftImage *aux = mean_reg_error_mag;
        mean_reg_error_mag = iftMask(mean_reg_error_mag, mask);
        iftDestroyImage(&aux);
    }

    puts("- Writing Mean Registration Error");
    iftWriteImageByExt(mean_reg_error_mag, out_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&reg_error_set);
    iftDestroyImage(&mask);
    iftDestroyImage(&mean_reg_error_mag);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Average a set of Registration Error Magnitudes from an image set.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--reg-error-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory of CSV with the registration error magnitudes."},
        {.short_name = "-o", .long_name = "--out-mean-reg-error", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname for the resulting mean registration error magnitude."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname with the mask of the target objects."},
        {.short_name = "", .long_name = "--use-stdev", .has_arg=false,
         .required=false, .help="Add/Use the Standard Deviation Registration Error Magnitudes into the output map."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, iftFileSet **reg_error_set, char **out_path,
                  char **mask_path, bool *use_stdev) {
    const char *reg_error_entry = iftGetConstStrValFromDict("--reg-error-set", args);
    *reg_error_set = iftLoadFileSetFromDirOrCSV(reg_error_entry, 0, true);
    *out_path = iftGetStrValFromDict("--out-mean-reg-error", args);

    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    *use_stdev = iftDictContainKey("--use-stdev", args, NULL);

    puts("-----------------------");
    printf("- Reg. Error Entry: %s\n", reg_error_entry);
    printf("- Output Mean Reg. Error Magnitude: %s\n", *out_path);
    puts("-----------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    printf("- Add/Use Standard Deviation: %s\n", iftBoolAsString(*use_stdev));
    puts("-----------------------\n");
}


iftImage *iftAverageRegErrorMagnitudes(  iftFileSet *reg_error_set, bool use_stdev) {
    iftImage *reg_error_0 = iftReadImageByExt(reg_error_set->files[0]->path);
    iftFImage *mean_reg_error_mag = iftCreateFImage(reg_error_0->xsize, reg_error_0->ysize, reg_error_0->zsize);
    iftDestroyImage(&reg_error_0);

    
    for (int i = 0; i < reg_error_set->n; i++) {
        iftImage *reg_error_mag = iftReadImageByExt(reg_error_set->files[i]->path);

        #pragma omp parallel for
        for (int p = 0; p < reg_error_mag->n; p++)
            mean_reg_error_mag->val[p] += (reg_error_mag->val[p] / ((float) reg_error_set->n));
        
        iftDestroyImage(&reg_error_mag);
    }

    if (use_stdev) {
        iftFImage *stdev_reg_error_mag = iftCreateFImage(mean_reg_error_mag->xsize, mean_reg_error_mag->ysize,
                                                         mean_reg_error_mag->zsize);
        
        for (int i = 0; i < reg_error_set->n; i++) {
            iftImage *reg_error_mag = iftReadImageByExt(reg_error_set->files[i]->path);

            #pragma omp parallel for
            for (int p = 0; p < reg_error_mag->n; p++)
                stdev_reg_error_mag->val[p] += powf(reg_error_mag->val[p] - mean_reg_error_mag->val[p], 2);
            
            iftDestroyImage(&reg_error_mag);
        }
        
        // adding stdev asymmetries to the mean asymmetries
        #pragma omp parallel for
        for (int p = 0; p < stdev_reg_error_mag->n; p++)
            mean_reg_error_mag->val[p] = mean_reg_error_mag->val[p] + (sqrtf(stdev_reg_error_mag->val[p] / reg_error_set->n));

        iftDestroyFImage(&stdev_reg_error_mag);
    }

    iftImage *rounded_mean_reg_error_mag = iftRoundFImage(mean_reg_error_mag);
    iftDestroyFImage(&mean_reg_error_mag);

    return rounded_mean_reg_error_mag;
}
/*************************************************************/


