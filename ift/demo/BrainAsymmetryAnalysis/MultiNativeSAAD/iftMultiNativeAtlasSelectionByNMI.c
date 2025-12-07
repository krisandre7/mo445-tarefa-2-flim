/**
 * @file
 * @brief Selects the most appropriate atlas to segment a testing image.
 * @note See the source code in @ref iftAtlasSelection.c
 *
 * @example iftAtlasSelection.c
 * @brief Selects the most appropriate atlas to segment a testing image.
 * @author Samuel Martins
 * @date Jan 27, 2017
 */



#include "ift.h"
#include "ift/medical/segm/MALF.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **template_img_path, iftFileSet **train_img_set,
                        int *n_atlases, char **out_csv);
void iftValidateRequiredArgs(const char *test_img_path, const char *train_img_entry, int n_atlases, char *out_csv);
void iftGetOptionalArgs(  iftDict *args, char **test_mask_path, iftFileSet **train_mask_set, iftFileSet **elastix_params);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path = NULL;
    char *template_img_path = NULL;
    iftFileSet *train_img_set = NULL;
    int n_atlases;
    char *out_csv = NULL;
    // optional args
    char *test_mask_path = NULL;
    iftFileSet *train_mask_set = NULL;
    iftFileSet *elastix_params = NULL;

    iftGetRequiredArgs(args, &test_img_path, &template_img_path, &train_img_set, &n_atlases, &out_csv);
    iftGetOptionalArgs(args, &test_mask_path, &train_mask_set, &elastix_params);
    
    timer *t1 = iftTic();

    puts("- Reading Testing Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);
    iftImage *test_mask = (test_mask_path) ? iftReadImageByExt(test_mask_path) : NULL;
    
    if (elastix_params) {
        puts("- Registration");
        iftImage *aux = test_img;
        iftImage *template_img = iftReadImageByExt(template_img_path);
    
        iftFileSet *df = NULL;
        test_img = iftRegisterImageByElastix(test_img, template_img, NULL, NULL, elastix_params, NULL, &df);
        iftDestroyImage(&template_img);
        iftDestroyImage(&aux);
        
        if (test_mask) {
            aux = test_mask;
            test_mask = iftTransformImageByTransformix(test_mask, df->files[df->n - 1]->path);
            iftDestroyImage(&aux);
        }
        iftRemoveFileSet(df);
        iftDestroyFileSet(&df);
    }
    
    puts("- Atlas Selection");
    iftFileSet *chosen_atlas_set = iftAtlasSelectionByNMI(test_img, train_img_set, n_atlases, test_mask,
                                                          train_mask_set, NULL);
    iftWriteFileSetAsCSV(chosen_atlas_set, out_csv);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftFree(template_img_path);
    iftDestroyFileSet(&train_img_set);
    iftFree(out_csv);
    iftFree(test_mask_path);
    iftDestroyFileSet(&train_mask_set);
    
    iftDestroyImage(&test_img);
    iftDestroyImage(&test_mask);
    iftDestroyFileSet(&chosen_atlas_set);
    iftDestroyFileSet(&train_img_set);
    iftDestroyFileSet(&elastix_params);

    return 0;
}




/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Selects the most appropriate atlas to segment a testing image (using method [1]).\n" \
        "- Atlas selection is done in a set of (affine or deformable) registered atlas set on a reference space, " \
        "choosing the ones with higher Normalized Mutual Information with the test image, which can be registered by this program.\n" \
        "- Input train atlases and test image must already have to be registered on a reference space.\n" \
        "- If masks are passed for the training images and testing image, they are masked using their corresponding mask " \
        "before normalized mutual information."
        "[1] Aljabar, 2009 - Neuroimage - Multi-atlas based segmentation of brain images: atlas selection and its effect on accuracy";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image on its own NATIVE space."},
        {.short_name = "-r", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template (Reference Image)."},
        {.short_name = "-t", .long_name = "--train-img-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Registered Train. Images or a CSV file with their pathnames."},
        {.short_name = "-n", .long_name = "--num-atlases", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Number of chosen/selected atlases."},
        {.short_name = "-o", .long_name = "--output-csv", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output CSV with the selected images/atlases."},
        {.short_name = "-m", .long_name = "--test-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Testing Mask."},
        {.short_name = "-l", .long_name = "--train-mask-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Set of masks for the training set. The pathnames must be in the same order of the training images."},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File for Initial Transform.\n"\
                               "For multiple parameter files (up to 3), use the sequence of options: --t0, --t1, and --t2"},
        {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **template_img_path, iftFileSet **train_img_set,
                        int *n_atlases, char **out_csv) {
    *test_img_path              = iftAbsPathname(iftGetConstStrValFromDict("--test-img", args));
    *template_img_path          = iftGetStrValFromDict("--template", args);
    const char *train_img_entry = iftGetConstStrValFromDict("--train-img-entry", args);
    *n_atlases                  = iftGetLongValFromDict("--num-atlases", args);
    *out_csv                    = iftGetStrValFromDict("--output-csv", args);
    
    iftValidateRequiredArgs(*test_img_path, train_img_entry, *n_atlases, *out_csv);

    *train_img_set = iftLoadFileSetFromDirOrCSV(train_img_entry, 0, true);

    puts("-----------------------");
    printf("- Test Image: \"%s\"\n", *test_img_path);
    printf("- Template Image: \"%s\"\n", *template_img_path);
    printf("- Train Image Entry: \"%s\"\n", train_img_entry);
    printf("- Number of selected/chosen atlases: %d\n", *n_atlases);
    printf("- Output CSV: \"%s\"\n", *out_csv);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *test_img_path, const char *train_img_entry, int n_atlases, char *out_csv) {
    // INPUT TEST IMAGE
    if (!iftIsImageFile(test_img_path))
        iftError("Invalid Input Test Image: \"%s\"", "iftValidateRequiredArgs", test_img_path);


    // LABEL IMAGE ENTRY
    if (iftFileExists(train_img_entry)) {
        if (!iftEndsWith(train_img_entry, ".csv")) {
            iftError("Train Image Set Entry File \"%s\" is not a CSV file... Try *.csv", "iftValidateRequiredArgs", train_img_entry);
        }
    }
    else if (!iftDirExists(train_img_entry))
        iftError("Invalid Pathname for the Train Image Set Entry: \"%s\"...\nTry a valid Directory or a CSV file",
                 "iftValidateRequiredArgs", train_img_entry);

    if (n_atlases <= 0)
        iftError("Invalid number of atlases: %d <= 0", "iftValidateRequiredArgs", n_atlases);

    // OUTPUT SEGMENTED IMAGE
    if (!iftEndsWith(out_csv, ".csv")) {
        iftError("Output CSV: \"%s\"", "iftValidateRequiredArgs", out_csv);
    }

    char *parent_dir = iftParentDir(out_csv);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
}


void iftGetOptionalArgs(  iftDict *args, char **test_mask_path, iftFileSet **train_mask_set, iftFileSet **elastix_params) {
    if (iftDictContainKey("--test-mask", args, NULL)) {
        *test_mask_path = iftGetStrValFromDict("--test-mask", args);
        printf("- Test Mask: \"%s\"\n", *test_mask_path);
    }
    else *test_mask_path = NULL;
    
    if (iftDictContainKey("--train-mask-set", args, NULL)) {
        const char *train_mask_set_entry = iftGetConstStrValFromDict("--train-mask-set", args);
        *train_mask_set = iftLoadFileSetFromDirOrCSV(train_mask_set_entry, 0, true);
        printf("- Train Mask Set: \"%s\"\n", train_mask_set_entry);
    }
    else *train_mask_set = NULL;
    
    // only incremental values for --tx, starting at 0 until 2, are considered (--t0, --t1, --t2)
    if (iftDictContainKey("--t0", args, NULL)) {
        iftSList *SL = iftCreateSList();
        int i = 0;
        char opt[16];
        sprintf(opt, "--t%d", i);
        while (iftDictContainKey(opt, args, NULL)) {
            iftInsertSListIntoTail(SL, iftGetStrValFromDict(opt, args));
            i++;
            sprintf(opt, "--t%d", i);
        }
        
        *elastix_params = iftCreateFileSet(SL->n);
        for (long i = 0; i < (*elastix_params)->n; i++) {
            char *path = iftRemoveSListHead(SL);
            (*elastix_params)->files[i] = iftCreateFile(path);
            iftFree(path);
        }
        iftDestroySList(&SL);
        
        
        printf("- Elastix Params:\n");
        for (long f = 0; f < (*elastix_params)->n; f++)
            printf("[%lu] %s\n", f, (*elastix_params)->files[f]->path);
    }
    else *elastix_params = NULL;
    
    puts("-----------------------\n");
}
/*************************************************************/











