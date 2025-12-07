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
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, iftFileSet **train_img_set, int *n_atlases,
                        char **out_csv);
void iftValidateRequiredArgs(const char *test_img_path, const char *train_img_entry, int n_atlases, char *out_csv);
void iftGetOptionalArgs(  iftDict *args,   iftFileSet *train_img_set, iftFileSet **train_label_set);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path       = NULL;
    iftFileSet *train_img_set = NULL;
    int n_atlases;
    char *out_csv             = NULL;

    // optional args
    iftFileSet *train_label_set = NULL;
    

    iftGetRequiredArgs(args, &test_img_path, &train_img_set, &n_atlases, &out_csv);
    iftGetOptionalArgs(args, train_img_set, &train_label_set);


    timer *t1 = iftTic();

    puts("- Reading Testing Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);

    puts("- Atlas Selection");
    iftFileSet *chosen_atlas_set = iftAtlasSelectionByNMI(test_img, train_img_set, n_atlases, NULL, NULL, NULL);
    iftWriteFileSetAsCSV(chosen_atlas_set, out_csv);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftDestroyFileSet(&train_img_set);
    iftFree(out_csv);
    iftDestroyFileSet(&train_label_set);
    iftDestroyImage(&test_img);
    iftDestroyFileSet(&chosen_atlas_set);

    return 0;
}




/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Selects the most appropriate atlas to segment a testing image (using method [1]).\n" \
        "- Atlas selection is done in a set of (affine or deformable) registered atlas set on a reference space, " \
        "choosing the ones with higher Normalized Mutual Information with the registered test image.\n" \
        "- Input train atlases and test image must already have to be registered on a reference space.\n\n" \
        "[1] Aljabar, 2009 - Neuroimage - Multi-atlas based segmentation of brain images: atlas selection and its effect on accuracy";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="(Registered) Test Image to be segmented."},
        {.short_name = "-t", .long_name = "--train-img-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the (Registered) Train. Images or a CSV file with their pathnames."},
        {.short_name = "-n", .long_name = "--num-atlas", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Number of chosen/selected atlases."},
        {.short_name = "-o", .long_name = "--output-csv", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output CSV with the selected images/atlases."},
        {.short_name = "-l", .long_name = "--train-label-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Dir with the (Registered) Train. Label Images or a CSV file with their pathnames.\n" \
                                "Used to mask the input images before computing NMI."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, iftFileSet **train_img_set, int *n_atlases,
                        char **out_csv) {
    *test_img_path              = iftAbsPathname(iftGetConstStrValFromDict("--test-img", args));
    const char *train_img_entry = iftGetConstStrValFromDict("--train-img-entry", args);
    *n_atlases                  = iftGetLongValFromDict("--num-atlas", args);
    *out_csv                    = iftGetStrValFromDict("--output-csv", args);
    
    iftValidateRequiredArgs(*test_img_path, train_img_entry, *n_atlases, *out_csv);

    *train_img_set = iftLoadFileSetFromDirOrCSV(train_img_entry, 0, true);

    puts("-----------------------");
    printf("- Test Image: \"%s\"\n", *test_img_path);
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


void iftGetOptionalArgs(  iftDict *args,   iftFileSet *train_img_set, iftFileSet **train_label_set) {
    if (iftDictContainKey("--train-label-entry", args, NULL)) {
        const char *train_label_set_entry = iftGetConstStrValFromDict("--train-label-entry", args);
        iftFileSet *full_label_set = iftLoadFileSetFromDirOrCSV(train_label_set_entry, 0, true);

        *train_label_set = iftFilterFileSetByFilename(train_img_set, full_label_set);

        iftDestroyFileSet(&full_label_set);

        printf("- Train Label Image Entry: \"%s\"\n", train_label_set_entry);
    }
    else *train_label_set = NULL;

    puts("-----------------------\n");
}
/*************************************************************/











