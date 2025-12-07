
#include "ift.h"
#include "ift/core/tools/OS.h"
#include "ift/medical/segm/MALF.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, iftFileSet **train_img_set,
                        iftFileSet **train_atlas_dict, char **out_img_path);
void iftGetOptionalArgs(  iftDict *args, iftFileSet **transf_files);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path       = NULL;
    iftFileSet *train_img_set = NULL;
    iftFileSet *train_atlas_set    = NULL;
    char *out_img_path        = NULL;

    // optional args
    iftFileSet *transf_files = NULL;

    iftGetRequiredArgs(args, &test_img_path, &train_img_set, &train_atlas_set, &out_img_path);
    iftGetOptionalArgs(args, &transf_files);

    timer *t1 = iftTic();

    char *tmpdir = NULL;

    if (!iftDictContainKey("--skip-registration", args, NULL)) {
        tmpdir = iftMakeTempDir("tmpdir_", NULL, NULL);

        char *train_img_csv = iftJoinPathnames(2, tmpdir, "train_img_set.csv");
        iftWriteFileSetAsCSV(train_img_set, train_img_csv);
        char program_args[IFT_STR_DEFAULT_SIZE];
        sprintf(program_args, "-m %s -f %s -o %s -n 4 --no-save-reg-img", train_img_csv, test_img_path, tmpdir);
        for (int i = 0; i < transf_files->n; i++)
            sprintf(program_args, "%s --t%d %s", program_args, i, transf_files->files[i]->path);
        iftRunProgram("iftRegisterImageSetByElastix", program_args);

        char *train_atlas_csv = iftJoinPathnames(2, tmpdir, "train_atlas_set.csv");
        iftWriteFileSetAsCSV(train_atlas_set, train_atlas_csv);
        sprintf(program_args, "-i %s -t %s -o %s -n 4", train_atlas_csv, tmpdir, tmpdir);
        iftRunProgram("iftTransformImageSetByTransformix", program_args);

        for (int f = 0; f < train_atlas_set->n; f++) {
            char *aux = train_atlas_set->files[f]->path;
            train_atlas_set->files[f]->path = iftJoinPathnames(2, tmpdir, iftFilename(train_atlas_set->files[f]->path, NULL));
            iftFree(aux);
        }
    }

    puts("- Segmenting by MALF-STAPLE");
    iftImage *test_img = iftReadImageByExt(test_img_path);
    iftImage *seg_img = iftSegmentByMALFSTAPLE(test_img, train_atlas_set);
    iftWriteImageByExt(seg_img, out_img_path);

    iftRemoveDir(tmpdir);
    iftFree(tmpdir);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftDestroyFileSet(&train_img_set);
    iftDestroyFileSet(&train_atlas_set);
    iftDestroyFileSet(&transf_files);
    iftDestroyImage(&test_img);
    iftDestroyImage(&seg_img);

    return 0;
}




/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segments an image by MALF with Atlas Selection and STAPLE for Label Fusion.\n" \
        "PS: STAPLE requires the software CRKIT. See for instructions in INSTALLING_CRKIT.txt\n" \
        "--> COMPILE THE PROGRAMS:\n" \
        "iftRegisterImageSetByElastix\n" \
        "iftTransformImageSetByTransformix";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image to be segmented."},
        {.short_name = "-t", .long_name = "--train-img-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Train. Images or a CSV file with their pathnames."},
        {.short_name = "-l", .long_name = "--train-label-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Train. Label Images or a CSV file with their pathnames."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output Segmented Image."},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File for Initial Transform.\n"\
                               "For multiple parameter files (up to 3), use the sequence of options: --t0, --t1, and --t2"},
        {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
        {.short_name = "", .long_name = "--t2", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
        {.short_name = "", .long_name = "--skip-registration", .has_arg=false,
         .required=false, .help="Skip Registration: The Images are already registered."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, iftFileSet **train_img_set,
                        iftFileSet **train_atlas_set, char **out_img_path) {
    *test_img_path                = iftGetStrValFromDict("--test-img", args);
    const char *train_img_entry   = iftGetConstStrValFromDict("--train-img-entry", args);
    const char *train_atlas_entry = iftGetStrValFromDict("--train-label-entry", args);
    *out_img_path                 = iftGetStrValFromDict("--output-img", args);
    
    *train_img_set = iftLoadFileSetFromDirOrCSV(train_img_entry, 0, true);
    *train_atlas_set = iftLoadFileSetFromDirOrCSV(train_atlas_entry, 0, true);

    puts("-----------------------");
    printf("- Test Image: \"%s\"\n", *test_img_path);
    printf("- Train Image Entry: \"%s\"\n", train_img_entry);
    printf("- Train Atlas Image Entry: \"%s\"\n", train_atlas_entry);
    printf("- Output Segmented Image: \"%s\"\n", *out_img_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, iftFileSet **transf_files) {
    iftSList *SL = iftCreateSList();
    int i = 0;
    char opt[16];
    sprintf(opt, "--t%d", i);
    while (iftDictContainKey(opt, args, NULL)) {
        iftInsertSListIntoTail(SL, iftGetStrValFromDict(opt, args));
        i++;
        sprintf(opt, "--t%d", i);
    }

    *transf_files = iftCreateFileSet(SL->n);
    for (size_t i = 0; i < (*transf_files)->n; i++) {
        char *path = iftRemoveSListHead(SL);
        (*transf_files)->files[i] = iftCreateFile(path);
        iftFree(path);
    }
    iftDestroySList(&SL);

    printf("- Parameter Files:\n");
    for (size_t i = 0; i < (*transf_files)->n; i++)
        printf("[%lu] %s\n", i, (*transf_files)->files[i]->path);
    puts("-----------------------\n");
}


/*************************************************************/











