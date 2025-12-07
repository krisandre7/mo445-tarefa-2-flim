#include "ift.h"
#include "ift/medical/registration/Elastix.h"


/************************** HEADERS **************************/
iftDict *iftGetArguments(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **moving_img_set, char **fixed_img_path,
                        char **out_dir, iftFileSet **transf_files);
void iftGetOptionalArgs(  iftDict *args, bool *save_reg_imgs, bool *save_def_fields, int *n_threads);
int iftGetNumberOfThreads(  iftDict *args);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArguments(argc, argv);

   // required args
   iftFileSet *moving_img_set = NULL;
   char *fixed_img_path       = NULL;
   char *out_dir              = NULL;
   iftFileSet *transf_files   = NULL;

   // optional args
   bool save_reg_imgs;
   bool save_def_fields;
   int n_threads;

   iftGetRequiredArgs(args, &moving_img_set, &fixed_img_path, &out_dir, &transf_files);
   iftGetOptionalArgs(args, &save_reg_imgs, &save_def_fields, &n_threads);

   timer *t1 = iftTic();

   puts("- Reading Fixed Image");
   iftImage *fixed_img = iftReadImageByExt(fixed_img_path);



   // Registers all Images onto the Reference Space (Fixed Image)
   #pragma omp parallel for num_threads(n_threads)
   for (size_t i = 0; i < moving_img_set->n; i++) {
       const char *moving_img_path = moving_img_set->files[i]->path;
       printf("[%lu/%lu] %s\n", i+1, moving_img_set->n, moving_img_path);

       char *moving_img_filename = iftFilename(moving_img_path, iftFileExt(moving_img_path));
       char *out_basename        = iftJoinPathnames(2, out_dir, moving_img_filename);
       char *out_reg_img_path    = iftConcatStrings(2, out_basename, iftFileExt(moving_img_path));
       puts(out_basename);

       puts("- Reading Moving Image");
       iftImage *moving_img = iftReadImageByExt(moving_img_path);

       puts("- Register Image by Elastix");
       iftFileSet *def_fields = NULL;
       iftImage *reg_img = iftRegisterImageByElastix(moving_img, fixed_img, NULL, NULL, transf_files, out_basename, &def_fields);
       iftDestroyImage(&moving_img);

       if (iftDictContainKey("--apply-minmax-normalization-without-outliers", args, NULL)) {
           puts("- Apply MinMax Normalization Without Outliers");
           int max_range = iftMaxImageRange(iftImageDepth(fixed_img));
           iftImage *norm = iftNormalizeWithNoOutliers(reg_img, 0, max_range, 0.98);
           iftDestroyImage(&reg_img);

           reg_img = norm;
       }

       if (save_reg_imgs)
           iftWriteImageByExt(reg_img, out_reg_img_path);
       iftDestroyImage(&reg_img);

       if (!save_def_fields) {
           for (int i = 0; i < def_fields->n; i++)
               iftRemoveFile(def_fields->files[i]->path);
       }

       iftFree(moving_img_filename);
       iftFree(out_basename);
       iftFree(out_reg_img_path);
   }

   puts("\nDone...");
   puts(iftFormattedTime(iftCompTime(t1, iftToc())));

   // DESTROYERS
   iftDestroyDict(&args);
   iftDestroyFileSet(&moving_img_set);
   iftFree(fixed_img_path);
   iftFree(out_dir);
   iftDestroyFileSet(&transf_files);
   iftDestroyImage(&fixed_img);


    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = \
        "This program registers a set of Moving Images (from a directory or CSV file) on to a Fixed Image using the Elastix program.\n" \
        "Elastix parameter files can be found in http://elastix.bigr.nl/wiki/index.php/Parameter_file_database.\n\n" \
        "- At least ONE Elastix Transformation File must be passed, using the option --t0.\n" \
        "- It is possible to use until 3 Multiple Transformation Files for different registrations.\n" \
        "- For this, we must pass the parameters using --tx, where x is the increment value, starting at 0 until 2,"\
        "representing the order of the execution from the parameter files.\n" \
        "- Examples:\n" \
        "(1) iftRegisterImageByElastix -m movinging.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0042rigid.txt\n" \
        "(2) iftRegisterImageByElastix -m movinging.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0000affine.txt --t1 Par0000bspline.txt\n" \
        "(3) iftRegisterImageByElastix -m movinging.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0042rigid.txt --t1 Par0000affine.txt --t2 Par0000bspline.txt\n" \
        "* PS1: Be sure that your machine has enough memory to register multiple images in parallel, because it is an expensive process.\n" \
        "* PS2: To register only one image, use the program demo/Registration/iftRegisterImageByElastix.c\n";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-m", .long_name = "--moving-image-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory or a CSV file with the Moving Images to be registered on to the Fixed Image"},
        {.short_name = "-f", .long_name = "--fixed-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Reference (Fixed) Image where the Moving Image will be registered"},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Directory where the registered images and their deformations fields will be stored."},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Elastix Transformation File for Initial Transform.\n"\
                               "For multiple parameter files (up to 3), use the sequence of options: --t0, --t1, and --t2"},
        {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
        {.short_name = "", .long_name = "--t2", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation File."},
        {.short_name = "-n", .long_name = "--num-of-threads", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Number of Threads used to process to Register the images in parallel.\n" \
                                "Default: 1."},
        {.short_name = "-z", .long_name = "--apply-minmax-normalization-without-outliers", .has_arg=false,
         .required=false, .help="Apply a Min-Max Normalization without outliers (the brightest 2\\% voxels) " \
                                "in the registered image. The normalization range is [0, max_range(fixed image)]"},
        {.short_name = "", .long_name = "--no-save-reg-img", .has_arg=false,
         .required=false, .help="Flag to Not Save the Resulting Registered Image."},
        {.short_name = "", .long_name = "--no-save-def-fields", .has_arg=false,
         .required=false, .help="Flag to Not Save the Deformation Fields."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **moving_img_set, char **fixed_img_path,
                        char **out_dir, iftFileSet **transf_files) {
    const char *moving_img_entry = iftGetConstStrValFromDict("--moving-image-entry", args);
    *fixed_img_path              = iftGetStrValFromDict("--fixed-image", args);
    *out_dir                     = iftGetStrValFromDict("--output-dir", args);

    *moving_img_set = iftLoadFileSetFromDirOrCSV(moving_img_entry, 0, true);

    // output basename dir
    if (!iftDirExists(*out_dir))
        iftMakeDir(*out_dir);
    


// only incremental values for --tx, starting at 0 until 2, are considered (--t0, --t1, --t2)
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

    puts("-----------------------");
    printf("- Moving Image Entry: \"%s\"\n", moving_img_entry);
    printf("- Fixed Image: \"%s\"\n", *fixed_img_path);
    printf("- Output Directory: \"%s\"\n", *out_dir);
    printf("- Parameter Files:\n");
    for (size_t i = 0; i < (*transf_files)->n; i++)
        printf("[%lu] %s\n", i, (*transf_files)->files[i]->path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, bool *save_reg_imgs, bool *save_def_fields, int *n_threads) {
    *save_reg_imgs      = !iftDictContainKey("--no-save-reg-imgs", args, NULL);
    *save_def_fields    = !iftDictContainKey("--no-save-def-fields", args, NULL);

    *n_threads = iftGetNumberOfThreads(args);

    printf("- Save Reg Images: %s\n", IFT_PRINT_BOOL(*save_reg_imgs));
    printf("- Save Def Fields: %s\n", IFT_PRINT_BOOL(*save_def_fields));
    printf("- Number of Threads: %d\n", *n_threads);
    puts("-----------------------\n");
}


int iftGetNumberOfThreads(  iftDict *args) {
    // int n_threads = omp_get_max_threads(); // default
    int n_threads = 1; // default
    if (iftDictContainKey("--num-of-threads", args, NULL)) {
        const char *threads = iftGetConstStrValFromDict("--num-of-threads", args);

        if (iftRegexMatch(threads, "^[0-9]+$")) { // check if passed value is a number
            n_threads = atoi(threads);
        }
        else iftError("Number of Threads: \"%s\" is not a valid number.", "iftGetNumberOfThreads", threads);
    }

    return n_threads;
}
/*************************************************************/
