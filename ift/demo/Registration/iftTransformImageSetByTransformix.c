#include "ift.h"
#include "ift/medical/registration/Elastix.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, char **def_fields_dir, char **out_dir);
void iftGetOptionalArgs(  iftDict *args, int *n_threads, bool *copy_non_mapped_files);
int iftGetNumberOfThreads(  iftDict *args);
iftDict *iftBuildDefFieldDict(const char *def_fields_dir, const char *def_fields_suffix);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *img_set  = NULL;
    char *def_fields_dir = NULL;
    char *out_dir        = NULL;

    // optional args
    int n_threads;
    bool copy_non_mapped_files;

    iftGetRequiredArgs(args, &img_set, &def_fields_dir, &out_dir);
    iftGetOptionalArgs(args, &n_threads, &copy_non_mapped_files);

    timer *t1 = iftTic();

    
    #pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < img_set->n; i++) {
        const char *img_path = img_set->files[i]->path;
        printf("- Thread %d: [%lu/%lu]: %s\n", omp_get_thread_num(), i+1, img_set->n, img_path);

        char *filename_without_ext = iftFilename(img_path, iftFileExt(img_path));
        char *regex                = iftConcatStrings(2, filename_without_ext, "\\.*\\.[0-9]*\\.txt");

        iftFileSet *def_fields_set = iftLoadFileSetFromDirByRegex(def_fields_dir, regex, true);

        if (def_fields_set->n == 0)
            iftWarning("Deformation field for image %s not found", "main", img_path);
        else {
            const char *def_fields_path = def_fields_set->files[def_fields_set->n-1]->path;
            char *filename     = iftFilename(img_path, NULL);
            char *out_img_path = iftJoinPathnames(2, out_dir, filename);

            iftImage *img        = iftReadImageByExt(img_path);
            iftImage *mapped_img = iftTransformImageByTransformix(img, def_fields_path);        
            iftWriteImageByExt(mapped_img, out_img_path);

            iftFree(filename);
            iftFree(out_img_path);
        }
        
        iftFree(filename_without_ext);
        iftFree(regex);
        iftDestroyFileSet(&def_fields_set);
    }


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftFree(def_fields_dir);
    iftFree(out_dir);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program transforms/maps a set of Images or Label Images to a given Coordinate Space " \
        "using the program Transformix.\n" \
        "- The program uses the Deformation Fields obtained via Elastix program to transform the Images "\
        "to the Target Coordinate Space.\n" \
        "- It automatically finds out the last deformation field for each input image, which must have the same basename.\n"
        "E.g:\n" \
        "- Image ==> \"./labels/000010_000001.scn\"\n" \
        "- \"./def_fields/000010_000001_DefFields.0.txt\"\n" \
        "- \"./def_fields/000010_000001_DefFields.1.txt\"\n" \
        "- \"./def_fields/000010_000001_DefFields.2.txt\"\n" \
        "It will consider the last deformation field file: ./def_fields/000010_000001_DefFields.2.txt\n\n" \
        "- To obtain the Deformation Field files, run the program iftRegisterImageSetByElastix.c\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--img-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Directory or a CSV file with the (Label) Images to be mapped."},
            {.short_name = "-t", .long_name = "--def-fields-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Directory with Deformation Fields Files"},
            {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Output Directory where the mapped images will be stored."},
            {.short_name = "-c", .long_name = "--copy-non-mapped-files", .has_arg=false,
             .required=false, .help="If passed, it copies the Image Files that are not mapped (Image files that do not have Deformation Files)"},
            {.short_name = "-n", .long_name = "--num-of-threads", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=false, .help="Number of Threads used in the Transformation.\n" \
             "Default: 1.\n"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, char **def_fields_dir, char **out_dir) {
    const char *img_entry = iftGetStrValFromDict("--img-entry", args);
    *def_fields_dir       = iftGetStrValFromDict("--def-fields-dir", args);
    *out_dir              = iftGetStrValFromDict("--output-dir", args);

    *img_set = iftLoadFileSetFromDirOrCSV(img_entry, 0, true);

    if (!iftDirExists(*out_dir))
        iftMakeDir(*out_dir);

    puts("-----------------------");
    printf("- Image Entry: \"%s\"\n", img_entry);
    printf("- Def. Fields Dir: \"%s\"\n", *def_fields_dir);
    printf("- Output Dir: \"%s\"\n", *out_dir);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, int *n_threads, bool *copy_non_mapped_files) {
    *n_threads             = iftGetNumberOfThreads(args);
    *copy_non_mapped_files = iftDictContainKey("--copy-non-mapped-files", args, NULL);

    printf("- Number of Threads: %d\n", *n_threads);
    printf("- Copy Non-Mapped Files: %s\n", IFT_PRINT_BOOL(*copy_non_mapped_files));
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


iftDict *iftBuildDefFieldDict(const char *def_fields_dir, const char *def_fields_suffix) {
   // DEF. FIELDS
    iftFileSet *def_fields = iftLoadFileSetFromDirOrCSV(def_fields_dir, 0, true);
    if (def_fields->n == 0)
        iftError("There are no Def. Fields (no files) from \"%s\"", "iftBuildDefFieldDict", def_fields_dir);

    iftDict *dict = iftCreateDictWithApproxSize(def_fields->n);

    for (long i = 0; i < def_fields->n; i++) {
        const char *path = def_fields->files[i]->path;

        if (!iftFileExists(path)) {
            printf("- Not found Def. Fields \"%s\"\n", path);
        }
        else {
            char *filename = iftFilename(path, NULL);

            if (iftRegexMatch(filename, "^.+%s$", def_fields_suffix)) {
                char *key = iftSplitStringAt(filename, def_fields_suffix, 0); // get the image basename (file key)
                iftInsertIntoDict(key, path, dict);
                iftFree(key);
            }
            // else fprintf(stderr, "Skipping Invalid Def. Field Pathname: \"%s\"\nTry *%s\n\n",
            //                 path, def_fields_suffix);
            iftFree(filename);
        }
    }
    if (iftIsDictEmpty(dict))
        iftError("No Def. Fields Files were found from \"%s\"", "iftBuildDefFieldDict", def_fields_dir);


    return dict;
}















