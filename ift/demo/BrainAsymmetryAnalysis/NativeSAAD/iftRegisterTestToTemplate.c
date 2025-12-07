#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **template_img_path, char **out_dir,
                        iftFileSet **elastix_params);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *test_img_path = NULL;
    char *template_img_path = NULL;
    char *out_dir = NULL;
    iftFileSet *elastix_params = NULL;
    
    iftGetRequiredArgs(args, &test_img_path, &template_img_path, &out_dir, &elastix_params);

    timer *t1 = iftTic();
    
    iftImage *test_img = iftReadImageByExt(test_img_path);
    iftImage *template_img = iftReadImageByExt(template_img_path);
    
    char *img_id = iftFilename(test_img_path, iftFileExt(test_img_path));
    char *df_basename = iftJoinPathnames(3, out_dir, "NativeOnTemplate", img_id);
    
    iftFileSet *df = NULL;
    iftImage *test_reg = iftRegisterImageByElastix(test_img, template_img, NULL, NULL, elastix_params,
                                                   df_basename, &df);
    iftWriteImageByExt(test_reg, iftJoinPathnames(3, out_dir, "NativeOnTemplate", iftCopyString("%s.nii.gz", img_id)));


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&test_img);
    iftDestroyImage(&template_img);
    iftDestroyImage(&test_reg);
    iftDestroyFileSet(&df);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Register a testing image template to a template.\n" \
        "- It saves in the output directory: registered template image and its deformation fields";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Test Image."},
        {.short_name = "-r", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Template (Reference Image)."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output directory where the registered template image and its deformation files are saved."},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Elastix Transformation File for Initial Transform.\n"\
                               "For multiple parameter files (up to 2), use the sequence of options: --t0, --t1"},
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


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **template_img_path, char **out_dir,
                        iftFileSet **elastix_params) {
    *test_img_path = iftGetStrValFromDict("--test-image", args);
    *template_img_path = iftGetStrValFromDict("--template", args);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    
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
    
    puts("-----------------------------");
    printf("- Test Image: %s\n", *test_img_path);
    printf("- Template: %s\n", *template_img_path);
    printf("- Output Dir: %s\n", *out_dir);
    puts("-------------------------");
    printf("- Elastix Params:\n");
    for (long i = 0; i < (*elastix_params)->n; i++)
        printf("[%lu] %s\n", i, (*elastix_params)->files[i]->path);
    puts("-----------------------------\n");
}
/*************************************************************/


