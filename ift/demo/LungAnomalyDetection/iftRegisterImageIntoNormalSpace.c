//
// Created by azaelmsousa on 24/03/21.
//

#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArguments(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **input_img_path, char **normal_reference_spaces_path, char **out_path,
                        iftFileSet **transf_files);
void iftGetOptionalArgs(  iftDict *args, char **input_mask_path, char **normal_mask_dir_path);
/*************************************************************/

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArguments(argc, argv);

    // required args
    char *img_path                      = NULL;
    char *normal_reference_spaces_path  = NULL;
    char *out_path                      = NULL;
    iftFileSet *transf_files            = NULL;

    // optional args
    char *input_mask_path      = NULL;
    char *normal_mask_dir_path = NULL;

    iftGetRequiredArgs(args, &img_path, &normal_reference_spaces_path, &out_path, &transf_files);
    iftGetOptionalArgs(args, &input_mask_path, &normal_mask_dir_path);

    /* Variables */
    char filename[4095];
    puts("--- Reading data");
    iftImage *input_img    = iftReadImageByExt(img_path);
    iftImage *input_mask   = NULL;
    iftDataSet *Z          = iftReadDataSet(normal_reference_spaces_path);
    if (input_mask_path != NULL)
        input_mask = iftReadImageByExt(input_mask_path);

    puts("---- Registering image into best normal space");
    iftFloatArray *registration_errors = NULL;
    int best_reg_id;
    char *best_reg_img_path = NULL;
    char def_files_basename[4095];
    sprintf(def_files_basename,"%s/%s",out_path,iftFilename(img_path,iftFileExt(img_path)));
    iftImage *best_reg = iftRegisterImageByElastixIntoBestNormalSpace(input_img,input_mask,Z,normal_mask_dir_path,transf_files,
                                                                      def_files_basename,&best_reg_id,&best_reg_img_path,
                                                                      &registration_errors);

    /* Writing output files */
    puts("--- Writing output files");
    char *f = iftFilename(img_path,".nii.gz");
    puts("----- Writing registration errors");
    sprintf(filename,"%s/%s_registration_errors.npy",out_path,f);
    iftWriteFloatArray(registration_errors,filename);
    iftDestroyFloatArray(&registration_errors);

    puts("----- Writing reference image");
    iftDict *json = iftCreateDict();
    iftInsertIntoDict("reference-image", best_reg_img_path, json);
    iftInsertIntoDict("registration-error-id", best_reg_id, json);
    sprintf(filename,"%s/%s_reference_image.npy",out_path,f);
    iftWriteJson(json, filename);
    iftDestroyDict(&json);
    free(best_reg_img_path);

    puts("----- Writing registered image");
    sprintf(filename,"%s/%s.nii.gz",out_path,f);
    iftWriteImageByExt(best_reg,filename);
    iftDestroyImage(&best_reg);
    free(f);

    /* Deallocating memmory */
    puts("--- Deallocating memmory");
    iftDestroyImage(&input_img);
    iftDestroyImage(&input_mask);

    return 0;
}

/************************** SOURCES **************************/
iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = \
        "This program finds the best reference space of a test image in a given set of normal reference spaces. The external software ELASTIX is required.\n" \
        "- Elastix parameter files can be found in http://elastix.bigr.nl/wiki/index.php/Parameter_file_database.\n" \
        "- At least ONE Elastix Transformation File must be passed, using the option --t0.\n" \
        "- It is possible to use until 3 Multiple Transformation Files for different registrations.\n" \
        "- For this, we must pass the parameters using --tx, where x is the increment value, starting at 0 until 2,"\
        "representing the order of the execution from the parameter files.\n" \
        "- Examples:\n" \
        "(1) iftRegisterImageByElastix -m moving.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0042rigid.txt\n" \
        "(2) iftRegisterImageByElastix -m moving.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0000affine.txt --t1 Par0000bspline.txt\n" \
        "(3) iftRegisterImageByElastix -m moving.hdr -f fixed.hdr -d 12 -o reg.hdr --t0 Par0042rigid.txt --t1 Par0000affine.txt --t2 Par0000bspline.txt\n";


    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="(Moving) Image to be register on to the Fixed Image"},
            {.short_name = "-n", .long_name = "--normal-reference-spaces", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Reference (Fixed) Image where the Moving Image will be registered"},
            {.short_name = "-o", .long_name = "--output-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output basename (no extension). Note that this software generates a few different outputs."},
            {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Elastix Transformation File for Initial Transform.\n"\
                               "For multiple parameter files (up to 3), use the sequence of options: --t0, --t1, and --t2"},
            {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Elastix Transformation File."},
            {.short_name = "", .long_name = "--t2", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Elastix Transformation File."},
            {.short_name = "-v", .long_name = "--moving-label-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Mask image of the Moving Image. It only performs registration on " \
                                "the part of the image that is within the mask."},
            {.short_name = "-r", .long_name = "--normal-label-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Directory containing the label images of all normal images"},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **input_img_path, char **normal_reference_spaces_path, char **out_path,
                        iftFileSet **transf_files) {
    *input_img_path                = iftGetStrValFromDict("--input-image", args);
    *normal_reference_spaces_path  = iftGetStrValFromDict("--normal-reference-spaces", args);
    *out_path                      = iftGetStrValFromDict("--output-basename", args);

    // output basename dir
    char *parent_dir = iftParentDir(*out_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    free(parent_dir);

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
    printf("- Input Image: \"%s\"\n", *input_img_path);
    printf("- Normal Reference Spaces: \"%s\"\n", *normal_reference_spaces_path);
    printf("- Output Basename: \"%s\"\n", *out_path);
    printf("- Parameter Files:\n");
    for (size_t i = 0; i < (*transf_files)->n; i++)
        printf("[%lu] %s\n", i, (*transf_files)->files[i]->path);
    puts("-----------------------");
}



void iftGetOptionalArgs(  iftDict *args, char **input_mask_path, char **normal_mask_dir_path) {
    if (iftDictContainKey("--moving-label-mask", args, NULL)) {
        *input_mask_path = iftGetStrValFromDict("--moving-label-mask", args);
        printf("- Moving Mask Image: \"%s\"\n", *input_mask_path);
    }
    else *input_mask_path = NULL;

    if (iftDictContainKey("--fixed-label-mask", args, NULL)) {
        *normal_mask_dir_path = iftGetStrValFromDict("--normal-label-dir", args);
        printf("- Fixed Mask Image: \"%s\"\n", *normal_mask_dir_path);
    }
    else *normal_mask_dir_path = NULL;

    puts("-----------------------\n");
}
/*************************************************************/
