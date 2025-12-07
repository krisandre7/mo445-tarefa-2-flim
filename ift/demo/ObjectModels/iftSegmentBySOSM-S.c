/**
 * @file
 * @brief Segments an Image on training reference image's space by the Statistical Multi-Object Shape Model SOSM-S.
 * @note See the source code in @ref iftSegmentBySOSM.c
 *
 * @example iftSegmentBySOSM-S.c
 * @brief Segments an Image on training reference image's space by the Statistical Multi-Object Shape Model SOSM-S.
 * @author Samuel Martins
 * @date Dec 28, 2016
 */



#include "ift.h"
#include "ift/medical/segm/SOSM-S.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **sosm_s_path, char **out_img_path);
void iftValidateRequiredArgs(const char *test_img_path, const char *sosm_s_path, const char *out_img_path);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path = NULL;
    char *sosm_s_path    = NULL;
    char *out_img_path  = NULL;
    
    iftGetRequiredArgs(args, &test_img_path, &sosm_s_path, &out_img_path); 

    timer *t1 = iftTic();

    puts("- Reading SOSM-S");
    iftSOSMS *sosm_s = iftReadSOSMS(sosm_s_path);

    puts("- Reading Test Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);
    
    puts("- Segmenting the Test Image by SOSM-S");
    iftImage *seg_img = iftSegmentBySOSMS(test_img, NULL, sosm_s);    
    iftWriteImageByExt(seg_img, out_img_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftFree(sosm_s_path);
    iftFree(out_img_path);
    iftDestroyImage(&test_img);
    iftDestroySOSMS(&sosm_s);
    iftDestroyImage(&seg_img);


    return 0;
}




/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segments an image on training reference image's space by the Statistical Multi-Object Shape Models SOSM-S [1].\n" \
        "- It applies an Object Location by MSPS translating the seed models over the test image gradient.\n" \
        "- Input Image must be registered on the reference image.\n\n" \
        "[1] Phellan, 2016 - Medical physics - Medical image segmentation via atlases and fuzzy object models";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image to be segmented. It might have to be previously registered on training " \
                                "reference image, or it will be in this program"},
        {.short_name = "-b", .long_name = "--sosm-s", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Object Shape Model Bank used for segmentation (*.zip)."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output Segmented Image."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **sosm_s_path, char **out_img_path) {
    *test_img_path = iftGetStrValFromDict("--test-img", args);
    *sosm_s_path    = iftGetStrValFromDict("--sosm-s", args);
    *out_img_path  = iftGetStrValFromDict("--output-img", args);

    iftValidateRequiredArgs(*test_img_path, *sosm_s_path, *out_img_path);


    puts("-----------------------");
    printf("- Test Image: \"%s\"\n", *test_img_path);
    printf("- SOSMS-S: \"%s\"\n", *sosm_s_path);
    printf("- Output Segmented Image: \"%s\"\n", *out_img_path);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *test_img_path, const char *sosm_s_path, const char *out_img_path) {
    // INPUT TEST IMAGE
    if (!iftIsImageFile(test_img_path))
        iftError("Invalid Input Test Image: \"%s\"", "iftValidateRequiredArgs", test_img_path);

    // MODEL
    if (iftFileExists(sosm_s_path)) {
        if (!iftEndsWith(sosm_s_path, ".zip")) {
            iftError("Invalid Extension for the Object Shape Model: \"%s\"... Try *.zip",
                     "iftValidateRequiredArgs", sosm_s_path);
        }
    }
    else iftError("Object Shape Model \"%s\" does not exist", "iftValidateRequiredArgs", sosm_s_path);

    // OUTPUT SEGMENTED IMAGE
    if (!iftIsImagePathnameValid(out_img_path)) {
        iftError("Invalid Output Grad Image's Pathname: \"%s\"", "iftValidateRequiredArgs", out_img_path);
    }

    char *parent_dir = iftParentDir(out_img_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    iftFree(parent_dir);
}

