#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **adapro_path, char **out_dir);
iftImage *iftNativeAdaProSegmentation(  iftImage *img, iftAdaPro *adapro, const char *out_dir, iftImage **img_out);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *img_path = NULL;
    char *adapro_path = NULL;
    char *out_dir = NULL;
    
    iftGetRequiredArgs(args, &img_path, &adapro_path, &out_dir);
    
    timer *t1 = iftTic();

    iftImage *img = iftReadImageByExt(img_path);
    iftAdaPro *adapro = iftReadAdaPro(adapro_path);
    if (adapro->labels->n != 2)
        iftError("Number of objects in AdaPro %ld is != 2", "main", adapro->labels->n);
    
    puts("- Run AdaPro Segmentation");
    iftImage *img_pre = NULL;
    iftImage *mask = iftNativeAdaProSegmentation(img, adapro, out_dir, &img_pre);
    
    iftWriteAdaPro(adapro, iftJoinPathnames(2, out_dir, "adapro.zip"));
    iftWriteImageByExt(img_pre, iftJoinPathnames(2, out_dir, "image.nii.gz"));
    iftWriteImageByExt(mask, iftJoinPathnames(2, out_dir, "mask.nii.gz"));

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    
    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyAdaPro(&adapro);
    iftDestroyImage(&img_pre);
    iftDestroyImage(&mask);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segment a Test Image on NATIVE space by AdaPro for Asymmetry Analysis.\n" \
        "- It saves in the output directory: the pre-processed test image, registered adapro and template on native space, " \
        "and the resulting segmentation mask.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Test Image."},
        {.short_name = "-m", .long_name = "--adapro", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="AdaPro model used for segmentation (*.zip)."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output directory."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **adapro_path, char **out_dir) {
    *img_path = iftGetStrValFromDict("--test-image", args);
    *adapro_path = iftGetStrValFromDict("--adapro", args);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    if (!iftDirExists(*out_dir))
        iftMakeDir(*out_dir);
    
    puts("--------------------------");
    printf("- Test Image: %s\n", *img_path);
    printf("- AdaPro: %s\n", *adapro_path);
    printf("- Output Dir: %s\n", *out_dir);
    puts("--------------------------\n");
}


iftImage *iftNativeAdaProSegmentation(  iftImage *img, iftAdaPro *adapro, const char *out_dir, iftImage **img_out) {
    iftFileSet *elastix_files = iftWriteAdaProElastixParamFiles(adapro, NULL);
    
    iftRegisterAdaProOnTestImageByElastix(adapro, img, elastix_files, NULL);
    
    iftImage *rough_mask_dilated = iftDilateAdaProRoughSegmentation(adapro);
    iftImage *img_pre = iftBrainMRIPreProcessing(img, 12, NULL, adapro->template_img, rough_mask_dilated, true, true, true, false, NULL);
    iftImage *mask = iftSegmentByAdaPro(img_pre, adapro, NULL);
    
    iftRemoveFileSet(elastix_files);
    iftDestroyFileSet(&elastix_files);
    iftDestroyImage(&rough_mask_dilated);
    
    *img_out = img_pre;
    
    return mask;
}
/*************************************************************/


