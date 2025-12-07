#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    const char *csv_path = iftGetConstStrValFromDict("--input-label-image-set", args);
    const char *gt_dir = iftGetConstStrValFromDict("--ground-truth-dir", args);
    int xslice = iftGetLongValFromDict("--xslice", args);
    int yslice = iftGetLongValFromDict("--yslice", args);
    int zslice = iftGetLongValFromDict("--zslice", args);
    const char *out_prob_img_basename = iftGetConstStrValFromDict("--output-prob-error-image-basename", args);

    timer *t1 = iftTic();

    iftFileSet *label_img_set = iftLoadFileSetFromCSV(csv_path, 0);

    iftImage *aux = iftReadImageByExt(label_img_set->files[0]->path);
    iftImage *prob_errors = iftCreateImage(aux->xsize, aux->ysize, aux->zsize);
    iftSetCbCr(prob_errors, 128);
    iftDestroyImage(&aux);


    for (long i = 0; i < label_img_set->n; i++) {
        char *gt_path = iftJoinPathnames(2, gt_dir, iftFilename(label_img_set->files[i]->path, NULL));

        printf("[%ld/%ld]\nLabel Image: %s\nGT: %s\n\n", i+1, label_img_set->n, label_img_set->files[i]->path, gt_path);

        iftImage *label_img = iftReadImageByExt(label_img_set->files[i]->path);
        iftImage *gt = iftReadImageByExt(gt_path);
        
        for (int p = 0; p < label_img->n; p++)
            prob_errors->val[p] += (label_img->val[p] != gt->val[p]);

        iftFree(gt_path);
        iftDestroyImage(&label_img);
        iftDestroyImage(&gt);
    }

    
    iftColorTable *ctab = iftCreateHotIronColorTable(256);

    // averaging
    for (int p = 0; p < prob_errors->n; p++) {
        int val = ((int) (prob_errors->val[p] / ((float) label_img_set->n) * 255));
        iftColor YCbCr = iftRGBtoYCbCr(ctab->color[val], 255);

        prob_errors->val[p] = YCbCr.val[0];
        prob_errors->Cb[p] = YCbCr.val[1];
        prob_errors->Cr[p] = YCbCr.val[2];
    }

    iftImage *axial = iftGetXYSlice(prob_errors, zslice);
    iftImage *coronal = iftGetZXSlice(prob_errors, yslice);
    iftImage *sagittal = iftGetYZSlice(prob_errors, xslice);
    iftFree(prob_errors->Cb);
    prob_errors->Cb = NULL;
    iftFree(prob_errors->Cr);
    prob_errors->Cr = NULL;

    // iftWriteImageByExt(prob_errors, "%s_prob_errors.hdr", out_prob_img_basename);
    iftWriteImageByExt(axial, "%s_axial_%d.png", out_prob_img_basename, zslice);
    iftWriteImageByExt(coronal, "%s_coronal_%d.png", out_prob_img_basename, yslice);
    iftWriteImageByExt(sagittal, "%s_sagittal_%d.png", out_prob_img_basename, xslice);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Generate a probabilistic error image with the segmentation errors of a set of registered label images.\n" \
        "- A heat map is applied to output image. 3D color (axial, coronal, sagittal) slices are saved with the basename passed.\n" \
        "- A directory with the ground-truth images is passed. The images' filenames must be equal in the set " \
        "and in the directory.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-label-image-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Set of Label Images (*.csv)."},
        {.short_name = "-g", .long_name = "--ground-truth-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory with the Ground-Truth Segmentations."},
        {.short_name = "-x", .long_name = "--xslice", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="X Slice to get the sagittal slice and save the 3D color."},
        {.short_name = "-y", .long_name = "--yslice", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Y Slice to get the coronal slice and save the 3D color."},
        {.short_name = "-z", .long_name = "--zslice", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Z Slice to get the axial slice and save the 3D color."},
        {.short_name = "-o", .long_name = "--output-prob-error-image-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Probabilistic Error Image (*.npy)."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}
/*************************************************************/


