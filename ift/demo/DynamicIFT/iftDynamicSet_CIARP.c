#include "ift.h"
#include "ift/segm/DynamicTrees.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(iftDict *args, char **img_path, char **markers_path, char **out_img_path, float *adj_rel_r);

void iftWriteDynamicSetMean(iftImage* img, iftImage* root, iftLabeledSet *seeds, const char* filename);
/*************************************************************/

int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *img_path      = NULL;
    char *markers_path  = NULL;
    char *out_img_path = NULL;

    float adj_rel_r;

    iftGetRequiredArgs(args, &img_path, &markers_path, &out_img_path, &adj_rel_r);

    timer *t1 = iftTic();

    puts("- Reading Input Image");
    iftImage *img = iftReadImageByExt(img_path);
    iftMImage *mimg = NULL;
    iftAdjRel *A = NULL;

    if(iftIs3DImage(img)) A = iftSpheric(adj_rel_r);
    else A = iftCircular(adj_rel_r);

    if (iftIsColorImage(img)){
        mimg = iftImageToMImage(img, LABNorm_CSPACE);
//        mimg = iftImageToMImage(img, RGB_CSPACE);
    } else {
        mimg = iftImageToMImage(img, GRAY_CSPACE);
    }

    puts("- Reading Markers");
    iftLabeledSet *seeds = iftReadSeeds(img, markers_path);

    puts("- Dynamic Sets");
    iftImage *mask = NULL, *aux_mask = NULL;

    mask = iftDynamicSetObjectPolicy(mimg, A, seeds);
    aux_mask = iftMask(img, mask);
    iftWriteImageByExt(aux_mask, "object.png");
    iftWriteImageByExt(mask, "mask_object.png");
//    iftDestroyImage(&mask);
    iftDestroyImage(&aux_mask);


    mask = iftDynamicSetRootPolicy(mimg, A, seeds);
    aux_mask = iftMask(img, mask);
    iftWriteImageByExt(aux_mask, "root.png");
    iftWriteImageByExt(mask, "mask_root.png");
    iftDestroyImage(&mask);
    iftDestroyImage(&aux_mask);


    mask = iftWaterCut(mimg, A, seeds, NULL);
    aux_mask = iftMask(img, mask);
    iftWriteImageByExt(aux_mask, "GCmax.png");
    iftWriteImageByExt(mask, "mask_GCmax.png");
    iftDestroyImage(&mask);
    iftDestroyImage(&aux_mask);

    mask = iftDynamicSetMinRootPolicy(mimg, A, seeds);
    aux_mask = iftMask(img, mask);
    iftWriteImageByExt(aux_mask, "minroot.png");
    iftWriteImageByExt(mask, "mask_minroot.png");
    iftDestroyImage(&mask);
    iftDestroyImage(&aux_mask);


    mask = iftGraphCutFromMImage(mimg, seeds, 100);
    aux_mask = iftMask(img, mask);
    iftWriteImageByExt(aux_mask, "GCsum.png");
    iftWriteImageByExt(mask, "mask_GCsum.png");
    iftDestroyImage(&mask);
    iftDestroyImage(&aux_mask);

//    puts("- Writing Image");
//    iftImage* aux_mask = iftMask(img, mask);
//    iftWriteImageByExt(aux_mask, out_img_path);
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(img_path);
    iftFree(markers_path);
    iftFree(out_img_path);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyLabeledSet(&seeds);
//    iftDestroyImage(&mask);
//    iftDestroyImage(&aux_mask);
    iftDestroyAdjRel(&A);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segmented an input image by the Improved IFT, which uses a new schema to estimate the arc-weights.\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input 2D Image to be segmented."},
            {.short_name = "-m", .long_name = "--markers", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Markers on input image."},
            {.short_name = "-o", .long_name = "--output-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Path of the Output Segmentation Mask."},
            {.short_name = "-a", .long_name = "--adjacency-relation", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Propagation adjacency relation."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict *args, char **img_path, char **markers_path, char **out_img_path, float *adj_rel_r) {
    *img_path      = iftGetStrValFromDict("--input-image", args);
    *markers_path  = iftGetStrValFromDict("--markers", args);
    *out_img_path = iftGetStrValFromDict("--output-mask", args);
    *adj_rel_r = (float) iftGetDblValFromDict("--adjacency-relation", args);

    char *parent_dir = iftParentDir(*out_img_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Markers: \"%s\"\n", *markers_path);
    printf("- Adjacency Relation Radius: \"%.3f\"\n", *adj_rel_r);
    printf("- Output Image: \"%s\"\n", *out_img_path);
    puts("-----------------------\n");
}
/*************************************************************/


void iftWriteDynamicSetMean(iftImage* img, iftImage* root, iftLabeledSet *seeds, const char* filename)
{

    iftImage *val = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iftImage *Cb = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iftImage *Cr = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iftImage *n = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iftImage* seed_img = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iftImage *out = iftCreateImageFromImage(img);

    for (int p = 0; p < img->n; p++) {
        val->val[p] = 0;
        Cb->val[p] = 0;
        Cr->val[p] = 0;
        n->val[p] = 0;
    }

    for (int p = 0; p < img->n; p++) {
        int r = root->val[p];
        val->val[r] += img->val[p];
        Cb->val[r] += img->Cb[p];
        Cr->val[r] += img->Cr[p];
        n->val[r] += 1;
    }

    for (int p = 0; p < img->n; p++) {
        int r = root->val[p];
        out->val[p] = val->val[r] / n->val[r];
        out->Cb[p] = Cb->val[r] / n->val[r];
        out->Cr[p] = Cr->val[r] / n->val[r];
    }


    iftSetImage(seed_img, -1);
    iftWriteSeedsOnImage(seed_img, seeds);

    iftColorTable *cmap = iftCreateColorTable(3, YCbCr_CSPACE);

    cmap->color[0].val[0] = 82;
    cmap->color[0].val[1] = 90;
    cmap->color[0].val[2] = 240;

    cmap->color[1].val[0] = 121;
    cmap->color[1].val[1] = 194;
    cmap->color[1].val[2] = 70;

    cmap->color[2].val[0] = 255;
    cmap->color[2].val[1] = 0;
    cmap->color[2].val[2] = 255;

    iftDrawSeeds(out, seed_img, cmap);

    iftWriteImageByExt(out, filename);

    iftDestroyImage(&val);
    iftDestroyImage(&Cb);
    iftDestroyImage(&Cr);
    iftDestroyImage(&n);
    iftDestroyImage(&out);
    iftDestroyImage(&seed_img);
    iftDestroyColorTable(&cmap);
}
