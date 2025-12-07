//
// Created by azaelmsousa on 31/03/20.
//

#include "ift.h"

iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **segm_path, char **out_path, int *n_svoxels);

iftImage *iftComputingSupervoxel(iftImage *img, iftImage *segm, int num_svoxels)
{
    puts("--- Converting image to mimage on Lab Color Space");
    iftMImage *mimg = iftImageToMImage(img,GRAYNorm_CSPACE);

    iftImage *opened = iftOpenBin(segm,3.);

    int volume = 0;
    for (int p =0; p < img->n; p++)
        if (opened->val[p] > 0)
            volume++;

    int numinitseeds = iftRound(pow(iftMin(iftMin(img->xsize,img->ysize),img->zsize)/5.0,3));;

    mimg->

    int masksize=0;
    for (int p=0; p < segm->n; p++)
        if (segm->val[p])
            masksize++;

    numinitseeds = iftMin(masksize/2,numinitseeds);

    puts("--- Executing DISF");

    iftAdjRel *A = iftSpheric(1.0);
    iftImage *svoxels = iftDISF(mimg, A, numinitseeds, num_svoxels, opened);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&opened);

    return svoxels;
}

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArgs(argc, argv);

    timer *t1,*t2;
    char *img_path         = NULL;
    char *segm_path        = NULL;
    char *out_path         = NULL;
    int n_svoxels          = 0;

    iftGetRequiredArgs(args,&img_path,&segm_path,&out_path,&n_svoxels);

    iftImage *img = iftReadImageByExt(img_path);
    iftImage *segm = iftReadImageByExt(segm_path);

    puts("- Volume Filtering");
    iftAdjRel *A = iftSpheric(sqrt(3.0));
    iftImage *objs = iftLabelComp(segm,A);
    iftDestroyAdjRel(&A);
    iftFloatArray *arr = iftAreaVolumeOfLabels(objs);
    float max = iftMaxFloatArray(arr->val,arr->n);
    iftSet *s = NULL;
    for (int i = 1; i < arr->n; i++){
        if (arr->val[i] < 0.2*max){
            iftInsertSet(&s,i);
        }
    }
    iftIntArray *removal_candidates = iftSetToArray(s);
    iftDestroySet(&s);
    iftDestroyFloatArray(&arr);
    iftDestroyImage(&segm);
    segm = iftRemoveLabels(objs,removal_candidates);
    iftDestroyImage(&objs);
    iftDestroyIntArray(&removal_candidates);

    t1 = iftTic();

    puts("- Normalizing image");
    int Imax = 4096;
    iftImage *norm = iftNormalize(img,0,Imax);
    iftDestroyImage(&img);

    puts("- Generating Supervoxels");
    iftImage *svoxels = iftComputingSupervoxel(norm,segm,n_svoxels);

    puts("- Saving Supervoxels");
    iftWriteImageByExt(svoxels,out_path);
    iftDestroyImage(&norm);
    iftDestroyImage(&svoxels);
    iftDestroyImage(&segm);

    t2 = iftToc();

    puts("\nDone...");
    fprintf(stdout,"Parenchyma Patch Extraction took ");
    puts(iftFormattedTime(iftCompTime(t1,t2)));

    return 0;

}

iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- iftAnomalyExtractionBySupervoxel. Detect anomalies located at the pulmonary parenchyma by means of a Supervoxel approach.\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Image."},
            {.short_name = "-l", .long_name = "--segm-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Segmentation Image."},
            {.short_name = "-n", .long_name = "--number-svoxels", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Number of supervoxels (e.g. 3000)."},
            {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output Image."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **segm_path, char **out_path, int *n_svoxels)
{
    *img_path        = iftGetStrValFromDict("--input-img", args);
    *segm_path       = iftGetStrValFromDict("--segm-img", args);
    *out_path        = iftGetStrValFromDict("--output-img", args);
    *n_svoxels       = iftGetDblValFromDict("--number-svoxels", args);

    if (!iftIsImageFile(*img_path)){
        iftError("Invalid Input Image: \"%s\"", "iftValidateRequiredArgs", img_path);
    }

    puts("");
    puts("--------------------------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Segmentation Image: \"%s\"\n", *segm_path);
    printf("- Number of supervoxels: \"%d\"\n", *n_svoxels);
    printf("- Output Image: \"%s\"\n", *out_path);
    puts("--------------------------------------");
    puts("");
}