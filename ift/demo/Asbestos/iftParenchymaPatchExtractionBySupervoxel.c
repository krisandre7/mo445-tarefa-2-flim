//
// Created by azaelmsousa on 24/03/20.
//

#include "ift.h"

int iftGetPatientID(const char *name)
{
    char *filename = iftFilename(name,"");
    char *ptr = strchr(filename,'_');
    char str[20];
    if (ptr == NULL)
        strcpy(str,filename);
    else
        strncpy(str,filename,ptr-filename);
    return atoi(str);
}

iftImage *iftComputingSupervoxel(iftImage *img, iftImage *segm, float radius, int n_samples)
{
    puts("--- Converting image to mimage");
    iftMImage *mimg = iftImageToMImage(img,GRAYNorm_CSPACE);

    puts("--- Selecting seeds by grid sampling");
    iftIntArray *arr = iftGridSamplingOnMask(segm,radius,-1,n_samples);
    iftImage *seeds = iftCreateImageFromImage(img);
    iftIntArrayToImage(arr,seeds,1);
    iftDestroyIntArray(&arr);

    iftAdjRel *A = iftSpheric(1.0);

    puts("--- Executing ISF");
    iftIGraph *igraph = iftImplicitIGraph(mimg, segm, A);
    iftIGraphISF_Root(igraph, seeds, 0.08, 3, 10);
    iftDestroyMImage(&mimg);
    iftDestroyImage(&seeds);
    iftDestroyAdjRel(&A);

    puts("--- Generating Supervoxels");
    iftImage *svoxels = iftIGraphLabel(igraph);
    iftDestroyIGraph(&igraph);

    return svoxels;
}

float iftComputeOverlapWithAnnotation(iftImage *annotation, iftBoundingBox bb)
{
    float score = 0.0;
    iftVoxel v;
    int p;

    for (int z = bb.begin.z; z < bb.end.z; z++) {
        for (int y = bb.begin.y; y < bb.end.y; y++) {
            for (int x = bb.begin.x; x < bb.end.x; x++) {
                v.x = x;
                v.y = y;
                v.z = z;

                if (iftValidVoxel(annotation, v)) {
                    p = iftGetVoxelIndex(annotation, v);
                    if (annotation->val[p] > 0)
                        score++;
                }
            }
        }
    }

    return score/((bb.end.x-bb.begin.x)*(bb.end.y-bb.begin.y)*(bb.end.z-bb.begin.z));
}

void iftExtractParenchymaPatches(iftImage *img, iftImage *svoxels, iftImage *annotation, int sx, int sy, int sz, int patient_id, char *out_dir)
{
    char filename[300];
    iftImage *patch = NULL;
    iftLabeledSet *centroids = NULL;
    iftBoundingBox bb;
    iftVoxel u,v;
    int class;
    int p, q, i, label;

    centroids = iftGeometricCenters(svoxels);

    iftImage *aux = iftCreateImageFromImage(img);
    iftLabeledSetToImage(centroids,aux,0);

    i = 0;
    while (centroids != NULL){

        p = iftRemoveLabeledSet(&centroids,&label);
        if ((label == 3) || (label == 0))
            continue;
        u = iftGetVoxelCoord(svoxels,p);

        bb.begin.x = u.x - sx / 2;
        bb.begin.y = u.y - sy / 2;
        bb.begin.z = u.z - sz / 2;

        bb.end.x = u.x + sx / 2;
        bb.end.y = u.y + sy / 2;
        bb.end.z = u.z + sz / 2;

        patch = iftCreateImage(sx,sy,sz);
        int pos = 0;
        for (int x = bb.begin.x; x < bb.end.x; x++)
            for (int y = bb.begin.y; y < bb.end.y; y++)
                for (int z = bb.begin.z; z < bb.end.z; z++) {
                    v.x = x;
                    v.y = y;
                    v.z = z;

                    if (iftValidVoxel(img,v)) {
                        q = iftGetVoxelIndex(svoxels, v);
                        patch->val[pos] = img->val[q];
                    }
                    pos++;
                }

        class = 1;
        if (annotation != NULL) {
            float score = iftComputeOverlapWithAnnotation(annotation,bb);
            if (score >= 0.5)
                class = 2;
        }

        sprintf(filename, "%s/%06d_%08d_%06d_%d_%d_%d_%d_%d_%d.scn",out_dir,class,patient_id,i+1,bb.begin.x,bb.begin.y,bb.begin.z,bb.end.x,bb.end.y,bb.end.z);
        iftWriteImageByExt(patch,filename);
        iftDestroyImage(&patch);
        i++;
    }
}

iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **segm_path, char **img_out, int *sx, int *sy, int *sz, float *radius, int *n_samples);
void iftGetOptionalArgs(  iftDict *args, char **annotation_path);

int main(int argc, const char *argv[])
{

    iftDict *args = iftGetArgs(argc, argv);

    timer *t1,*t2;
    char *img_path         = NULL;
    char *segm_path        = NULL;
    char *annotation_path  = NULL;
    char *out_dir          = NULL;
    float radius;
    int sx, sy, sz, n_samples;

    iftGetRequiredArgs(args,&img_path,&segm_path,&out_dir,&sx,&sy,&sz,&radius,&n_samples);
    iftGetOptionalArgs(args,&annotation_path);

    iftImage *img        = iftReadImageByExt(img_path);
    iftImage *segm       = iftReadImageByExt(segm_path);
    iftImage *annotation = NULL;

    if (annotation_path != NULL){
        annotation = iftCreateImageFromImage(img);
        iftLabeledSet *ls = iftReadSeeds(img,annotation_path);
        while (ls != NULL){
            int label;
            int p = iftRemoveLabeledSet(&ls,&label);
            annotation->val[p] = 1;
        }
    }

    int patient_id = iftGetPatientID(img_path);

    t1 = iftTic();

    puts("- Generating Supervoxels");
    iftImage *svoxels = iftComputingSupervoxel(img,segm,radius,n_samples);
    iftDestroyImage(&segm);
    iftWriteImageByExt(svoxels,"svoxels.nii");

    puts("- Extracting patches");
    iftExtractParenchymaPatches(img, svoxels, annotation, sx, sy, sz, patient_id, out_dir);
    iftDestroyImage(&img);
    iftDestroyImage(&svoxels);

    t2 = iftToc();

    puts("\nDone...");
    fprintf(stdout,"Parenchyma Patch Extraction took ");
    puts(iftFormattedTime(iftCompTime(t1,t2)));

    return 0;
}

iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- Extract patches from the parenchyma based on the supervoxel.\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Image."},
            {.short_name = "-l", .long_name = "--segm-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Segmentation Image."},
            {.short_name = "-sx", .long_name = "--x-size", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=true, .help="Patch size along X."},
            {.short_name = "-sy", .long_name = "--y-size", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=true, .help="Patch size along Y."},
            {.short_name = "-sz", .long_name = "--z-size", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=true, .help="Patch size along Z."},
            {.short_name = "-r", .long_name = "--radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Segmentation Image."},
            {.short_name = "-n", .long_name = "--n-samples", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=true, .help="Segmentation Image."},
            {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output Directory."},
            {.short_name = "-a", .long_name = "--annotation", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Annotation file."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **segm_path, char **dir_out, int *sx, int *sy, int *sz, float *radius, int *n_samples)
{
    *img_path    = iftGetStrValFromDict("--input-img", args);
    *segm_path   = iftGetStrValFromDict("--segm-img", args);
    *dir_out     = iftGetStrValFromDict("--output-dir", args);
    *sx          = iftGetLongValFromDict("--x-size", args);
    *sy          = iftGetLongValFromDict("--y-size", args);
    *sz          = iftGetLongValFromDict("--z-size", args);
    *radius      = iftGetDblValFromDict("--radius", args);
    *n_samples   = iftGetLongValFromDict("--n-samples", args);

    if (!iftIsImageFile(*img_path)){
        iftError("Invalid Input Image: \"%s\"", "iftValidateRequiredArgs", img_path);
    }

    puts("");
    puts("--------------------------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Segmentation Image: \"%s\"\n", *segm_path);
    printf("- Patch size: (%d,%d,%d)\n", *sx,*sy,*sz);
    printf("- Radius: \"%.2f\"\n", *radius);
    printf("- Number of samples: \"%d\"\n", *n_samples);
    printf("- Output Directory: \"%s\"\n", *dir_out);
    puts("--------------------------------------");
    puts("");
}

void iftGetOptionalArgs(  iftDict *args, char **annotation_path)
{
    if (iftDictContainKey("--annotation", args, NULL)){
        *annotation_path = iftGetStrValFromDict("--annotation", args);
    }
}