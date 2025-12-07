//
// Created by azaelmsousa on 18/07/20.
//

#include "ift.h"

iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **svoxels_path, char **out_path, char **out_dir_path);
void iftGetOptionalArgs(  iftDict *args, char **markers_path, int *class);

#define FEAT_SIZE 4095

iftImage *iftCreateImageFromBoundingBox(iftImage *img, iftBoundingBox bb)
{
    iftImage *dst = iftCreateImage(bb.end.x-bb.begin.x+1,bb.end.y-bb.begin.y+1,bb.end.z-bb.begin.z+1);
    iftCopyVoxelSize(img,dst);

    iftVoxel v;
    int q = 0;


//#pragma omp parallel for private (v,q)
    for (v.z = bb.begin.z; v.z <= bb.end.z; v.z++) {
        for (v.y = bb.begin.y; v.y <= bb.end.y; v.y++) {
            for (v.x = bb.begin.x; v.x <= bb.end.x; v.x++) {
                if (iftValidVoxel(img,v)) {
                    int p = iftGetVoxelIndex(img, v);
                    dst->val[q] = img->val[p];
                    q++;
                }
            }
        }
    }

    return dst;
}

void iftClassifySamplesFromMarkers(iftImage *svoxels, iftDataSet *Z, iftLabeledSet *S)
{
    bool supervised = false;
    iftLabeledSet *Saux = S;
    iftMatrix *M = iftCreateMatrix(3,Z->nsamples);

    while (Saux != NULL)
    {
        int p = Saux->elem;
        int label = Saux->label;
        M->val[iftGetMatrixIndex(M,label+1,svoxels->val[p]-1)]++;
        Saux = Saux->next;
    }

    Z->nclasses = 2;
    Z->ngroups = 1;

    for (int i = 0; i < M->nrows; i++){ //for each sample, skip sample 0 (background)
        int max = 0;
        int index = 0;
        for (int j = 1; j < M->ncols; j++)
            if (M->val[iftGetMatrixIndex(M,j,i)] > max) {
                max = M->val[iftGetMatrixIndex(M, j, i)];
                index = j;
            }
        if (index > 0) {
            supervised = true;
            Z->sample[i].truelabel = index;
            Z->sample[i].group = 1;
        }
    }

    if (supervised)
        iftAddStatus(Z,IFT_SUPERVISED);

    iftDestroyMatrix(&M);
}

iftHist **iftExtractFeatureVectorFromHistogram(iftImage *img, iftImage *svoxels)
{
    iftHist **H = NULL;
    int n_samples = iftMaximumValue(svoxels);
    int n_feats = FEAT_SIZE;
    int n_hist_out;

    printf("-- Computing histogram for each supervoxel... ");
    fflush(stdout);
    H = iftCalcGrayImageHistForLabels(img, svoxels, n_feats, iftMaximumValue(img), false, &n_hist_out);
    puts("Done");

    if ((n_hist_out-1) != n_samples) {
        iftError("Number of histograms differs: expected %d, computed %d", "iftExtractFeatureVector", n_samples,
                 (n_hist_out-1));
    }

    return H;
}

iftDataSet *iftConvertSupervoxelsToDataset(iftImage *img, iftImage *svoxels, iftHist **H, char *out_dir_path)
{
    iftDataSet *Z = NULL;
    int n_samples = iftMaximumValue(svoxels);
    int n_feats = FEAT_SIZE;
    char *filename;

    printf("-- Creating dataset... ");
    fflush(stdout);
    Z = iftCreateDataSet(n_samples,n_feats);
    puts("Done");

    printf("-- Storing features into dataset... ");
    fflush(stdout);
    for (int i = 0; i < n_samples; i++)
    {
        Z->sample[i].id = i;
        iftAddSampleStatus(&Z->sample[i],IFT_TRAIN);
        for (int j = 0; j < n_feats; j++){
            Z->sample[i].feat[j] = H[i]->val[j];
        }
    }
    puts("Done");

    printf("-- Creating reference data... ");
    fflush(stdout);
    iftFileSet *fs = iftCreateFileSet(n_samples);
    for (int i = 0; i < n_samples; i++){

        /* Extracting bounding box */
        iftVoxel gc_out;
        iftBoundingBox bb;
        iftImage *obj = iftExtractObject(svoxels,i+1);
        bb = iftMinBoundingBox(obj,&gc_out);
        iftDestroyImage(&obj);

        /* Creating new image from bounding box */
        filename = (char *)calloc(200,sizeof(char));
        sprintf(filename,"%s/%s/svoxel_%04d_%d_%d_%d_%d_%d_%d.nii.gz",getenv("PWD"),out_dir_path,i+1,bb.begin.x,bb.begin.y,bb.begin.z,bb.end.x,bb.end.y,bb.end.z);

        iftImage *bb_img = iftCreateImageFromBoundingBox(img, bb);
        iftWriteImageByExt(bb_img, filename);
        iftDestroyImage(&bb_img);

        /* Assigning reference data */
        fs->files[i] = iftCreateFile(filename);
        fs->files[i]->sample = i;
        free(filename);
    }
    puts("Done");

    Z->ref_data_type = IFT_REF_DATA_FILESET;
    Z->ref_data = fs;
    fs = NULL;

    return Z;
}

int main(int argc, const char *argv[])
{

    iftDict *args = iftGetArgs(argc, argv);

    timer *t1,*t2;
    char *img_path         = NULL;
    char *svoxels_path     = NULL;
    char *out_path         = NULL;
    char *out_dir_path     = NULL;
    char *markers_path     = NULL;
    int class              = 0;

    iftDataSet *Z = NULL;
    iftHist   **H = NULL;

    iftGetRequiredArgs(args,&img_path,&svoxels_path,&out_path,&out_dir_path);
    iftGetOptionalArgs(args,&markers_path,&class);

    iftImage *img = iftReadImageByExt(img_path);
    iftImage *svoxels = iftReadImageByExt(svoxels_path);

    t1 = iftTic();

    H = iftExtractFeatureVectorFromHistogram(img,svoxels);
    Z = iftConvertSupervoxelsToDataset(img,svoxels,H,out_dir_path);

    if (class > 0) {
        Z->nclasses = 1;
        iftAddStatus(Z,IFT_SUPERVISED);
        for (int s = 0; s < Z->nsamples; s++) {
            Z->sample[s].truelabel = class;
        }
    }

    int n_hist = iftMaximumValue(svoxels);
    for (int i = 0; i < n_hist; i++)
        iftDestroyHist(&H[i]);
    H = NULL;
    iftDestroyImage(&img);

    if (markers_path != NULL)
    {
        printf("-- Getting labels from markers... ");
        fflush(stdout);
        iftLabeledSet *S = iftReadSeeds(svoxels,markers_path);
        iftClassifySamplesFromMarkers(svoxels,Z,S);
        iftDestroyLabeledSet(&S);
        puts("Done");
    }
    iftDestroyImage(&svoxels);

    iftWriteDataSet(Z,out_path);
    iftDestroyDataSet(&Z);

    t2 = iftToc();

    puts("\nDone...");
    fprintf(stdout,"Dataset creation took ");
    puts(iftFormattedTime(iftCompTime(t1,t2)));

    return 0;
}

iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- iftExtractFeatureVectorFromSupervoxels. The main purpose of this program is to extract handcrafted" \
        "features from each supervoxel and save them on a iftDataset data structure.\n\n" \
        "**** Remember to normalize the images beforehand!****\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Image."},
            {.short_name = "-s", .long_name = "--svoxels-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Supervoxel label image."},
            {.short_name = "-d", .long_name = "--output-dir-reference-data", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output folder for reference data."},
            {.short_name = "-o", .long_name = "--output-dataset", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output Dataset."},
            {.short_name = "-m", .long_name = "--markers", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Annotation markers."},
            {.short_name = "-c", .long_name = "--manual-class", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=false, .help="Manually assign the class for every sample."},
            {.short_name = "-g", .long_name = "--group-class", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Manually assign the group for every sample."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **svoxels_path, char **out_path, char **out_dir_path)
{
    *img_path        = iftGetStrValFromDict("--input-img", args);
    *svoxels_path    = iftGetStrValFromDict("--svoxels-img", args);
    *out_path        = iftGetStrValFromDict("--output-dataset", args);
    *out_dir_path    = iftGetStrValFromDict("--output-dir-reference-data", args);

    if (!iftIsImageFile(*img_path)){
        iftError("Invalid Input Image: \"%s\"", "iftValidateRequiredArgs", img_path);
    }

    puts("");
    puts("--------------------------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Supervoxels Image: \"%s\"\n", *svoxels_path);
    printf("- Reference Data Path: \"%s\"\n", *out_dir_path);
    printf("- Output Dataset: \"%s\"\n", *out_path);
    puts("--------------------------------------");
}

void iftGetOptionalArgs(  iftDict *args, char **markers_path, int *class)
{
    if (iftDictContainKey("--markers", args, NULL)) {
        *markers_path = iftGetStrValFromDict("--markers", args);
        printf("- Markers: \"%s\"\n", *markers_path);
    }

    if (iftDictContainKey("--manual-class", args, NULL)) {
        *class = iftGetLongValFromDict("--manual-class", args);
        printf("- Class: \"%d\"\n", *class);
    }

    puts("--------------------------------------");

    puts("");
}