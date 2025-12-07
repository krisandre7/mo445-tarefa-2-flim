//
// Created by azaelmsousa on 18/04/23.
//

#include "ift.h"

#define stdev_factor 0.001

void SaveImage(char *dir, char *basename, iftMImage *mimg)
{
    char filename[200];

    sprintf(filename, "%s/%s.mimg", dir, basename);
    iftWriteMImage(mimg, filename);
}

iftMImage *ReadImageWithColorConv(char *filename, const char *ext, iftColorSpace cspace)
{
    iftMImage *input = NULL;
    if (strcmp(ext,".mimg") == 0){
        input = iftReadMImage(filename);
    }else{
        iftImage  *img = iftReadImageByExt(filename);
        input          = iftImageToMImage(img, cspace);
        iftDestroyImage(&img);
    }

    return(input);
}

void MyReadMeanStdev(char *basepath, iftMImage *mean, iftMImage *stdev, int ninput_channels) {
    char filename[2][200];
    FILE *fp[2];

    sprintf(filename[0], "%s-mean.txt", basepath);
    sprintf(filename[1], "%s-stdev.txt", basepath);
    fp[0] = fopen(filename[0], "r");
    fp[1] = fopen(filename[1], "r");
    float val;
    for (int b = 0; b < ninput_channels; b++) {
        //printf("%d\n",fscanf(fp[0], "%f", &val));
        if (fscanf(fp[0], "%f", &val) > 0) { mean->val[0][b] = val;}
        if (fscanf(fp[1], "%f", &val) > 0) { stdev->val[0][b] = val;}
    }
    fclose(fp[0]);
    fclose(fp[1]);
}

void NormalizeAndSavePerVoxel(iftFileSet *fs, const char *ext, iftMImage *mean, iftMImage *stdev, char *output_dir)
{
    for (int i=0; i < fs->n; i++) {
        iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);

#pragma omp parallel for
        for (int p=0; p < img->n; p++){
            for (int b=0; b < img->m; b++){
                img->val[p][b] = (img->val[p][b] - mean->val[p][b]) / stdev->val[p][b];
            }
        }
        char *basename  = iftFilename(fs->files[i]->path, ext);
        SaveImage(output_dir, basename, img);
        iftDestroyMImage(&img);
    }
}

void NormalizeAndSavePerBand(iftFileSet *fs, iftFileSet *fs_label, const char *ext, iftMImage *mean, iftMImage *stdev, char *output_dir)
{
    for (int i=0; i < fs->n; i++) {
        iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);
        iftImage *label = NULL;
        if (fs_label){
            label = iftReadImageByExt(fs_label->files[i]->path);
        }

#pragma omp parallel for
        for (int p=0; p < img->n; p++){
            for (int b=0; b < img->m; b++){
                if (label) {
                    if (label->val[p]) {
                        img->val[p][b] = (img->val[p][b] - mean->val[0][b]) / stdev->val[0][b];
                    } else {
                        img->val[p][b] = 0;
                    }
                } else {
                    img->val[p][b] = (img->val[p][b] - mean->val[0][b]) / stdev->val[0][b];
                }
            }
        }
        char *basename  = iftFilename(fs->files[i]->path, ext);
        SaveImage(output_dir, basename, img);
        iftDestroyMImage(&img);
        iftDestroyImage(&label);
    }
}

int main(int argc, char *argv[])
{
    timer *tstart = NULL;

    if ((argc < 5) || (argc > 6))
        iftError("Usage: iftBatchNormalization <P1> <P2> <P3> <P4 optional> <P5>\n"
                 "P1: Input folder/csv file with the images \n"
                 "    (.mimg, .nii.gz, etc)\n"
                 "P2: 1- per band in dataset, 2 - per voxel in dataset\n"
                 "P3: Mean and Standard deviation basename (without -mean.mimg and -stdev.mimg)\n"
                 "P4: Label folder/csv file with the label images (optional)\n"
                 "P5: Output folder with normalized images (.mimg). \n",
                 "main");

    tstart = iftTic();

    /* Get input parameters */

    iftFileSet *fs_input  = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);
    iftFileSet *fs_label  = NULL;
    int normalization     = atoi(argv[2]);
    char *basename        = argv[3];
    char *output_dir;
    if (argc == 6){
        fs_label = iftLoadFileSetFromDirOrCSV(argv[4], 1, 1);
        output_dir = argv[5];
    } else {
        output_dir = argv[4];
    }
    const char       *ext = iftFileExt(fs_input->files[0]->path);
    iftMakeDir(output_dir);

    /* Verify input parameters and perform initial procedures */
    iftMImage *input=NULL, *mean=NULL, *stdev=NULL;
    input = ReadImageWithColorConv(fs_input->files[0]->path, ext,
                                   LABNorm2_CSPACE);
    int nBands=input->m;
    iftDestroyMImage(&input);

    /* Normalization process */

    char mean_stdev_filename[200];
    switch (normalization) {
        case 1: /* per band in dataset */
            mean = iftCreateMImage(1,1,1,nBands);
            stdev = iftCreateMImage(1,1,1,nBands);
            MyReadMeanStdev(basename,mean,stdev,nBands);
            NormalizeAndSavePerBand(fs_input,fs_label,ext,mean,stdev,output_dir);
            break;
        case 2: /* per voxel in dataset */
            sprintf(mean_stdev_filename,"%s-mean.mimg",basename);
            mean  = iftReadMImage(mean_stdev_filename);
            sprintf(mean_stdev_filename,"%s-stdev.mimg",basename);
            stdev = iftReadMImage(mean_stdev_filename);
            NormalizeAndSavePerVoxel(fs_input,ext,mean,stdev,output_dir);
            break;
        default:
            iftError("Invalid normalization type. Please select either 1 or 2","main");
            break;
    }
    iftDestroyMImage(&mean);
    iftDestroyMImage(&stdev);

    /* Destroy memory */
    iftDestroyFileSet(&fs_input);
    iftDestroyFileSet(&fs_label);
    iftDestroyMImage(&mean);
    iftDestroyMImage(&stdev);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
