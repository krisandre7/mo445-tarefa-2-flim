#include "ift.h"

void MyStatisticsFromAllSeeds(iftFileSet *fs_seeds, char *inputdata_dir, float *mean, float *stdev, float stdev_factor) {
    int nseeds = 0, ninput_channels = 0;
    char *basename = NULL;
    char filename[200];
    iftMImage *input = NULL;

    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s.mimg", inputdata_dir, basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            nseeds += 1;
            for (int b = 0; b < ninput_channels; b++) {
                mean[b] += input->val[p][b];
            }
        }
        iftDestroyMImage(&input);
    }

    for (int b = 0; b < ninput_channels; b++) {
        mean[b] = mean[b] / nseeds;
    }

    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s.mimg", inputdata_dir, basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            for (int b = 0; b < ninput_channels; b++) {
                stdev[b] += (input->val[p][b] - mean[b]) * (input->val[p][b] - mean[b]);
            }
        }
        iftDestroyMImage(&input);
    }

    for (int b = 0; b < ninput_channels; b++) {
        stdev[b] = sqrtf(stdev[b] / nseeds) + stdev_factor;
    }

}

void MyWriteMeanStdev(char *basepath, float *mean, float *stdev, int ninput_channels){
    char filename[2][200];
    FILE *fp[2];

    sprintf(filename[0], "%s-mean.txt", basepath);
    sprintf(filename[1], "%s-stdev.txt", basepath);
    fp[0] = fopen(filename[0], "w");
    fp[1] = fopen(filename[1], "w");
    for (int b = 0; b < ninput_channels; b++) {
        fprintf(fp[0], "%f ", mean[b]);
        fprintf(fp[1], "%f ", stdev[b]);
    }
    fclose(fp[0]);
    fclose(fp[1]);
}

/* 

Create a patch dataset with true labels defined by seed labels in the
folder a model is going to be saved. Seed labels are incremented by
one when they start at value 0. Use iftConvertImagesToMImages.c (from
demo/Miscellaneous/ImageConverters) to convert the original images
into .mimg images, at the first convolutional layer.

*/


int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 6)
      iftError("Usage: iftPatchDataSetFromSeeds P1 P2 P3 P4 P5\n"
	       "P1: input folder with training images (.mimg)\n"
	       "P2: input folder with seed files (-seeds.txt)\n"
	       "P3: input network architecture (.json)\n"
	       "P4: input layer for patch definition (1, 2, 3, etc)\n"
	       "P5: output model's folder\n",
	       "main");
    
    tstart = iftTic();

    // Load input parameters
    
    iftFileSet  *fs_seeds = iftLoadFileSetFromDirBySuffix(argv[2], "-seeds.txt", 1);
    iftFLIMArch *arch     = iftReadFLIMArch(argv[3]);
    int          layer    = atoi(argv[4]);
    char *basename        = iftFilename(fs_seeds->files[0]->path, "-seeds.txt");    iftMakeDir(argv[5]);
 
    char filename[300];
    sprintf(filename, "%s/%s%s", argv[1], basename, ".mimg");
    iftMImage *input      = iftReadMImage(filename);
    iftAdjRel  *A         = iftFLIMAdjRelFromKernel(arch->layer[layer-1], iftIs3DMImage(input));
    int ninput_channels   = input->m;

    // Estimate and save mean and stdev
    
    float *mean  = iftAllocFloatArray(ninput_channels); 
    float *stdev = iftAllocFloatArray(ninput_channels); 
    MyStatisticsFromAllSeeds(fs_seeds, argv[1], mean, stdev, arch->stdev_factor);
    sprintf(filename,"%s/conv%d",argv[5],layer);
    MyWriteMeanStdev(filename,mean,stdev,ninput_channels);  

    // Get dataset from all seeds

    iftDataSet *Z   = iftDataSetFromAllSeeds(argv[2],argv[1],A);
    
    sprintf(filename,"%s/dataset%d.zip",argv[5],layer);
    iftWriteDataSet(Z,filename);
    
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFLIMArch(&arch);
    iftFree(basename);
    iftDestroyMImage(&input);
    iftDestroyAdjRel(&A);
    iftDestroyDataSet(&Z);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}


