//
// Created by peixinho on 6/29/15.
//

#include <ift.h>

int main(int argc, char** argv)
{
    if(argc<7)
    {
        iftError("Usage: <train path> <dataset path> <image size> <path size> <stride size> <outputpath>", "BagOfWords.c");
        return 1;
    }

    char output_path[100];
    int imgSize, pathSize, stride;

    sscanf(argv[3], "%d", &imgSize);
    sscanf(argv[4], "%d", &pathSize);
    sscanf(argv[5], "%d", &stride);

    strcpy(output_path, argv[6]);

    if(output_path[ strlen(output_path)-1 ]!='/'){
        strcat(output_path, "/");
    }

    iftBagOfFeatures * bow = iftCreateBow(TRUE, imgSize, imgSize, 1, pathSize, pathSize, 1, stride, stride, 1, iftBowPatchesRandomSampler,
                                iftBowPatchRawFeatsExtractor, iftBowKMeansKernels, iftBowSoftCoding, 100, 128);

    printf("Validate params ...");

    iftBowSetup(bow);

    printf(" Ok.\n");

    iftFileSet* trainfiles = iftLoadFileSetFromDirBySuffix(argv[1], "pgm");
    iftFileSet* files = iftLoadFileSetFromDirBySuffix(argv[2], "pgm");

    //int* labels = iftImageLabels(files);

    printf("Learning dictionary ...");
    fflush(stdout);

    iftBowLearn(bow, trainfiles);

    printf(" Ok.\n");

    printf("Extracting features\n\n");
    fflush(stdout);

    //serial number to count even with parallel
    int n = 0;

    #pragma omp parallel for
    for (int i = 0; i < files->n; ++i) {

        iftMImage* mimg;
        iftImage* img;
        char pathname[100];
        char* filename;

        img = iftReadImageP5(files->files[i]->path);
        mimg = iftBowTransform(bow, img);

        filename = basename(files->files[i]->path);
        strcpy(pathname, output_path);
        strcat(pathname, filename);

        n+=1;
        printf("%c[2K", 27);
        printf("\r%d/%d", n, files->n);
        fflush(stdout);

        //change extension: dumb way
        pathname[strlen(pathname)-4] = '\0';
        strcat(pathname, ".mig");

        iftWriteMImage(mimg, pathname);
//        iftWriteMImageBands(mimg, pathname);

        iftDestroyImage(&img);
        iftDestroyMImage(&mimg);
    }

    //free(labels);

    printf(" Ok.\n");

    iftDestroyBow(&bow);
    return 0;
}
