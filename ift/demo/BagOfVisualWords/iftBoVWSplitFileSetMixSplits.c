#include "ift.h"

/*
   Splits a fileset according to what is spected for Bag of Visual Words learning (demo/BagOfVisualWords/).
   It first samples the fileset into a number of splits (according to the chosen sampling method) and then each
   split is divided into learning, training and testing sets.

   Warning: The difference between this program and iftBoVWSplitFileSetSepSplits is that this program produces
   splits that share the same samples, while the other produces splits with no common samples

   author: Cesar Castelo
   date  : March, 5 2019.
*/

char *iftSamplingMethodName(int sampMet)
{
    char *metName = NULL;
    switch (sampMet) {
        case 0: // Random
            metName = iftCopyString("random");
            break;

        case 1: // Stratified
            metName = iftCopyString("stratified");
            break;

        default:
            iftError("Invalid sampling method chosen", "iftSamplingMethodName");
    }

    return metName;
}

int main(int argc, char *argv[])
{
    if(argc != 8) {
        iftError("\nUsage: iftBoVWSplitFileSetMixSplits <...>\n"
                    "[1] input_fileset: Input fileset or directory containing the files\n"
                    "[2] n_splits: Number of splits to be created\n"
                    "[3] sampling_method: Method to split the fileset\n"
                    "    0: Random\n"
                    "    1: Stratified\n"
                    "[4] learn_perc: Learning percentage for each split\n"
                    "[5] train_perc: Training percentage for each split\n"
                    "[6] test_perc: Testing percentage for each split\n"
                    "[7] output_basename: Basename for the output files\n",
                 "iftBoVWSplitFileSetMixSplits.c");
    }

    /* read parameters */
    iftFileSet *fileset = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);
    int nSplits = atoi(argv[2]);
    int sampMet = atoi(argv[3]);
    float learnPerc = atof(argv[4]);
    float trainPerc = atof(argv[5]);
    float testPerc = atof(argv[6]);
    char *outputBasename = iftCopyString(argv[7]);

    int MemDinInicial = iftMemoryUsed(1);

    iftRandomSeed(time(NULL));

    if(learnPerc + trainPerc + testPerc != 1)
        iftError("learn_perc + train_perc + test_perc must be equal to 1", "iftBoVWSplitFileSetMixSplits.c");

    /* create the output dir */
    char outputDir[1024];
    sprintf(outputDir, "%s_splits%d_learn%d_train%d_test%d_%s-mix-splits", outputBasename,
                                                                nSplits,
                                                                (int)(learnPerc*100.0+0.5),
                                                                (int)(trainPerc*100.0+0.5),
                                                                (int)(testPerc*100.0+0.5),
                                                                iftSamplingMethodName(sampMet));
    if(!iftDirExists(outputDir))
        iftMakeDir(outputDir);

    int nSamples = fileset->n;
    printf("- Number of files: %d\n", nSamples);
    printf("- Number of classes: %d\n", iftFileSetLabelsNumber(fileset));

    timer *tstart = iftTic();

    char filename[2048];

    printf("\n--> Splitting filesets:\n");

    /* split the fileset according to the learning percentage (it creates several splits) */
    iftSampler *sampler = NULL;
    int *labels = NULL;
    
    switch (sampMet) {
        case 0:
            sampler = iftRandomSubsampling(fileset->n, nSplits, learnPerc*fileset->n);
            break;

        case 1:
            labels = iftFileSetLabels(fileset);
            sampler = iftStratifiedRandomSubsampling(labels, fileset->n, nSplits, learnPerc*fileset->n);
            iftFree(labels);
            break;

        default:
            iftError("Invalid sampling method chosen", "iftBoVWSplitFileSetMixSplits.c");
    }

    /* for each split, we split the testing set in two */
    for(int i = 0; i < nSplits; i++) {
        printf("-split %d ...\n", i+1);
        
        iftSampleFileSet(fileset, sampler, i);

        iftFileSet *learnFileset = iftExtractFileSamples(fileset, IFT_TRAIN);
        iftFileSet *fs1 = iftExtractFileSamples(fileset, IFT_TEST);

        iftFileSet *trainFileset = NULL, *testFileset = NULL;
        iftSplitFileSetInTwo(fs1, trainPerc*fileset->n, sampMet, &trainFileset, &testFileset);

        printf(" learning set: %ld, training set: %ld, testing set: %ld\n", learnFileset->n, trainFileset->n, testFileset->n);

        /* save the filesets created */
        sprintf(filename,"%s/learn_%03d.csv", outputDir, i+1);
        iftWriteFileSetAsCSV(learnFileset, filename);
        sprintf(filename,"%s/train_%03d.csv", outputDir, i+1);
        iftWriteFileSetAsCSV(trainFileset, filename);
        sprintf(filename,"%s/test_%03d.csv", outputDir, i+1);
        iftWriteFileSetAsCSV(testFileset, filename);

        iftDestroyFileSet(&fs1);
        iftDestroyFileSet(&learnFileset);
        iftDestroyFileSet(&trainFileset);
        iftDestroyFileSet(&testFileset);
    }
    printf("\nProcessing time: %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    iftDestroyFileSet(&fileset);

    int MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    return(0);
}
