#include "ift.h"

/*
   Splits a fileset according to what is spected for Bag of Visual Words learning (demo/BagOfVisualWords/).
   It first divides the fileset into a number of separated subsets (according to the chosen sampling method)
   and then each subset is divided into learning, training and testing sets.

   Warning: The difference between this program and iftBoVWSplitFileSetMixSplits is that this program produces
   splits with no common samples, while the other produces splits that share the same samples

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
        iftError("\nUsage: iftBoVWSplitFileSetSepSplits <...>\n"
                    "[1] input_fileset: Input fileset or directory containing the files\n"
                    "[2] n_splits: Number of splits to be created\n"
                    "[3] sampling_method: Method to split the fileset\n"
                    "    0: Random\n"
                    "    1: Stratified\n"
                    "[4] learn_perc: Learning percentage for each split\n"
                    "[5] train_perc: Training percentage for each split\n"
                    "[6] test_perc: Testing percentage for each split\n"
                    "[7] output_basename: Basename for the output files\n",
                 "iftBoVWSplitFileSetSepSplits.c");
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
        iftError("learn_perc + train_perc + test_perc must be equal to 1", "iftBoVWSplitFileSetSepSplits.c");

    /* create the output dir */
    char outputDir[1024];
    sprintf(outputDir, "%s_splits%d_learn%d_train%d_test%d_%s-sep-splits", outputBasename,
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

    int nSamplesPerSplit = nSamples/nSplits;

    timer *tstart = iftTic();

    iftFileSet *remainFileset = iftCopyFileSet(fileset);
    char filename[2048];

    printf("\n--> Splitting filesets:\n");

    for(int i = 0; i < nSplits; i++) {
        printf("-split %d ...\n", i+1);
        
        iftFileSet *fs1 = NULL, *fs2 = NULL;
        if(i < nSplits-1) {
            /* split the dataset in two subsets (fs1 and fs2) */
            iftSplitFileSetInTwo(remainFileset, nSamplesPerSplit, sampMet, &fs1, &fs2);
        }
        else {
            /* copy the remaining samples (from the last split) to fs1 */
            fs1 = iftCopyFileSet(remainFileset);
        }

        /* split the fs1 subset in learn, train and test subsets */
        iftFileSet *learnFileset = NULL, *fs3 = NULL, *trainFileset = NULL, *testFileset = NULL;
        iftSplitFileSetInTwo(fs1, learnPerc*fs1->n, sampMet, &learnFileset, &fs3);
        /* since learn_perc + train_perc + test_perc equals 1, the training percentage must be computed with fs1 (instead of fs3) */
        iftSplitFileSetInTwo(fs3, trainPerc*fs1->n, sampMet, &trainFileset, &testFileset);

        printf(" learning set: %ld, training set: %ld, testing set: %ld\n", learnFileset->n, trainFileset->n, testFileset->n);

        /* save the filesets created */
        sprintf(filename,"%s/learn_%03d.csv", outputDir, i+1);
        iftWriteFileSetAsCSV(learnFileset, filename);
        sprintf(filename,"%s/train_%03d.csv", outputDir, i+1);
        iftWriteFileSetAsCSV(trainFileset, filename);
        sprintf(filename,"%s/test_%03d.csv", outputDir, i+1);
        iftWriteFileSetAsCSV(testFileset, filename);

        /* save the remaining samples for the next split */
        iftDestroyFileSet(&remainFileset);
        if(i < nSplits-1)
            remainFileset = iftCopyFileSet(fs2);

        iftDestroyFileSet(&fs1);
        if(i < nSplits-1) iftDestroyFileSet(&fs2);
        iftDestroyFileSet(&fs3);
        iftDestroyFileSet(&learnFileset);
        iftDestroyFileSet(&trainFileset);
        iftDestroyFileSet(&testFileset);
    }
    printf("\nProcessing time: %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    iftDestroyFileSet(&fileset);
    iftDestroyFileSet(&remainFileset);

    int MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    return(0);
}
