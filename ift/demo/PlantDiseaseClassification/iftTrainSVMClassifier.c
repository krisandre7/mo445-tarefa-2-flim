#include "ift.h"

enum { DEMO_LINEAR = 0, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */

iftDict *iftGetArguments(int argc, const char **argv);


int main(int argc, const char **argv)
{
    iftDataSet      *Z=NULL, *Z1[4], *Z2[4],*Ztrain=NULL,*Ztest=NULL;
    iftSVM          *svm=NULL;
    timer           *t1=NULL,*t2=NULL;
    float            stdev, mean, *acc;
    long              i, niters, num_of_comps = 0, kernel_type;
    double           C = 10.0, sigma = 0.1, perc_train;
    char            *input_dataset = NULL, *reduction = NULL, *OVO_or_OVA = NULL;
    iftDict         *args = NULL;

    /*--------------------------------------------------------*/

    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    args = iftGetArguments(argc, argv);

    input_dataset = iftGetStrValFromDict("--input-training-dataset", args);
    perc_train    = iftGetDblValFromDict("--training-percentage", args);
    niters        = iftGetLongValFromDict("--num-iterations", args);
    reduction     = iftGetStrValFromDict("--preprocess-type", args);

    if(iftDictContainKey("--number-of-components", args, NULL)) {
        num_of_comps = iftGetLongValFromDict("--number-of-components", args);
    }

    OVO_or_OVA    = iftGetStrValFromDict("--multiclass", args);
    if(iftDictContainKey("--SVM-C", args, NULL)) {
        C = iftGetDblValFromDict("--SVM-C", args);
    }
    iftRandomSeed(IFT_RANDOM_SEED);

    acc  = iftAllocFloatArray(niters);

    /* Initialization */

    Z  = iftReadOPFDataSet(input_dataset); // Read dataset Z
    printf("Total number of samples  %d\n",Z->nsamples);
    printf("Total number of features %d\n",Z->nfeats);
    printf("Total number of classes  %d\n",Z->nclasses);

    iftSetDistanceFunction(Z, 1);


    if ((num_of_comps <= 0) && !iftCompareStrings(reduction,"NoReduction"))
        iftError("Cannot reduce feature space to 0 or less components",
                 "iftDataClassifyByKNN");

    /* SVM creation */
    kernel_type = DEMO_LINEAR;

    if(iftCompareStrings(iftGetStrValFromDict("--kernel-type", args), "Linear")){
        svm = iftCreateLinearSVC(C);
    } else if(iftCompareStrings(iftGetStrValFromDict("--kernel-type", args), "RBF")) {
        kernel_type = DEMO_RBF;
        svm = iftCreateRBFSVC(C, sigma);
    } else if(iftCompareStrings(iftGetStrValFromDict("--kernel-type", args), "PreComputed")) {
        kernel_type = DEMO_PRECOMPUTED;
        svm = iftCreatePreCompSVC(C);
    } else {
        iftError("Invalid kernel type", "main");
    }

    t1     = iftTic();

    for (i=0; i < 4; i++){
        Z1[i]=Z2[i]=NULL;
    }

    for (i=0; i < niters; i++) {

        //    usleep(5);
        if (Z1[2] != NULL) iftDestroyDataSet(&Z1[2]);
        if (Z2[2] != NULL) iftDestroyDataSet(&Z2[2]);
        if (Z1[3] != NULL) iftDestroyDataSet(&Z1[3]);
        if (Z2[3] != NULL) iftDestroyDataSet(&Z2[3]);

        iftSelectSupTrainSamples(Z,perc_train); // Select training samples

        Z1[0] = iftExtractSamples(Z,IFT_TRAIN);
        Z2[0] = iftExtractSamples(Z,IFT_TEST);

        if (iftCompareStrings(reduction, "NoReduction")) {
            printf("- Normalizing DataSet\n");
            Z1[2] = iftNormalizeDataSetByZScore(Z1[0], NULL);
            iftDestroyDataSet(&Z1[0]);
            Z2[2] = iftNormalizeTestDataSet(Z1[2], Z2[0]);
            iftDestroyDataSet(&Z2[0]);
        } else if (iftCompareStrings(reduction,"PCA")) {
            Z1[1] = iftCentralizeDataSet(Z1[0]);
            iftDestroyDataSet(&Z1[0]);
            Z1[2] = iftTransFeatSpaceByPCA(Z1[1],num_of_comps);
            iftDestroyDataSet(&Z1[1]);
            Z2[1] = iftCentralizeTestDataSet(Z1[2],Z2[0]);
            iftDestroyDataSet(&Z2[0]);
            Z2[2] = iftTransformTestDataSetByPCA(Z1[2],Z2[1]);
            iftDestroyDataSet(&Z2[1]);
        } else if (iftCompareStrings(reduction,"SupPCA")) {
            Z1[1] = iftCentralizeDataSet(Z1[0]);
            iftDestroyDataSet(&Z1[0]);
            Z1[2] = iftTransFeatSpaceBySupPCA(Z1[1], num_of_comps);
            iftDestroyDataSet(&Z1[1]);
            Z2[1] = iftCentralizeTestDataSet(Z1[2], Z2[0]);
            iftDestroyDataSet(&Z2[0]);
            Z2[2] = iftTransformTestDataSetByPCA(Z1[2], Z2[1]);
            iftDestroyDataSet(&Z2[1]);
        }else {
            iftError("Invalid reduction option", "iftDataClassifyBySVM");
        }


        if (kernel_type == DEMO_PRECOMPUTED) {
            uchar traceNormalize = 0;
            float ktrace;

            printf("- Kernealizing DataSet\n");
            Z1[3] = iftKernelizeDataSet(Z1[2], Z1[2], LINEAR, traceNormalize, &ktrace);
            Ztrain = Z1[3];

            if (iftCompareStrings(OVO_or_OVA, "OVO")) { /* OVO */
                Z2[3] = iftKernelizeDataSet(Z1[2], Z2[2], LINEAR, 0, &ktrace);
                //	if (ktrace != 0.0) /* It will use the trace of Z1[3] */
                //	  iftMultDataSetByScalar(Z2[3], 1.0 / ktrace);
                iftDestroyDataSet(&Z1[2]);
                iftDestroyDataSet(&Z2[2]);

                Ztest  = Z2[3];
            } else {
                Ztest  = Z2[2];
            }
        } else {
            Ztrain = Z1[2];
            Ztest  = Z2[2];
        }


        if ((iftCompareStrings(OVO_or_OVA, "OVO"))) { /* OVO */
            iftSVMTrainOVO(svm, Ztrain, Z1[2]); // Training
            iftSVMClassifyOVO(svm, Ztest, IFT_TEST); // Classification
        } else {
            printf("- Training Linear SVM\n");
            iftSVMTrainOVA(svm, Ztrain, NULL);
            if ( (kernel_type == DEMO_PRECOMPUTED) || (kernel_type == DEMO_LINEAR) ) {
                printf("- Classifying Linear SVM\n");
                iftSVMLinearClassifyOVA(svm, Ztest, IFT_TEST, NULL); // Classification
            } else {
                iftSVMClassifyOVA(svm, Ztest, IFT_TEST); // Classification
            }
        }

        acc[i] = iftTruePositives(Ztest); // Compute accuracy on test set

        printf("acc[%ld] = %f\n",i,acc[i]);
    }

    /*
    if (atoi(argv[7])==0)
      iftSVMClassifyOVO(svm, Ztest, IFT_TEST); // just to complete the label map
    else{
      iftSVMClassifyOVA(svm, Ztest, IFT_TEST); // Classification
    }
    */

    iftDestroySVM(svm);
    iftDestroyDataSet(&Z);

    t2     = iftToc();

    fprintf(stdout,"classifications in %f ms\n",iftCompTime(t1,t2));

    mean = 0.0;
    for (i=0; i < niters; i++) {
        mean += acc[i];
    }
    mean /= niters;
    stdev = 0.0;

    for (i=0; i < niters; i++) {
        stdev += (acc[i]-mean)*(acc[i]-mean);
    }
    if (niters > 1)
        stdev = sqrtf(stdev/(niters - 1));

    free(acc);

    fprintf(stdout,"Accuracy of classification is mean=%f, stdev=%f\n",mean,stdev);

    if (Ztrain->nfeats == 2){
        iftImage *img = iftDraw2DFeatureSpace(Ztest,LABEL,0);
        iftWriteImageP6(img,"labels.ppm");
        iftDestroyImage(&img);
        img = iftDraw2DFeatureSpace(Ztest,CLASS,0);
        iftWriteImageP6(img,"classes.ppm");
        iftDestroyImage(&img);
        img = iftDraw2DFeatureSpace(Ztrain,STATUS,IFT_TRAIN);
        iftWriteImageP6(img,"train.ppm");
        iftDestroyImage(&img);
    }

    if (Z1[2] != NULL) iftDestroyDataSet(&Z1[2]);
    if (Z2[2] != NULL) iftDestroyDataSet(&Z2[2]);
    if (Z1[3] != NULL) iftDestroyDataSet(&Z1[3]);
    if (Z2[3] != NULL) iftDestroyDataSet(&Z2[3]);

    iftDestroyDict(&args);
    free(input_dataset);
    free(reduction);
    free(OVO_or_OVA);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return(0);
}







iftDict *iftGetArguments(int argc, const char **argv) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-training-dataset", .has_arg=true, .required=true, .help="Input dataset for training"},
            {.short_name = "-t", .long_name = "--training-percentage", .has_arg=true, .required=true, .help="Percentage of dataset that should be used for training"},
            {.short_name = "-n", .long_name = "--num-iterations", .has_arg=true, .required=true, .help="Number of iterations for training"},
            {.short_name = "-o", .long_name = "--ouptut-classifier", .has_arg=true, .required=true, .help="The output classifier filename"},
            {.short_name = "-p", .long_name = "--preprocess-type", .has_arg=true, .required=true, .help="Pre-processing type [NoReduction, PCA, SupPCA]"},
            {.short_name = "-c", .long_name = "--number-of-components", .has_arg=true, .required=false, .help="Number of components if data reduction is to be performed"},
            {.short_name = "-k", .long_name = "--kernel-type", .has_arg=true, .required=true, .help="Kernel type to be selected [Linear, RBF, PreComputed]"},
            {.short_name = "-m", .long_name = "--multiclass", .has_arg=true, .required=true, .help="Multi-class processing mode for SVM [OVO (one-versus-one), OVA (one-versus-all)]"},
            {.short_name = "",   .long_name = "--SVM-C", .has_arg=true, .required=false, .help="SVM parameter C"}
    };

    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[2048] = "This program takes as input a pre-computed dataset and trains an SVM classifier on it.";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}





