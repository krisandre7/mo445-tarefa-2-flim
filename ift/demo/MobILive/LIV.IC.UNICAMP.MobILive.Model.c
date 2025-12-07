#include "ift.h"
#include <ctype.h>
#include <malloc.h>

#include "LIV.IC.UNICAMP.MobILive.h"

enum { DEMO_LINEAR, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */


int main(int argc, char *argv[])
{
    iftDataSet      **Z_train = NULL, *Z1[3];
    iftSVM          *svm=NULL;
    timer           *t1=NULL,*t2=NULL;
    int              i, cl, num_of_comps, reduction;

    /*--------------------------------------------------------*/

#ifndef _SILENCE
    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;
#endif

    /*--------------------------------------------------------*/

    if (argc != 6)
        iftError("Usage: LIV.IC.UNICAMP.MobILive.Model <train_dataset#.dat>" \
             " <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>" \
             " <num_of_comps>" \
             " <kernel_type [0=linear,1=RBF,2=precomputed]" \
             " <multiclass [0=OVO,1=OVA]>",
                 "LIV.IC.UNICAMP.MobILive");

//  iftRandomSeed();

    /* Initialization */
    if (strrchr(argv[1],'#') == NULL)
        iftError("wildcard # is missing in train_dataset#.data","LIV.IC.UNICAMP.MobILive.Model");

    Z_train = (iftDataSet**)malloc(sizeof(iftDataSet*)*nCNNs);
    if (Z_train == NULL)
        iftError("allocating Ztrain vector","LIV.IC.UNICAMP.MobILive.Model");

    char *file,scl[10];

    char fileConvNet[200];
    for(cl=0;cl<nCNNs;cl++) {
        sprintf(scl,"%d",cl);
        strcpy(fileConvNet,argv[1]);
        file=replace_str(fileConvNet,"#",scl);
#ifndef _SILENCE
        fprintf(stdout,"loading %s dataset\n",file);
#endif
        Z_train[cl] = iftReadOPFDataSet(file); // Read dataset Z
        free(file);
    }

    for(cl=0;cl<nCNNs;cl++) {
#ifndef _SILENCE
        printf("***** TRAINING DATASET *****\n");
        printf("Total number of samples  %d\n"  ,Z_train[cl]->nsamples);
        printf("Total number of features %d\n"  ,Z_train[cl]->nfeats);
        printf("Total number of classes  %d\n\n",Z_train[cl]->nclasses);
#endif


        iftSetDistanceFunction(Z_train[cl], 1);
        iftSetStatus(Z_train[cl], IFT_TRAIN); // set all elements as IFT_TRAIN
    }

    reduction    = atoi(argv[2]);
    num_of_comps = atoi(argv[3]);

    if ((num_of_comps <= 0) && (reduction > 0))
        iftError("Cannot reduce feature space to 0 or less components",
                 "LIV.IC.UNICAMP.MobILiveModel");

    t1     = iftTic();

    for (i=0; i < 3; i++){
        Z1[i]=NULL;
    }

    // SVM
    int kernel_type = atoi(argv[4]);
    float C = 1e10;
    float sigma = 0.1;
    for(cl=0;cl<nCNNs;cl++) {
        // SVM
        switch(kernel_type){
            case DEMO_LINEAR:
                svm = iftCreateLinearSVC(C);
                break;
            case DEMO_RBF:
                svm = iftCreateRBFSVC(C, sigma);
                break;
            case DEMO_PRECOMPUTED:
                svm = iftCreatePreCompSVC(C);
                break;
            default:
                iftError("Invalid kernel type","LIV.IC.UNICAMP.MobILiveModel");
        }

        if (Z1[2] != NULL) iftDestroyDataSet(&Z1[2]);

        Z1[0] = iftCopyDataSet(Z_train[cl]);

        switch(reduction) {
            case 0:
                Z1[2] = iftNormalizeDataSet(Z1[0]);
                iftDestroyDataSet(&Z1[0]);

                // saving feature normalization vector
                char fileNorm[200];
                sprintf(scl,"%d",cl);
                file=replace_str(NORM_NAME,"#",scl);
                sprintf(fileNorm,"%s.%d.data",file,Z1[2]->nfeats);
                free(file);

                FILE* pFn = fopen(fileNorm,"wb+");
                if ( (pFn == NULL) )
                    iftError("Can't create mean feature file","LIV.IC.UNICAMP.MobILive.Model");
#ifndef _SILENCE
                fprintf(stdout,"creating %s feature normalization data\n",fileNorm);
#endif
                fwrite(Z1[2]->fsp.mean ,sizeof(float),Z1[2]->nfeats,pFn);
                fwrite(Z1[2]->fsp.stdev,sizeof(float),Z1[2]->nfeats,pFn);
                fclose(pFn);
                break;
            case 1:
                Z1[1] = iftNormalizeDataSet(Z1[0]);
                iftDestroyDataSet(&Z1[0]);
                Z1[2] = iftTransFeatSpaceByPCA(Z1[1],num_of_comps);
                iftDestroyDataSet(&Z1[1]);
                // ToDo's
                break;
            case 2:
                Z1[1] = iftNormalizeDataSet(Z1[0]);
                iftDestroyDataSet(&Z1[0]);
                Z1[2] = iftTransFeatSpaceBySupPCA(Z1[1],num_of_comps);
                iftDestroyDataSet(&Z1[1]);
                // ToDo's
                break;
            default:
                iftError("Invalid reduction option","LIV.IC.UNICAMP.MobILive.Model");
        }


        if (kernel_type == DEMO_PRECOMPUTED) {
            iftDataSet *Zaux, *Z1k, *Z2k;
            uchar traceNormalize = 1;
            float ktrace;

            Z1k = iftKernelizeDataSet2(Z1[2], Z1[2], LINEAR, traceNormalize, &ktrace);

            if (ktrace != 0.0)
                iftMultDataSetByScalar(Z2k, 1.0 / ktrace);

            Zaux = Z1[2];
            Z1[2] = Z1k;
            iftDestroyDataSet(&Zaux);

            // ToDo's
        }

        if (atoi(argv[5])==0){
            iftSVMTrainOVO(svm, Z1[2]); // Training
            // ToDo's
        }else{
            iftSVMTrainOVA(svm, Z1[2]);
            if (Z1[2]->nclasses == 2) {
                float rho;
                iftSample w = iftSVMGetNormalHyperplane(svm,0,Z1[2],&rho);
                char fileModel[200];
                sprintf(scl,"%d",cl);
                file=replace_str(MODEL_NAME,"#",scl);
                sprintf(fileModel,"%s.%d.data",file,Z1[2]->nfeats);
                free(file);
#ifndef _SILENCE
                fprintf(stdout,"creating %s model data\n",fileModel);
#endif
                FILE *pFM = fopen(fileModel,"wb+");
                if ( (pFM == NULL) )
                    iftError("Can't save model classification file","LIV.IC.UNICAMP.MobILive.Model");
                fwrite((w.feat),sizeof(float),Z1[2]->nfeats,pFM);
                fprintf(stdout,"rho: %f, class: %lf %lf\n",rho,svm->class[0],svm->class[1]);
                fwrite(&rho,sizeof(float),1,pFM);
                fwrite(&(svm->class[0]),sizeof(double),1,pFM);
                fwrite(&(svm->class[1]),sizeof(double),1,pFM);
                fclose(pFM);

                free(w.feat);
            }


        }

        iftDestroyDataSet(&Z1[2]);
        iftDestroySVM(svm);
    }



    for(cl=0;cl<nCNNs;cl++)
        iftDestroyDataSet(&(Z_train[cl]));
    free(Z_train);

    t2     = iftToc();

#ifndef _SILENCE
    fprintf(stdout,"analysis  in %f ms\n",iftCompTime(t1,t2));
#endif


    /* ---------------------------------------------------------- */

#ifndef _SILENCE
    info = mallinfo();
    MemDinFinal = info.uordblks;
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);
#endif

    return(0);
}





