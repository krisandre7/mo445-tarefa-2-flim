#include "ift.h"

iftMatrix *initialSolution(unsigned int numberRows, unsigned int numberColumns, double mean, double variance){
    iftMatrix *randomSolution = iftCreateMatrix(numberColumns,numberRows);
    unsigned int i,N;
    N = numberColumns*numberRows;
    for (i = 0; i < N; i++) {
        randomSolution->val[i] = ( (((double)i)/N) * variance) + mean;
    }
    return randomSolution;
}

iftMatrix* t_sne_p(iftMatrix *P, int no_dimens){
    int n = P->nrows;
    int N;
    double mometum = 0.5;
    double final_mometum = 0.8;
    int mom_switch_iter = 250;
    int stop_lying_iter = 100;
    int max_iter = 250;
    double epsilon = 500;
    double min_gain = 0.01;
    int iter,k,i,j,shift;
    double sum = 0;
    //double KL_constant;
    //double cost;

    iftMatrix *y_data = NULL;
    iftMatrix *y_data_transpose = NULL;
    iftMatrix *y_data_mul_transpose = NULL;
    iftMatrix *y_squared = NULL;
    iftMatrix *y_incs = NULL;
    iftMatrix * P_Log = NULL;
    iftMatrix *gains = NULL;
    iftMatrix *sum_ydata2 = NULL;

    iftMatrix *Q = iftCreateMatrix(n,n);
    iftMatrix *L = iftCreateMatrix(n,n);
    iftMatrix * diagonalMatrix = iftCreateMatrix(n,n);
    iftMatrix *y_grads = NULL;
    iftDblArray *Lsum = NULL;
    iftDblArray *Lmean = NULL;
    //iftMatrix * KL_matrix = NULL;
    iftMatrix * P_transpose = iftTransposeMatrix(P);


    iftSetDiagonalValue(P, 0);//make sure that all diagonal elements are zero
    iftMatricesAdditionPointWiseInPlace(P,P_transpose,&P);
    iftMultMatrixByScalar(P,0.5);
    sum = iftMatrixSum(P);
    iftComputeDivisionBetweenMatrixScalar(P,sum,'f');
    P_Log = iftCopyMatrix(P);
    iftLogarithmMatrix(P_Log);
    //KL_matrix = iftComputeOperationBetweenMatricesInPointWise(P,P_transpose,'*');
    //KL_constant = iftMatrixSum(KL_matrix);
    iftMultMatrixByScalar(P,4);

    y_data = initialSolution(n, no_dimens, 0, 1);
    y_squared = iftCreateMatrix(no_dimens,n);
    y_data_transpose = iftCreateMatrix(no_dimens,n);
    double *y_data_column_mean = (double *) iftAlloc(no_dimens, sizeof(double));
    double * sum_ydataSquared = (double *) iftAlloc(n, sizeof(double));
    y_incs = iftCreateSetMatrix(n, no_dimens, 0);
    gains = iftCreateSetMatrix(n, no_dimens, 1);
    N = n*no_dimens;

    for(iter=1; iter<=max_iter; iter++){

        /**Compute joint probability that point i and j are neighbours**/
        k = 0;
        y_data_transpose->nrows = y_data->nrows;
        y_data_transpose->ncols = y_data->ncols;
        for(i=0; i<y_data->nrows; i++){
            sum_ydataSquared[i] = 0;
            for(j=0; j<y_data->ncols; j++){
                y_data_transpose->val[k] = y_data->val[k];//copying data
                y_squared->val[k] = y_data->val[k]*y_data->val[k];//pointwise square
                sum_ydataSquared[i] += y_squared->val[k];
                k++;
            }
        }
        iftTransposeMatrixInPlace(y_data_transpose);
        iftMultMatricesInplace(y_data,y_data_transpose,&y_data_mul_transpose);
        k=0;
        sum = 0;
        for(i=0; i<y_data_mul_transpose->nrows; i++){
            for(j=0; j<y_data_mul_transpose->ncols; j++){
                y_data_mul_transpose->val[k] = (-2*y_data_mul_transpose->val[k]) + sum_ydataSquared[i] + sum_ydataSquared[j];//distance
                y_data_mul_transpose->val[k] = 1/(1+y_data_mul_transpose->val[k]);//student-t distribution
                k++;
            }
        }
        //y_data_transpose = num
        iftSetDiagonalValue(y_data_mul_transpose,0);
        sum = iftMatrixSum(y_data_mul_transpose);
        k=0;
        for(i=0; i<y_data_mul_transpose->nrows; i++) {
            for (j = 0; j < y_data_mul_transpose->ncols; j++) {
                Q->val[k] = (y_data_mul_transpose->val[k]/sum);
                k++;
            }
        }
        iftMatrixElementsSaturation(Q, 1E-307, 'l');
        /********************************************************/

        /*********Compute the gradients****************/
        k = 0;
        for(i=0; i<Q->nrows; i++) {
            for (j = 0; j < Q->ncols; j++) {
                L->val[k] = (P->val[k]-Q->val[k])*y_data_mul_transpose->val[k];
                k++;
            }
        }
        Lsum = iftComputeSumVector(L, 'c');
        for(i=0,shift=0; i<L->nrows; i++,shift+=L->ncols){
            diagonalMatrix->val[i+shift] = Lsum->val[i];
        }
        iftSubtractMatricesInPlace(diagonalMatrix,L,L);
        iftMultMatricesInplace(L,y_data,&y_grads);
        iftComputeMultiplicationBetweenMatrixScalar(y_grads,4,'f');
        /********************************************/

        /*************Update solution****************/
        for(k=0; k<N ;k++){
            if((y_grads->val[k] > 0 && y_incs->val[k] > 0) || (y_grads->val[k] < 0 && y_incs->val[k] < 0) ){
                gains->val[k] = 0.8 * gains->val[k];
            }else{
                gains->val[k] = 0.2 + gains->val[k];
            }
        }
        iftMatrixElementsSaturation(gains, min_gain,'l');

        k = 0;
        for(i=0; i< y_data->nrows; i++) {
            for (j = 0; j < y_data->ncols; j++) {
                y_incs->val[k] = (mometum * y_incs->val[k]) - (epsilon * gains->val[k] * y_grads->val[k]);
                y_data->val[k] += y_incs->val[k];
                k++;
            }
        }
        for (j = 0; j < y_data->ncols; j++) {
            y_data_column_mean[j] = 0;
            for (i = 0; i < y_data->nrows; i++) {
                y_data_column_mean[j] += y_data->val[iftGetMatrixIndex(y_data,j,i)];
            }
            y_data_column_mean[j] /= y_data->nrows;
            for (i = 0; i < y_data->nrows; i++) {
                y_data->val[iftGetMatrixIndex(y_data,j,i)] -= y_data_column_mean[j];
            }
        }
        /*****************************/

        if(iter == mom_switch_iter){
            mometum = final_mometum;

        }
        if (iter == stop_lying_iter){
            iftComputeOperationBetweenMatrixScalar(P,4,'f','/');
        }

       printf("iter %d\n",iter);
//        printf("ydata:\n");
//        iftPrintMatrix(y_data);
//        printf("\n");
    }

    iftDestroyMatrix(&diagonalMatrix);
    iftDestroyDblArray(&Lsum);
    iftDestroyDblArray(&Lmean);
    iftDestroyMatrix(&y_data_mul_transpose);
    iftDestroyMatrix(&y_squared);
    iftDestroyMatrix(&y_incs);
    iftDestroyMatrix(&P_Log);
    iftDestroyMatrix(&P_transpose);
    iftDestroyMatrix(&Q);
    iftDestroyMatrix(&L);
    iftDestroyMatrix(&gains);
    iftDestroyMatrix(&sum_ydata2);
    iftDestroyMatrix(&y_grads);
    iftDestroyMatrix(&y_data_transpose);
    free(sum_ydataSquared);
    free(y_data_column_mean);

    return y_data;
}

//P tem que estar alocado
void Hbeta(iftDblArray *D,float beta,iftDblArray *P, double *H){
    int i;
    double sum_P = 0;
    double sum_DP = 0;
    iftDblArray *DP = iftCreateDblArray(P->n);

    for (i=0; i<P->n; i++){
        P->val[i] = exp(-D->val[i]*beta);
        DP->val[i] = P->val[i] * D->val[i];
    }
    for(i=0; i<P->n; i++){
        sum_P += P->val[i];
        sum_DP += DP->val[i];
    }
    (*H) = log(sum_P) + (beta*(sum_DP/sum_P));
    for(i=0; i<P->n; i++){
        P->val[i] = P->val[i]/sum_P;

    }
}



void convertDistances2Probabilities(iftMatrix *D, double u,double tol,iftMatrix **Pp, iftDblArray** beta){
    int n = D->nrows;
    iftMatrix* P = iftCreateMatrix(n, n);
    iftDblArray *thisP = iftCreateDblArray(n);
    iftDblArray* distances =  iftCreateDblArray(n);
    (*beta) = iftCreateDblArray(n);
    double logU = log(u);
    double betaMin;
    double betaMax;
    double H,Hdiff;
    int i,j,tries;

    for (i=0; i<n; i++){
        (*beta)->val[i] = 1.0;
    }
    for (i=0; i<n; i++){

        for(j=0; j<n; j++){
            if(i == j){
                distances->val[j] = IFT_INFINITY_DBL;
            }else{
                distances->val[j] = D->val[iftGetMatrixIndex(D,j,i)];
            }
        }

        betaMin = IFT_INFINITY_DBL_NEG;
        betaMax = IFT_INFINITY_DBL;
        Hbeta(distances,(*beta)->val[i],thisP, &H);
        Hdiff = H - logU;
        tries = 0;
        while(fabs(Hdiff) > tol && tries < 50){
            if(Hdiff > 0){
                betaMin = (*beta)->val[i];
                if(betaMax == IFT_INFINITY_DBL){
                    (*beta)->val[i] = (*beta)->val[i]*2.0;
                }else{
                    (*beta)->val[i] = ((*beta)->val[i] + betaMax)/2.0;
                }
            }
            else{
                betaMax = (*beta)->val[i];
                if(betaMin == IFT_INFINITY_DBL_NEG){
                    (*beta)->val[i] = (*beta)->val[i]/2.0;
                }else{
                    (*beta)->val[i] = ((*beta)->val[i] + betaMin)/2.0;
                }
            }

            //recompute the values
            Hbeta(distances,(*beta)->val[i],thisP, &H);
            Hdiff = H - logU;
            tries++;
        }
        for(j=0; j<n; j++){
            P->val[iftGetMatrixIndex(D,j,i)] = thisP->val[j];
        }
    }

    *Pp = P;

    iftDestroyDblArray(&thisP);
    iftDestroyDblArray(&distances);
}

void dataNormalization(iftMatrix *data){
    double minimumValue = iftGetMinimumValueInMatrix(data);
    iftComputeSubtractionBetweenMatrixScalar(data,minimumValue,'f');
    double maximumValue = iftGetMaximumValueInMatrix(data);
    iftComputeDivisionBetweenMatrixScalar(data,maximumValue,'f');
    iftMatrix *meansColumn = iftMatrixMeanColumn(data);
    iftSubtractionMatrixVectorByRow(data,meansColumn);
    iftDestroyMatrix(&meansColumn);
}

void applyPCAInData(iftMatrix **data){
    unsigned int numberOfPoints = (*data)->nrows;
    unsigned int numberOfDimensions = (*data)->ncols;
    iftMatrix *C;
    iftMatrix *U;
    iftMatrix *S;
    iftMatrix *Vt;
    iftMatrix *meansColumn;
    iftMatrix * dataTransposed = iftTransposeMatrix((*data));
    iftMatrix * dataPCA;

    if(numberOfPoints > numberOfDimensions){
        C = iftMultMatrices(dataTransposed,(*data));
    }else{
        C = iftMultMatrices((*data),dataTransposed);
        iftComputeOperationBetweenMatrixScalar(C,1.0/(*data)->nrows,'f','*');
    }
    iftSingleValueDecomp(C, &U, &S, &Vt);

    if(numberOfDimensions >= numberOfPoints){
        //TODO
        printf("TODO:PCA\n");
    }
    meansColumn = iftMatrixMeanColumn((*data));
    iftSubtractionMatrixVectorByRow((*data),meansColumn);
    dataPCA = iftMultMatrices((*data), U);
    iftDestroyMatrix(data);
    (*data)  = iftCopyMatrix(dataPCA);
    iftDestroyMatrix(&C);
    iftDestroyMatrix(&U);
    iftDestroyMatrix(&S);
    iftDestroyMatrix(&Vt);
    iftDestroyMatrix(&meansColumn);
    iftDestroyMatrix(&dataTransposed);
    iftDestroyMatrix(&dataPCA);
}

iftMatrix* computePairWiseDistanceMatrix(iftMatrix *data){
    int i,j,k;
    k = 0;

    iftMatrix *distances;
    iftMatrix* dataSquared = iftCreateMatrix(data->ncols,data->nrows);
    iftMatrix * dataTransposed = iftCreateMatrix(data->ncols,data->nrows);
    double * sumDataSquared = (double *) iftAlloc(data->nrows, sizeof(double));;
    for(i=0; i<data->nrows; i++){
        sumDataSquared[i] = 0;
        for(j=0; j<data->ncols; j++){
            dataTransposed->val[k] = data->val[k];//copying data
            dataSquared->val[k] = data->val[k]*data->val[k];//pointwise square
            sumDataSquared[i] += dataSquared ->val[k];
            k++;
        }
    }



    k=0;
    iftTransposeMatrixInPlace(dataTransposed);

    //timer* t1 = iftTic();
    distances = iftMultMatricesGPU(data,dataTransposed);
    //timer* t2 = iftToc();
    //iftPrintFormattedTime(iftCompTime(t1,t2));
    for(i=0; i<distances->nrows; i++){
        for(j=0; j<distances->ncols; j++){
            distances->val[k] = (-2*distances->val[k]) + sumDataSquared[i] + sumDataSquared[j];//(X-Y)^2 = X^2 + Y^2 - 2XY
            k++;
        }
    }
    iftDestroyMatrix(&dataSquared);
    iftDestroyMatrix(&dataTransposed);
    free(sumDataSquared);
    return distances;
}


iftMatrix *tSNE(iftMatrix* inputData, unsigned int dimensionOut,double perplexity){

    iftMatrix *distances;
    iftMatrix * dataOut;
    iftDblArray* beta;
    iftMatrix *Probabilities;



    dataNormalization(inputData);
    applyPCAInData(&inputData);

    distances = computePairWiseDistanceMatrix(inputData);

    convertDistances2Probabilities(distances, perplexity,0.00001, &Probabilities,&beta);


    iftDestroyMatrix(&distances);

    timer* t1 = iftTic();
    dataOut = t_sne_p(Probabilities,dimensionOut);
    timer* t2 = iftToc();
    iftPrintFormattedTime(iftCompTime(t1,t2));



    return dataOut;
}


int main(int argc, char *argv[]) {

    //argv[1] = dataset
    //argv[2] = reduced dimension
    //argv[3] = perplexity 30

    iftDataSet *Zinput = iftReadOPFDataSet(argv[1]);
    int  reducedDimension;
    double perplexity;
    sscanf(argv[2],"%d",&reducedDimension);
    sscanf(argv[3],"%lf",&perplexity);
    iftMatrix *X       = iftDataSetToFeatureMatrix(Zinput);
    iftMatrix* dataOut = tSNE(X,reducedDimension,perplexity);
    iftDataSet *Zoutput = iftFeatureMatrixToDataSet(dataOut);

    int i;
    for(i=0; i < Zinput->nsamples; i++){
        Zoutput->sample[i].truelabel = Zinput->sample[i].truelabel;
        Zoutput->sample[i].label = Zinput->sample[i].label;
    }

    iftImage *outputImage = iftDraw2DFeatureSpace(Zoutput,CLASS,IFT_TEST);
    iftWriteImageP6(outputImage,"output.ppm");

    iftDestroyDataSet(&Zinput);
    iftDestroyDataSet(&Zoutput);
    iftDestroyMatrix(&X);
    iftDestroyMatrix(&dataOut);


    return 0;
}

