#include "ift.h"


iftMatrix *iftCSVToMatrix(  iftCSV *csv) {
    iftMatrix *M = iftCreateMatrix(csv->ncols, csv->nrows);
    
    for (int r = 0; r < csv->nrows; r++)
        for (int c = 0; c < csv->ncols; c++)
            iftMatrixElem(M, c, r) = atof(csv->data[r][c]);

    return M;
}


void iftParseCSV(  iftCSV *csv, iftMatrix **Xout, iftDblArray **yout) {
    iftMatrix *X = iftCreateMatrix(1, csv->nrows);
    iftDblArray *y = iftCreateDblArray(csv->nrows);

    for (int i = 0; i < csv->nrows; i++) {
        iftMatrixElem(X, 0, i) = atof(csv->data[i][0]);
        y->val[i] = atof(csv->data[i][1]);
    }
    
    *Xout = X;
    *yout = y;
}


iftDblArray *iftDblArrayStandarization(  iftDblArray *X) {
    iftDblArray *Xstand = iftCreateDblArray(X->n);

    float mean = 0.0f;
    float stdev = 0.0f;

    for (int i = 0; i < X->n; i++)
        mean += X->val[i];
    mean /= X->n;

    for (int i = 0; i < X->n; i++)
        stdev += iftPowerOfTwo(X->val[i] - mean);
    stdev = sqrtf(stdev / X->n);

    for (int i = 0; i < X->n; i++)
        Xstand->val[i] = (X->val[i] - mean) / stdev;

    return Xstand;
}


iftMatrix *iftMatrixStandarization(  iftMatrix *X) {
    iftMatrix *Xstand = iftCreateMatrix(X->ncols, X->nrows);

    #pragma omp parallel for
    for (int j = 0; j < X->ncols; j++) {
        float mean = 0.0f;
        float stdev = 0.0f;

        for (int i = 0; i < X->nrows; i++)
            mean += iftMatrixElem(X, j, i);
        mean /= X->nrows;

        for (int i = 0; i < X->nrows; i++)
            stdev += iftPowerOfTwo(iftMatrixElem(X, j, i) - mean);
        stdev = sqrtf(stdev / X->nrows);

        for (int i = 0; i < X->nrows; i++)
            iftMatrixElem(Xstand, j, i) = (iftMatrixElem(X, j, i) - mean) / stdev;
    }

    return Xstand;
}


#define func(x1, b0, b1)(b0 + (b1 * x1))

double funcN(  float *Xi,   iftDblArray *coefs, double bias) {
    double fx = 0.0;

    for (int j = 0; j < coefs->n; j++) {
        printf("fx += (%lf * %lf)\n", Xi[j], coefs->val[j]);
        fx += (Xi[j] * coefs->val[j]);
    }
    printf("fx += %lf\n", bias);
    fx += bias;
    printf("fx = %lf\n\n", fx);

    return fx; 
}


double iftMSE(  iftDblArray *y_pred,   iftDblArray *y) {
    double mse = 0.0;

    for (int i = 0; i < y->n; i++)
        mse += iftPowerOfTwo(y_pred->val[i] - y->val[i]);
    mse /= (2 * y->n);

    return mse;
}


void iftFitSimpleLinearRegression(  iftMatrix *X,   iftDblArray *y, double *b0_out, double *b1_out) {
    iftDblArray *y_pred = iftCreateDblArray(y->n);

    double b0 = 0.0;
    double b1 = 0.0;

    double alpha = 0.5; // learning rate

    double prev_mse = 0.0;
    double mse = -1.0;
    

    for (int it = 0; it < 1000 && !iftAlmostZero(prev_mse - mse); it++) {
        prev_mse = mse;

        mse = 0.0;
        double deriv_b0 = 0.0;
        double deriv_b1 = 0.0;

        for (int i = 0; i < y->n; i++) {
            y_pred->val[i] = func(iftMatrixElem(X, 0, i), b0, b1);

            mse += iftPowerOfTwo(y_pred->val[i] - y->val[i]);
            deriv_b0 += (y_pred->val[i] - y->val[i]);
            deriv_b1 += ((y_pred->val[i] - y->val[i]) * iftMatrixElem(X, 0, i));
        }
        mse /= (2 * y->n);
        deriv_b0 /= y->n;
        deriv_b1 /= y->n;

        printf("[%d] b0 = %.8lf\nb1 = %.8lf\nmse = %.8lf\n\n", it, b0, b1, mse);
        b0 = b0 - alpha * deriv_b0;
        b1 = b1 - alpha * deriv_b1;
    }
    puts("\ny_pred");
    iftPrintDoubleArray(y_pred->val, y_pred->n);
    iftDestroyDblArray(&y_pred);

    *b0_out = b0;
    *b1_out = b1;
}


iftDblArray *iftFitLinearRegresion(  iftMatrix *X,   iftDblArray *y, double *bias_out) {
    int n_samples = X->nrows;
    int n_feats = X->ncols;

    iftDblArray *y_pred = iftCreateDblArray(n_samples);

    // Zero is the initial coeficients and bias
    iftDblArray *coefs = iftCreateDblArray(n_feats);
    double bias = 0.0;


    double alpha = 0.5; // learning rate
    long double prev_mse = IFT_INFINITY_INT_NEG;
    long double mse = -1.0;

    long double *deriv_coefs = iftAlloc(n_feats, sizeof(long double));
    long double deriv_bias = 0.0;

    for (int it = 0; it < 100 && !iftAlmostZero(prev_mse - mse); it++) {
        printf("\nit [%d]\n", it);
        printf("coefs: ");
        for (int j = 0; j < n_feats; j++)
            printf("%f, ", coefs->val[j]);
        puts("");
        printf("bias = %lf\n\n", bias);
        
        prev_mse = mse;
        mse = 0.0;

        for (int i = 0; i < n_samples; i++) {
            y_pred->val[i] = funcN(&iftMatrixElem(X, 0, i), coefs, bias);

            double error = (y_pred->val[i] - y->val[i]);
            mse += (iftPowerOfTwo(error) / (2 * n_samples));

            for (int j = 0; j < n_feats; j++)
                deriv_coefs[j] += (error * iftMatrixElem(X, j, i) / n_samples);
            deriv_bias += (error / n_samples);
        }

        for (int j = 0; j < n_feats; j++) {
            coefs->val[j] = coefs->val[j] - alpha * deriv_coefs[j];
            deriv_coefs[j] = 0.0; // reinitialize derivative array
        }
        bias = bias - alpha * deriv_bias;
        deriv_bias = 0.0; // reinitialize derivative bias 

    }
    puts("\ny_pred");
    iftPrintDoubleArray(y_pred->val, y_pred->n);

    puts("[1]");
    iftDestroyDblArray(&y_pred);
    puts("[2]");
    iftFree(deriv_coefs);
    puts("[3]");

    *bias_out = bias;

    return coefs;
}


int main(int argc, const char *argv[]) {
    if (argc != 2)
        iftError("iftFitLinearRegression <data.csv>", "main");

    iftCSV *csv = iftReadCSV(argv[1], ',');

    iftMatrix *X;
    iftDblArray *y;
    iftParseCSV(csv, &X, &y);

    puts("X");
    iftMatrix *X_norm = iftMatrixStandarization(X);
    iftPrintMatrix(X_norm);
    
    puts("y");
    iftPrintDoubleArray(y->val, y->n);

    double bias;
    iftDblArray *coefs = iftFitLinearRegresion(X_norm, y, &bias);

    printf("bias: %lf\n", bias);
    puts("coefs");
    iftPrintDoubleArray(coefs->val, coefs->n);


    iftDestroyCSV(&csv);

    return 0;
}


