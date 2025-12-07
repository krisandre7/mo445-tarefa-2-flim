#include "ift.h"

#define KERNEL_SIZE 5

float *iftMagnitudeGaborFilter(iftAdjRel *A, float theta, float freq, int sigma_x, int sigma_y, double frequencyDamping)
{
  float *gabor = (float *)calloc(A->n, sizeof(float));
  float realGabor, imagGabor;
  for (int i = 0; i < A->n; i++)
  {
    int x = A->dx[i];
    int y = A->dy[i];

    float x0  = x * sin(theta) + y * cos(theta);
    float y0  = -x * cos(theta) + y * sin(theta);
    realGabor = frequencyDamping * (-0.5 * (((x0 * x0) / (sigma_x * sigma_y)) + ((y0 * y0) / (sigma_y * sigma_y)))) * cos(2 * IFT_PI * freq * x0);
    imagGabor = frequencyDamping * (-0.5 * (((x0 * x0) / (sigma_x * sigma_y)) + ((y0 * y0) / (sigma_y * sigma_y)))) * sin(2 * IFT_PI * freq * x0);
    gabor[i]  = sqrt(realGabor * realGabor + imagGabor * imagGabor);
  }

  return gabor;
}


iftMMKernel *iftMagnitudeGaborBank(int kernel_size, int n_orientations, int n_frequencies, double motherFrequency, int sigma_x, int sigma_y, double alpha)
{
  float *theta     = (float *)calloc(n_orientations, sizeof(float));
  float theta_step = IFT_PI / n_orientations;
  float *freqs     = (float *)calloc(n_frequencies, sizeof(float));

  float motherDistance = 1.0/motherFrequency;
  for (int i = 0; i < n_frequencies; i++) freqs[i]  = 1 / (motherDistance + 2 * i);
  for (int i = 0; i < n_orientations; i++) theta[i] = i * theta_step;


  int n_kernels  = n_orientations * n_frequencies;
  iftAdjRel   *A = iftRectangular(kernel_size, kernel_size);
  iftMMKernel *K = iftCreateMMKernel(A, 1, n_kernels);

  int k = 0;
  for (int i = 0; i < n_orientations; i++){
    for (int j = 0; j < n_frequencies; j++){
      double frequencyDamping = pow(alpha, -j);
      float *gabor            = iftMagnitudeGaborFilter(A, theta[i], freqs[j], sigma_x, sigma_y, frequencyDamping);

      for (int l = 0; l < A->n; l++) K->weight[k][0].val[l] = gabor[l];

      k++;
      free(gabor);
    }
  }

  free(theta);
  iftDestroyAdjRel(&A);

  return K;
}

iftMImage *extractGaborFeatures(iftMatrix *gaborBank, iftImage *img){
  iftAdjRel *A           = iftRectangularWithDilationForConv(KERNEL_SIZE, KERNEL_SIZE , 1, 1);
  iftMImage *mimg        = iftImageToMImage(img,GRAY_CSPACE);
  iftMatrix *XI          = iftMImageToFeatureMatrix(mimg,A,NULL);
  iftMatrix *XJ          = iftMultMatrices(XI, gaborBank);
  iftMImage *activ       = iftMatrixToMImage(XJ, mimg->xsize, mimg->ysize, mimg->zsize, gaborBank->ncols, 'c');


  iftDestroyAdjRel(&A);
  iftDestroyMImage(&mimg);
  iftDestroyMatrix(&XI);
  iftDestroyMatrix(&XJ);

  return activ;
}


iftMImage *computeGaborFeatureDifference(iftMImage *qFeatures, iftMImage *hFeatures){
  iftMImage *featureDifference = iftCreateMImage(hFeatures->xsize, hFeatures->ysize, 1, hFeatures->m);
  /* Compute mean and std for each band of reference features */
  double *mu  = (double *)calloc(hFeatures->m, sizeof(double));
  double *std = (double *)calloc(hFeatures->m, sizeof(double));

  for (int b = 0; b < hFeatures->m; b++) {
    double var = 0; 
    double mean = 0;

    // mean
    for (int p = 0; p < hFeatures->n; p++) mean += (double)hFeatures->val[p][b];
    mean /= (double)(hFeatures->n);

    // variance
    for (int p = 0; p < hFeatures->n; p++) var += pow((double)hFeatures->val[p][b] - mean, 2);
    
    var         /= (double)(hFeatures->n);
    double stdev = sqrt(var);

    mu[b] = mean;
    std[b] = stdev;
  }

  /* Compute feature difference for each band */

  int tau = 3; /* std tolerance */
  for (int b = 0; b < hFeatures->m; b++){
    for (int p = 0; p < hFeatures->n; p++){
      featureDifference->val[p][b] = fabs(qFeatures->val[p][b] - mu[b]) >= tau * std[b] ? qFeatures->val[p][b] : 0.0;
    }
  }

  free(mu);
  free(std);

  return featureDifference;
}

iftMImage *computeFeatureFusion(iftMImage *featureDifference, int n_orientations, int n_frequencies){
  iftMImage *Cq = iftCreateMImage(featureDifference->xsize, featureDifference->ysize, 1, n_orientations); /* Eq. 12.1*/
  iftMImage *H  = iftCreateMImage(featureDifference->xsize, featureDifference->ysize, 1, 1); /* Eq. 12.2*/
  
  for (int q = 0; q < n_orientations; q++){
    for (int p = 0; p < Cq->n; p++){
      for (int f = 0; f < n_frequencies; f++){
        int id         = q * n_frequencies + f;
        Cq->val[p][q] += featureDifference->val[p][id];
      }
    }
  }

  for (int p = 0; p < H->n; p++){
    for (int q = 0; q < n_orientations - 1; q++){
      H->val[p][0] += (1.0 / (n_orientations - 1)) * sqrt(((double)Cq->val[p][q] * (double)Cq->val[p][q + 1]));
    }
  }

  iftDestroyMImage(&Cq);

  return H;

}



int main(int argc, const char *argv[])
{

  if (argc != 4)
    iftError("iftDefectByGabor <query patch> <reference_patch> <output image>", "main");
  
  iftImage  *qPatch = iftReadImageByExt(argv[1]);
  iftImage  *hPatch = iftReadImageByExt(argv[2]);

  /* Defining Gabor family parameters */
  int L = 180; /* Number of angles considered */
  int S = 9; /* Number of frequencies */
  
  double fh          = 1.0/12.0;
  double fl          = 1.0/18.0;
  double mainFreq    = 1.0/12.0;
  double alpha       = pow((fh/fl), (1.0/(S-1))); /* Eq. 5.1 */
  double sigmaX      = sqrt(2 * log(2)) * (alpha + 1) / (2 * IFT_PI * fh * (alpha - 1)); /* Eq 5.2 */
  double tmp         = sqrt(2 * log(2) - pow((2 * log(2))/(2 * IFT_PI * sigmaX * fh), 2)); /* Eq. 5.3 (a)*/
  double sigmaY      = tmp * pow(2 * IFT_PI * tan(IFT_PI/(2*L)) * (fh - 2 * log(1/(4 * IFT_PI * IFT_PI * fh * sigmaX * sigmaX))), -1); /* Eq. 5.3 (b)*/
  
  /* Creating Gabor Filter Family */
  iftMMKernel *gaborBank = iftMagnitudeGaborBank(KERNEL_SIZE, L, S, mainFreq, sigmaX, sigmaY, alpha);

  /* Convoluting with patch to extract features */
  iftMatrix *M           = iftMMKernelToMatrix(gaborBank);
  iftMatrix *M_t         = iftTransposeMatrix(M);
  iftMImage *hFeatures   = extractGaborFeatures(M_t, hPatch);
  iftMImage *qFeatures   = extractGaborFeatures(M_t, qPatch);

  /* Compute feature difference and feature fusion */
  iftMImage *featureDiff     = computeGaborFeatureDifference(qFeatures, hFeatures);
  iftMImage *featureFusion   = computeFeatureFusion(featureDiff, L, S);

  printf("alpha: %lf, sigma x: %lf, sigma y: %lf\n", alpha, sigmaX, sigmaY);

  

  iftWriteMImage(featureFusion, "activ.mimg");
   /* Compute final defect detection */
  // int otsuThresh             = iftOtsu(featureFusion);
  // iftImage  *defectDetection = iftThreshold(featureFusion, 0, 255, otsuThresh);



  // iftWriteImageByExt(defectDetection, argv[3]);



  iftDestroyImage(&qPatch);
  iftDestroyImage(&hPatch);
  iftDestroyMMKernel(&gaborBank);
  iftDestroyMatrix(&M);
  iftDestroyMatrix(&M_t);
  iftDestroyMImage(&hFeatures);
  iftDestroyMImage(&qFeatures);
  iftDestroyMImage(&featureDiff);
  iftDestroyMImage(&featureFusion);
  // iftDestroyImage(&defectDetection);





  return 0;
}





