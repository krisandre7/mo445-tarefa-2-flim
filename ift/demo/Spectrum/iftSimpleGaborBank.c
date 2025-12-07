#include "ift.h"

#define SIGMA_X 4
#define SIGMA_Y 4
#define MAX_FREQ 13 // 0.08 Hz
#define MIN_FREQ 9 // 0.20 Hz
#define GABOR_KERNEL_SIZE 27
#define COHERENCE_WINDOW_SIZE 9
#define COHERENCE_THRESHOLD 0.95
#define OVEC_RADIUS 8.0
#define ADJ_RADIUS 11
#define VARIANCE_BLK_SIZE 16
#define VARIANCE_THRESHOLD 0.1

// #define DEBUG 1 // uncomment this line if wish to see filters and activations


void printKernelBank(iftMMKernel *K_bank, int xsize, int ysize){
  /*
  Prints a Gabor iftMMKernel.
  Each kernel has 2 bands representing the real and the imaginary parts 
  */
  char filename[200];
  iftVoxel u;
  int j, p, dmin[3];
  iftImage *img;

  for (int i = 0; i < K_bank->nkernels; i++){
    img = iftCreateImage(xsize, ysize, 1);
    dmin[0]=IFT_INFINITY_INT;
    dmin[1]=IFT_INFINITY_INT;
    dmin[2]=IFT_INFINITY_INT;
    float min_weight = IFT_INFINITY_FLT;
    float max_weight = IFT_INFINITY_FLT_NEG;

    for (j=0; j < K_bank->A->n; j++) {
      if (K_bank->A->dx[j]<dmin[0])
        dmin[0] = K_bank->A->dx[j];
      if (K_bank->A->dy[j]<dmin[1])
        dmin[1] = K_bank->A->dy[j];
      if (K_bank->A->dz[j]<dmin[2])
        dmin[2] = K_bank->A->dz[j];
    }

    for (j=0; j < K_bank->A->n; j++){
      if(K_bank->weight[i][0].val[j] > max_weight)
        max_weight = K_bank->weight[i][0].val[j];
      
      if (K_bank->weight[i][0].val[j] < min_weight)
        min_weight = K_bank->weight[i][0].val[j];
    }

    for (j=0; j < K_bank->A->n; j++) {
      K_bank->weight[i][0].val[j] = 255.0*(K_bank->weight[i][0].val[j] - min_weight)/(max_weight - min_weight);
    }

    for (j=0; j < K_bank->A->n; j++) {
      u.x = K_bank->A->dx[j]-dmin[0];
      u.y = K_bank->A->dy[j]-dmin[1];
      u.z = K_bank->A->dz[j]-dmin[2];
      p   = iftGetVoxelIndex(img,u);
      img->val[p] = iftRound(K_bank->weight[i][0].val[j]);
    }
    sprintf(filename,"kernel-%d.png", i);
    iftWriteImageByExt(img, filename);
    iftDestroyImage(&img);
  }
  
}

iftImage *GetMask(iftImage *bin, float radius)
{
  iftImage *mask     = iftAsfCOBin(bin,radius);
  iftImage *cbasins  = iftCloseBasins(mask,NULL,NULL);
  iftDestroyImage(&mask);
  
  return(cbasins);
}

float *getFrequencyArray(int n_frequencies){
  float *freq      = (float *)calloc(n_frequencies, sizeof(float));
  float freq_step  = (float)(MAX_FREQ - MIN_FREQ) / n_frequencies;

  /*Getting array of frequencies*/
  for (int i = 0; i < n_frequencies; i++){
      freq[i] = 1/(MIN_FREQ + i * freq_step);
  }
  return freq;
}

float *getOrientationArray(int n_orientations){
  float *theta = (float *)calloc(n_orientations, sizeof(float));
  float theta_step = IFT_PI / n_orientations;
  /*Getting array of orientations*/
  for (int i = 0; i < n_orientations; i++){
      theta[i] = i * theta_step;
  }

  return theta;
}

int getFreqIndex(int b, int n_frequencies) {
  return (b % n_frequencies);
}


iftImage *getMaxGaborActivByPixelInWindow(iftMImage *mimg, int *bmax, int n_frequencies, iftImage *mask){
  iftImage *img   = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
  iftFImage *fimg = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  iftAdjRel *A    = iftRectangular(ADJ_RADIUS,ADJ_RADIUS);//iftCircular(ADJ_RADIUS);
  
  for (int p = 0; p < mimg->n; p++){
    if (mask->val[p] != 0){
      int max  = IFT_INFINITY_INT_NEG;
      bmax[p]  = 0; 
      /*Searching for max band in neighborhood*/    
      iftVoxel u = iftGetVoxelCoord(img, p);
      for (int i = 0; i < A->n; i++){
	iftVoxel v = iftGetAdjacentVoxel(A, u, i);
	if (iftValidVoxel(img, v)){
	  int q = iftGetVoxelIndex(img, v);
	  for (int b = 0; b < mimg->m; b++){
	    if (mimg->val[q][b] > max){
	      max = mimg->val[q][b];
	      bmax[p] = b;
	    }
	  }
	}
      }
      fimg->val[p] = mimg->val[p][bmax[p]];
      if (fimg->val[p] < 0) /* ReLU */
      	fimg->val[p]=0;
    }
  }

  float image_max = IFT_INFINITY_FLT_NEG;
  float image_min = IFT_INFINITY_FLT;

  for (int i = 0; i < fimg->n; i++){
    if (image_max < fimg->val[i]) image_max = fimg->val[i];
    if (image_min > fimg->val[i]) image_min = fimg->val[i];
  }

  for (int i = 0; i < img->n; i++){
    img->val[i] = (int)(255.0 * (fimg->val[i] - image_min)/(image_max - image_min));
  }

  iftDestroyAdjRel(&A);
  return img;
}

iftImage *getSumGaborActivByPixelInWindow(iftMImage *mimg, int *bmax, int n_frequencies, iftImage *mask){
  iftImage *img   = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
  iftFImage *fimg = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  iftAdjRel *A    = iftRectangular(ADJ_RADIUS,ADJ_RADIUS);//iftCircular(ADJ_RADIUS);


  for (int p = 0; p < mimg->n; p++){
    float max_sum = IFT_INFINITY_FLT_NEG;
    int   max_id  = 0;
    float *band_sum = (float *)calloc(mimg->m, sizeof(float));

    /* Getting the sum in the neighbourhood for each band */
    iftVoxel u = iftGetVoxelCoord(img, p);
    for (int b = 0; b < mimg->m; b++){
      band_sum[b] = 0.0;
      for (int i = 0; i < A->n; i++){
        iftVoxel v = iftGetAdjacentVoxel(A, u, i);
        if (iftValidVoxel(img, v)){
          int q = iftGetVoxelIndex(img, v);
          band_sum[b] += mimg->val[q][b];
        }
      }
    }
    /* Finding the band with the highest sum */

    for (int b = 0; b < mimg->m; b++){
      if (band_sum[b] > max_sum){
        max_sum = band_sum[b];
        max_id  = b;
      }
    }
    if (mask->val[p])
      fimg->val[p] = mimg->val[p][max_id];
    
    free(band_sum);
  }


  float image_max = IFT_INFINITY_FLT_NEG;
  float image_min = IFT_INFINITY_FLT;

  for (int i = 0; i < fimg->n; i++){
    if (image_max < fimg->val[i]) image_max = fimg->val[i];
    if (image_min > fimg->val[i]) image_min = fimg->val[i];
  }

  for (int i = 0; i < img->n; i++){
    img->val[i] = (int)(255.0 * (fimg->val[i] - image_min)/(image_max - image_min));
  }


  iftDestroyAdjRel(&A);
  return img;
}

iftImage *getFrequestGaborActivByPixelInWindow(iftMImage *mimg, int *bmax, int n_frequencies, iftImage *mask){
  iftImage *img   = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
  iftFImage *fimg = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  iftAdjRel *A    = iftRectangular(ADJ_RADIUS,ADJ_RADIUS);//iftCircular(ADJ_RADIUS);


  for (int p = 0; p < mimg->n; p++){
    float max_freq   = IFT_INFINITY_FLT_NEG;
    int   max_id     = 0;
    float *band_freq = (float *)calloc(mimg->m, sizeof(float));

    /* Getting the freq in the neighbourhood for each band */
    iftVoxel u = iftGetVoxelCoord(img, p);
    for (int i = 0; i < A->n; i++){
      float band_max = IFT_INFINITY_FLT_NEG;
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(img, v)){
        int q = iftGetVoxelIndex(img, v);
        for (int b = 0; b < mimg->m; b++){
          if (band_max < mimg->val[q][b]){
            band_max = mimg->val[q][b];
            max_id = b;
          }
        }
      }
      if (band_max != IFT_INFINITY_FLT_NEG)
        band_freq[max_id] += 1;
    }
    max_id = 0;

    for (int b = 0; b < mimg->m; b++){
      if (max_freq < band_freq[b]){
        max_freq = band_freq[b];
        max_id = b;
      }
    }

    if (mask->val[p])
          fimg->val[p] = mimg->val[p][max_id];

    free(band_freq);
  }


  float image_max = IFT_INFINITY_FLT_NEG;
  float image_min = IFT_INFINITY_FLT;

  for (int i = 0; i < fimg->n; i++){
    if (image_max < fimg->val[i]) image_max = fimg->val[i];
    if (image_min > fimg->val[i]) image_min = fimg->val[i];
  }

  for (int i = 0; i < img->n; i++){
    img->val[i] = (int)(255.0 * (fimg->val[i] - image_min)/(image_max - image_min));
  }


  iftDestroyAdjRel(&A);
  return img;
}

char *Basename(char *path)
{
  char *basename     = iftBasename(path);
  iftSList *slist    = iftSplitString(basename,"/");
  strcpy(basename,slist->tail->elem);
  iftDestroySList(&slist);
  return(basename);
}

float *gaborFilter(iftAdjRel *A, float theta, float freq){
    float *gabor = (float *)calloc(A->n, sizeof(float));

    for (int i = 0; i < A->n; i++){
        int x = A->dx[i];
        int y = A->dy[i];

        float x0 =  x * sin(theta) + y * cos(theta);
        float y0 = -x * cos(theta) + y * sin(theta); 

        gabor[i] = exp(-0.5 * (((x0 * x0)/(SIGMA_X * SIGMA_X)) + ((y0 * y0)/(SIGMA_Y * SIGMA_Y)))) * cos(2 * IFT_PI * freq * x0);
    }

    return gabor;
}


int getThetaIndex(int b, int n_frequencies){
  return (b / n_frequencies);
}


float maxArray(int n, float *arr){
  float max = IFT_INFINITY_FLT_NEG;
  for (int i = 0; i < n; i++){
    if (arr[i] > max) max = arr[i];
  }

  return max;
}

float minArray(int n, float *arr){
  float min = IFT_INFINITY_FLT;
  for (int i = 0; i < n; i++){
    if (arr[i] < min) min = arr[i];
  }

  return min;
}

float meanArray(int n, float *arr){
  float mean = 0.0;
  for (int i = 0; i < n; i++){
    mean += arr[i];
  }
  mean /= n;
  return mean;
}

iftMMKernel *iftGaborKernelBank(int kernel_size, int n_orientations, int n_frequencies){
    float *theta = getOrientationArray(n_orientations);
    float *freq  = getFrequencyArray(n_frequencies);

    int n_kernels = n_orientations * n_frequencies;
    //    iftAdjRel   *A = iftRectangularWithDilationForConv(GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE, 1, 1);
    iftAdjRel   *A = iftRectangular(GABOR_KERNEL_SIZE,GABOR_KERNEL_SIZE);

    iftMMKernel *K = iftCreateMMKernel(A, 1, n_kernels);

    int k = 0;
    for (int i = 0; i < n_orientations; i++){
        for (int j = 0; j < n_frequencies; j++){
            float *gabor = gaborFilter(A, theta[i], freq[j]);


	    float mag = 0.0;
            for (int l = 0; l < A->n; l++){
              K->weight[k][0].val[l] = gabor[l];
	      mag += gabor[l]*gabor[l];
            }
	    mag = sqrt(mag);
            /* for (int l = 0; l < A->n; l++){ */
	    /*   K->weight[k][0].val[l] /= mag; */
            /* } */
	    
            k++;
            free(gabor);
        }
    }

    free(theta);
    free(freq);
    iftDestroyAdjRel(&A);

    return K;
}


float imageMean(iftImage *img){
    float mean = 0.0;
    for (int i = 0; i < img->n; i++){
        mean += (float) img->val[i];
    }
    mean = mean / img->n;
    return mean;
}

float imageVar(iftImage *img, float mean){
    float var = 0.0;
    for (int i = 0; i < img->n; i++){
        var += (float)(img->val[i] - mean) * (img->val[i] - mean);
    }
    var = var / img->n;
    return var;
}

iftImage *normalizeFingerprint(iftImage *img, int m0, float var0){
    float mean = imageMean(img);
    float var  = imageVar(img, mean);


    iftImage *normalized_img = iftCreateImageFromImage(img);
    for (int i = 0; i < img->n; i++){
        // if   (img->val[i] > mean) normalized_img->val[i] = (int)(m0 + sqrt(var0/var * (img->val[i] - mean) * (img->val[i] - mean)));
        // else                      normalized_img->val[i] = (int)(m0 - sqrt(var0/var * (img->val[i] - mean) * (img->val[i] - mean)));
        normalized_img->val[i] = (int)((img->val[i] - mean)/sqrt(var));
    }

    return normalized_img;
}


void RectifiedLinearUnit(iftMImage *mimg) {
  for (int p = 0; p < mimg->n; p++){
    for (int b = 0; b < mimg->m; b++) {
      if (mimg->val[p][b]<0)
	mimg->val[p][b]=0.0;
    }
  }
}

iftMImage *normalizeBandsByZscore(iftMImage *mimg) {
  iftMImage *normalized_mimg = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);


  // compute mean and standard deviation
  for (int b = 0; b < mimg->m; b++) {
    double var = 0; 
    double mean = 0;

    // mean
    for (int p = 0; p < mimg->n; p++){
      mean += (double)mimg->val[p][b];
    }
    mean /= (double)(mimg->n);

    // variance
    for (int p = 0; p < mimg->n; p++){
      var += pow((double)mimg->val[p][b] - mean, 2);
    }
    var /= (double)(mimg->n);
    double stdev = sqrt(var);

    for (int p = 0; p < mimg->n; p++){
      double z_score = ((double)mimg->val[p][b] - mean)/stdev;
      normalized_mimg->val[p][b] =  (int)(iftSigm(z_score)*255.0);
    }
  }
  
  return normalized_mimg;
}

iftMImage *normalizeVolumeByZscore(iftMImage *mimg) {
  iftMImage *normalized_mimg = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);


  // compute mean and standard deviation
  double var = 0; 
  double mean = 0;
  // mean
  for (int p = 0; p < mimg->n; p++){
    for (int b = 0; b < mimg->m; b++) {
      mean += (double)mimg->val[p][b];
    }
  }
  mean /= (double)(mimg->n * mimg->m);

  // variance
  for (int p = 0; p < mimg->n; p++){
    for (int b = 0; b < mimg->m; b++) {
      var += pow((double)mimg->val[p][b] - mean, 2);
    }
  }
  var /= (double)(mimg->n * mimg->m);

  double stdev = sqrt(var);
  printf("volume mean=%lf ,stdev=%lf\n", mean, stdev);

  // update values in normalized image
  for (int p = 0; p < mimg->n; p++) {
    for (int b = 0; b < mimg->m; b++) {

      // standerdized pixel value
      double z_score = ((double)mimg->val[p][b] - mean)/stdev;

      normalized_mimg->val[p][b] =  (int)(iftSigm(z_score)*255.0);
    }
    // printf("\n");
    // printf("band %d: mean=%f ,stdev=%lf\n", b, mean, stdev);
  }
  // iftWriteMImageBands(normalized_mimg, "output_gabor_band");
  return normalized_mimg;
}

iftImage *normalizeImageByZscore(iftImage *img) {
  iftImage *normalized_img = iftCreateImage(img->xsize, img->ysize, img->zsize);

  // image mean value
  float var = 0; 
  float mean = 0;
  for (int p = 0; p < img->n; p++){
    mean += (float)img->val[p];
  }
  mean /= (float)img->n;
  
  // image standard deviation
  for (int p = 0; p < img->n; p++){
    var += (float)pow((float)img->val[p] - mean, 2);
  }
  var /= ((float)img->n);
  float stdev = (float)sqrt(var);

  for (int i = 0; i < img->n; i++) {

    // standardized pixel value
    float z_score = ((float)img->val[i] - mean)/stdev;
    
    normalized_img->val[i] = (int)(255*iftSigm(z_score));
  }
  // printf("\n");
  printf("mean=%f ,stdev=%lf\n", mean, stdev);

  // iftWriteMImageBands(normalized_img, "output_gabor_band");
  return normalized_img;
}



void printFilteredImageBank(iftMImage *fimg_bank, char basename[], int filter_id){
  /*
  Prints a bank of filtered images from gabor filtering.
  */
  char filename[200];
  double max, min;
  max = IFT_INFINITY_DBL_NEG;
  min = IFT_INFINITY_DBL;
  int img_index_counter = 0;
  iftImage *img;
  
  for (int i = 0; i < fimg_bank->m; i = i + 2){
    img = iftCreateImage(fimg_bank->xsize, fimg_bank->ysize, fimg_bank->zsize);
    for (int j = 0; j < fimg_bank->n; j++){
      img->val[j] = sqrt(fimg_bank->val[j][i]*fimg_bank->val[j][i] + fimg_bank->val[j][i + 1] * fimg_bank->val[j][i + 1]);
      if (img->val[j] > max)
        max = img->val[j];
      if (img->val[j] < min)
        min = img->val[j];
    }
    for (int j = 0; j < fimg_bank->n; j++){
      img->val[j] = 255.0 * (img->val[j] - min)/(max - min);
    }
    if (basename[2] == 'a') // checking if printing images from space
      sprintf(filename,"%s_filtered_image-%d.png",basename, img_index_counter++);
    
    else // printing images from spectrum
      sprintf(filename,"%s_filtered_image-%d.png",basename, filter_id);
    
    iftWriteImageByExt(img, filename);
    iftDestroyImage(&img);
  
  
  }
}

void normalizeAndExtractMaskInPlace(iftMImage *mimg, iftImage *mask){
  iftMImage *aux =   iftCopyMImage(mimg);
  float meani = 0.0;
  float vari  = 0.0;

  // Calculating initial mean and var
  for (int p = 0; p < mimg->n; p++){
    meani += mimg->val[p][0]; // FP images are grayscale, they only have one band.
  }
  meani /= mimg->n;
  for (int p = 0; p < mimg->n; p++){
    vari += (mimg->val[p][0] - meani) * (mimg->val[p][0] - meani); 
  }
  vari /= mimg->n;

  // normalizing auxiliar image
  for (int p = 0; p < mimg->n; p++){
    aux->val[p][0] = (mimg->val[p][0] - meani) / sqrt(vari);
  }

  // finding segmentation mask using local variance
  iftAdjRel *A = iftRectangular(VARIANCE_BLK_SIZE, VARIANCE_BLK_SIZE);
  //iftAdjRel *A = iftCircular(5.0);
  for (int p = 0; p < mimg->n; p++){
    iftVoxel u = iftGetVoxelCoord(mask, p);

    int blk_counter = 0;
    float blk_mean  = 0.0;
    float blk_var   = 0.0;
    for (int i = 0; i < A->n; i++){
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(mask, v)){
        int q = iftGetVoxelIndex(mask, v);
        blk_mean += aux->val[q][0];
        blk_counter++;
      }
    }
    blk_mean /= blk_counter;
    for (int i = 0; i < A->n; i++){
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(mask, v)){
        int q = iftGetVoxelIndex(mask, v);
        blk_var += (aux->val[q][0] - blk_mean) * (aux->val[q][0] - blk_mean);
      }
    }
    blk_var /= blk_counter;
    
    if (blk_var > VARIANCE_THRESHOLD)
      mask->val[p] = 255;
  }

  // calculating new mean and variance, only in segmentation mask
  float meanf        = 0.0;
  float varf         = 0.0;
  int   mask_counter = 0;

  for (int p = 0; p < mimg->n; p++){
    if (mask->val[p]){
      meanf += mimg->val[p][0];
      mask_counter++;
    }
  }
  meanf /= mask_counter;

  for (int p = 0; p < mimg->n; p++){
    if (mask->val[p]){
      varf += (mimg->val[p][0] - meanf) * (mimg->val[p][0] - meanf);
    }
  }
  varf /= mask_counter;
  
  for (int p = 0; p < mimg->n; p++){
    if (mask->val[p]){
      mimg->val[p][0] = (mimg->val[p][0] - meanf) / sqrt(varf);
    }
    else 
      mimg->val[p][0] = 0;
  }

  

  iftDestroyMImage(&aux);
}

int main(int argc, char *argv[])
{
  timer           *tstart=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 5) {
    printf("usage iftSimpleGaborBank: <P1> <P2> <P3> <P4>\n");
    printf("<P1>: input grayscale image (.png)\n");
    printf("<P2>: number of different orientations\n");
    printf("<P3>: number of different frequencies\n");
    printf("<P4>: output image basename\n");
    exit(0);
  }

  char *basename   = Basename(argv[1]);
  char filename[200];
  tstart = iftTic();



  /* Reading input parameters */
  int n_orientations = atoi(argv[2]);
  int n_frequencies  = atoi(argv[3]);

  iftImage  *orig          = iftReadImageByExt(argv[1]);
  iftMImage *mimg          = iftImageToMImage(orig, GRAY_CSPACE);
  iftImage  *mask          = iftCreateImageFromImage(orig);
  normalizeAndExtractMaskInPlace(mimg, mask);

  /* Eliminates image frame present in some images */
  
  iftImage *mask_aux       = iftSelectLargestComp(mask, NULL);
  iftDestroyImage(&mask);
  mask = mask_aux;

  /* Convoluting with a bank of gabor filters */

  iftMImage   *padded_mimg = iftMAddFrame(mimg, GABOR_KERNEL_SIZE - 1, GABOR_KERNEL_SIZE - 1, 0, 0);
  iftMMKernel *K_bank      = iftGaborKernelBank(GABOR_KERNEL_SIZE, n_orientations, n_frequencies);
  iftMImage   *fimg_bank   = iftMMLinearFilter(padded_mimg, K_bank);
  iftMImage   *aux         = fimg_bank;
  fimg_bank                = iftMRemFrame(fimg_bank, (GABOR_KERNEL_SIZE - 1)/2, (GABOR_KERNEL_SIZE - 1)/2, 0);
  iftDestroyMImage(&aux);
  // iftMImage *fimg_bank_norm = normalizeVolumeByZscore(fimg_bank);

  /* Array to save band of max activation */
  int *bmax = (int *)calloc(mimg->n, sizeof(int));
  
  /* Constructing gabor image from band of activations */
  iftImage *filtered_image = getMaxGaborActivByPixelInWindow(fimg_bank, bmax, n_frequencies, mask);
  // iftImage *filtered_image = getSumGaborActivByPixelInWindow(fimg_bank, bmax, n_frequencies, mask);
  //iftImage *filtered_image = getFrequestGaborActivByPixelInWindow(fimg_bank, bmax, n_frequencies, mask);

  /* Estimating binary and Gabor enhancement image */

  iftImage* bin_image = iftThreshold(filtered_image,1,255,255);
    
  /* Saving results */

  sprintf(filename, "%s_filt.png", argv[4]);
  iftWriteImageByExt(filtered_image, filename);

  sprintf(filename, "%s_mask.png", argv[4]);
  iftWriteImageByExt(mask, filename);

  sprintf(filename, "%s_bin.png", argv[4]);
  iftWriteImageByExt(bin_image, filename);

  #ifdef DEBUG
    printKernelBank(K_bank, GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE);
    printFilteredImageBank(fimg_bank_norm, "space", 0);
  #endif

  
  iftDestroyImage(&orig);
  iftDestroyImage(&mask);
  iftDestroyImage(&filtered_image);
  iftDestroyImage(&bin_image);
  iftDestroyMImage(&mimg);
  iftDestroyMImage(&padded_mimg);
  iftDestroyMImage(&fimg_bank);

  free(bmax);
  iftDestroyMMKernel(&K_bank);
  iftFree(basename);
  


  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return 0;
}
