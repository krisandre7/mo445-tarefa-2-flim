#include "ift.h"
iftImage **MImagetoImages(iftMImage *mimg)
{
  iftImage **img = (iftImage **)(calloc(mimg->m, sizeof(iftImage)));
  double min_weights[mimg->m];
  double max_weights[mimg->m];
  double aux[mimg->m][mimg->n];
  

  for (int i = 0; i < mimg->m; i++){
    min_weights[i] = IFT_INFINITY_DBL;
    max_weights[i] = IFT_INFINITY_DBL_NEG;
  }

  for (int i = 0; i < mimg->m; i++){
    img[i] = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    for (int p = 0; p < mimg->n; p++){
      aux[i][p] = mimg->val[p][i];
      if (aux[i][p] < min_weights[i]){
        min_weights[i] = aux[i][p];
      }

      if (aux[i][p] > max_weights[i]){
        max_weights[i] = aux[i][p];
      }
    }
  }
  for (int i = 0; i < mimg->m; i++){
    for (int p = 0; p < mimg->n; p++){
      img[i]->val[p] = (int)(255.0 * (aux[i][p] - min_weights[i])/(max_weights[i] - min_weights[i]));
      // printf("%d\t", img[i]->val[p]);
    }
  }

  return(img);
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

void printKernelBank(iftMMKernel *K_bank, int xsize, int ysize){
  /*
  Prints a Gabor iftMMKernel.
  Each kernel has 2 bands representing the real and the imaginary parts 
  */
  char filename[200];
  iftVoxel u;
  int j, p, dmin[3];
  int img_index_counter = 0;
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
    if (i%2==0){
      sprintf(filename,"space_kernel-%d-real.png", img_index_counter);
      iftWriteImageByExt(img, filename);
    }
    else{
      sprintf(filename,"space_kernel-%d-imag.png",img_index_counter++);
      iftWriteImageByExt(img, filename);
    }
    iftDestroyImage(&img);
  }
  
}

void printGaborFilter(iftMImage *gabor, int filter_id){
  /*
  Prints a single Gabor Filter created by iftGaborFilter2D in Spectrum.c
  */
  char filename[200];
  iftImage **gaborFilt = MImagetoImages(gabor);
  sprintf(filename,"spec_filter-%d-real.png", filter_id); 
  iftWriteImageByExt(gaborFilt[0],filename);

  sprintf(filename,"spec_filter-%d-imag.png", filter_id); 
  iftWriteImageByExt(gaborFilt[1],filename);
  
  iftDestroyImage(&gaborFilt[0]);
  iftDestroyImage(&gaborFilt[1]);

}

void printGaborSpectrum(iftMImage *spec_gabor, int filter_id){
  /*
  Prints the fourier transform of a given gabor filter.
  */
  char filename[200];
  iftImage *spec_gabor_mag = iftViewLogMagnitude(spec_gabor);

  sprintf(filename,"spec_magnitude-%d.png", filter_id); 
  iftWriteImageByExt(spec_gabor_mag,filename);

  iftDestroyImage(&spec_gabor_mag);
}

float **readGaborParamsList(int n_params, char* filename){
  /*
  Reads a gabor list of parameters. Configuration file is a txt file as follows:
  u0_1 v0_1 a_1 b_1 theta_1 P_1 x0_1 y0_1
  u0_2 v0_2 a_2 b_2 theta_2 P_2 x0_2 y0_2
  ...
  Ex:
  0.113 0.054 0.1 0.1 0.0 0.0 0 0
  0.106 0.087 0.1 0.1 0.0 0.0 0 0

  The number of the txt file lines must be equal to the n_params input parameter.

  */
  FILE *fp = fopen(filename, "r");
  if(!fp)
      iftError("Unable to open file", "readGaborParamsList");

  float **params = (float **)(calloc(n_params, sizeof(float*)));
  for (int i =0; i < n_params; i++){
    params[i] = (float *)(calloc(8, sizeof(float)));
  }

  for (int i=0; i < n_params; i++)
    if (fscanf(fp,"%f %f %f %f %f %f %f %f",&params[i][0], &params[i][1], &params[i][2], &params[i][3], &params[i][4], &params[i][5], &params[i][6], &params[i][7])!=8)
        iftError("Reading error", "readGaborParamsList");

  fclose(fp);

  return(params);
}

char *Basename(char *path)
{
  char *basename     = iftBasename(path);
  iftSList *slist    = iftSplitString(basename,"/");
  strcpy(basename,slist->tail->elem);
  iftDestroySList(&slist);
  return(basename);
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
  
  if (argc != 6) {
    char msg[500];
    sprintf(msg, "iftGaborBank <P1> <P2> <P3>\n P1: input image\n P2: xsize of the spatial Gabor filters\n P3: ysize of the Gabor spatial filters\n P4: number of gabor filters\n P5: gabor parameters configuration file");
    iftError(msg,"main");
  }

  char *basename   = Basename(argv[1]);
  tstart = iftTic();

  // input parameters
  int xsize    = atof(argv[2]); // x size of gabor convolutional filter
  int ysize    = atof(argv[3]); // y size of gabor convolutional filter
  int n_params = atoi(argv[4]); 

  float **gabor_parameters = readGaborParamsList(n_params, argv[5]);
  
  iftImage  *orig  = iftReadImageByExt(argv[1]);
  iftMImage *mimg  = iftImageToMImage(orig, GRAY_CSPACE);
  iftDestroyImage(&orig);

  timer           *space_start=NULL;
  //timer           *freq_start=NULL;
  

  // ------------------------------ SPACE METHOD --------------------
  // Bank of kernels via matrix mult
  space_start = iftTic();

  iftMImage   *padded_mimg = iftMAddFrame(mimg, xsize - 1, ysize - 1, 0, 0);
  iftMMKernel *K_bank      = iftGaborMMKernelBank2D(xsize, ysize, n_params, gabor_parameters, mimg->m);
  iftMImage   *fimg_bank   = iftMMLinearFilter(padded_mimg, K_bank);

  fimg_bank = iftMRemFrame(fimg_bank, xsize - 1, ysize - 1, 0);
  printf("Gabor bank finished in: \n");
  puts(iftFormattedTime(iftCompTime(space_start, iftToc())));


  printKernelBank(K_bank, xsize, ysize);
  printFilteredImageBank(fimg_bank, "space", 0);

  iftDestroyMImage(&padded_mimg);
  iftDestroyMMKernel(&K_bank);
  iftDestroyMImage(&fimg_bank);
  printf("----------------------------------------------------\n");
  // // ----------------- FREQUENCY METHOD ---------------------
  // fixed parameters
  
  // int filter_id = 0;
  // iftMImage *gabor;
  // iftMImage *spec_gabor;
  // iftMImage *spec_image;
  // iftMImage *mult_spec;
  // iftMImage *filtered_image;

  // freq_start = iftTic();

  // for (int i = 0; i < n_params; i++){

  //   gabor = iftGaborFilter2D(mimg, gabor_parameters[i][0], gabor_parameters[i][1], gabor_parameters[i][5], gabor_parameters[i][6], gabor_parameters[i][7], gabor_parameters[i][2], gabor_parameters[i][3], gabor_parameters[i][4]); // projecting filter in space domain
  //   spec_gabor = iftFFT2D(gabor); // transforming to frequency domain
  //   spec_image   = iftFFT2D(mimg); // transforming input image to frequency domain
  //   mult_spec   = iftMultSpectra(spec_image,spec_gabor); // filtering in frequency domain
  //   filtered_image    = iftInvFFT2D(mult_spec); // returning filtered image to space domain
    
  //   // Saving output images: gabor filters, gabor spectrum and filtered image
  //   printGaborFilter(gabor, filter_id); // gabor filters in space
  //   printGaborSpectrum(spec_gabor,filter_id); // gabor spectrum
  //   printFilteredImageBank(filtered_image, "spec", filter_id); // filtered images

  //   filter_id++;

  //   // Clearing memory
  //   iftDestroyMImage(&gabor);
  //   iftDestroyMImage(&spec_gabor);
  //   iftDestroyMImage(&spec_image);
  //   iftDestroyMImage(&mult_spec);
  //   iftDestroyMImage(&filtered_image);
  // }

  
  // printf("Tempo de execução do algoritmo usando FFT: \n");
  // puts(iftFormattedTime(iftCompTime(freq_start, iftToc())));


  iftFree(basename);
  iftDestroyMImage(&mimg);
  for (int i = 0; i < n_params; i++) iftFree(gabor_parameters[i]);
  


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








