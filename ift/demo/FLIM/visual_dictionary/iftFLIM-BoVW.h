void MyReadMeanStdev(char *basepath, float *mean, float *stdev, int ninput_channels) {
    char filename[2][200];
    FILE *fp[2];

    sprintf(filename[0], "%s-mean.txt", basepath);
    sprintf(filename[1], "%s-stdev.txt", basepath);
    fp[0] = fopen(filename[0], "r");
    fp[1] = fopen(filename[1], "r");
    for (int b = 0; b < ninput_channels; b++) {
        if (fscanf(fp[0], "%f", &mean[b]) != 1);
        if (fscanf(fp[1], "%f", &stdev[b]) != 1);
    }
    fclose(fp[0]);
    fclose(fp[1]);
}

void MyWriteMeanStdev(char *basepath, float *mean, float *stdev, int ninput_channels) {
    char filename[2][200];
    FILE *fp[2];

    sprintf(filename[0], "%s-mean.txt", basepath);
    sprintf(filename[1], "%s-stdev.txt", basepath);
    fp[0] = fopen(filename[0], "w");
    fp[1] = fopen(filename[1], "w");
    for (int b = 0; b < ninput_channels; b++) {
        fprintf(fp[0], "%f ", mean[b]);
        fprintf(fp[1], "%f ", stdev[b]);
    }
    fclose(fp[0]);
    fclose(fp[1]);
}



iftMImage *MyReadInputMImage(char *filename) {
    iftImage  *img     = iftReadImageByExt(filename);
    iftMImage *input   = iftImageToMImage(img, LABNorm2_CSPACE); 
    iftDestroyImage(&img);    
    return (input);
}

iftMatrix *iftBoVWDistances(iftMatrix *A, iftMatrix *B)
{
  if (A->ncols != B->nrows) {
    iftError("Cannot apply soft assignment between %d x %d and %d x %d matrices", "iftSoftAssignment", A->nrows, A->ncols, B->nrows, B->ncols);
  }

  iftMatrix *M = iftCreateMatrix(B->ncols, A->nrows);
  
  for (int rowA=0; rowA < A->nrows; rowA++){
    for (int colB=0; colB < B->ncols; colB++){
      for (int rowB=0; rowB < B->nrows; rowB++){
	iftMatrixElem(M, colB, rowA) += (iftMatrixElem(A, rowB, rowA)-iftMatrixElem(B, colB, rowB))*
	  (iftMatrixElem(A, rowB, rowA)-iftMatrixElem(B, colB, rowB));
      }
      iftMatrixElem(M, colB, rowA) = exp(-sqrt(iftMatrixElem(M, colB, rowA)));
    }
  }

  
  return (M);
}

void MyNormalizeImageByZScore(iftMImage *img, float *mean, float *stdev) {
    int ninput_channels = img->m;

#pragma omp parallel for
    for (int p = 0; p < img->n; p++) {
        for (int b = 0; b < ninput_channels; b++) {
            img->val[p][b] = (img->val[p][b] - mean[b]) / stdev[b];
        }
    }
}

iftMImage **MyConvolutionalLayer(iftMImage **input, int nimages, iftFLIMArch *arch, int layer, int layer_index, int atrous_factor, char *param_dir)
{
  
  /* Set input parameters */

    iftAdjRel *A = iftFLIMAdaptiveAdjRelFromKernel(arch->layer[layer], atrous_factor, iftIs3DMImage(input[0]));
    char *basename = NULL;
    int ninput_channels = input[0]->m;
    iftMImage **output = (iftMImage **) calloc(nimages, sizeof(iftMImage *));

    /* Read parameters of the current layer */

    char filename[200];
    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftMatrix *kernelbank = iftReadMatrix(filename);
    float *mean = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    MyReadMeanStdev(filename, mean, stdev, ninput_channels);

    /* Apply convolutional layer */

    for (int i = 0; i < nimages; i++) {

      /* MyNormalizeImageByZScore(input[i], mean, stdev); */
      
      /* convolution */

      iftMatrix *imgM = iftMImageToFeatureMatrix(input[i], A, NULL);
      iftMatrix *conv = iftBoVWDistances(imgM,kernelbank);
      /* iftMatrix *conv = iftMultMatrices(imgM, kernelbank); */
      /* for (int row=0; row < conv->nrows; row++) */
      /* 	for (int col=0; col < conv->ncols; col++) */
      /* 	  if (iftMatrixElem(conv,col,row) < 0) */
      /* 	    iftMatrixElem(conv,col,row)=0; */
      
      iftDestroyMatrix(&imgM);
      iftMImage *activ = iftMatrixToMImage(conv, input[i]->xsize, input[i]->ysize, input[i]->zsize, kernelbank->ncols,
					   'c');
      iftDestroyMatrix(&conv);
      
      /* pooling */
      
      if (strcmp(arch->layer[layer].pool_type, "no_pool") != 0) {
	iftMImage *pool = NULL;
	if (strcmp(arch->layer[layer].pool_type, "avg_pool") == 0) { /* ignore the stride to learn the model */
	  pool = iftFLIMAtrousAveragePooling(activ, arch->layer[layer].pool_size[0], arch->layer[layer].pool_size[1], arch->layer[layer].pool_size[2], atrous_factor, arch->layer[layer].pool_stride);
	  iftDestroyMImage(&activ);
	  activ = pool;
	} else {
	  if (strcmp(arch->layer[layer].pool_type, "max_pool") == 0) { /* ignore the stride to learn the model */
	    pool = iftFLIMAtrousMaxPooling(activ, arch->layer[layer].pool_size[0], arch->layer[layer].pool_size[1], arch->layer[layer].pool_size[2], atrous_factor, arch->layer[layer].pool_stride);
	    iftDestroyMImage(&activ);
	    activ = pool;
	  } else {
	    iftWarning("Invalid pooling type has been ignore", "FLIMConvolutionalLayer");
	  }
	}
      }

      output[i] = iftCopyMImage(activ);
      iftDestroyMImage(&activ);	
    }

    iftDestroyAdjRel(&A);
    iftDestroyMatrix(&kernelbank);
    iftFree(mean);
    iftFree(stdev);

    return output;
}

