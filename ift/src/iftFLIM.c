#include "iftFLIM.h"
#include "iftFLIM.cuh"

iftLabeledSet *LabelMarkers(iftMImage *img, iftLabeledSet *S);

void NormalizeImagePerBand(iftMImage *img, iftImage *label);

iftMatrix *LearnKernelBank(iftMImage *input, iftLabeledSet *M, iftAdjRel *A, int nsamples, int nkernels_per_image, int nkernels_per_marker);

iftMatrix *ConsensusKernelbank(iftFileSet *fs_seeds, char *inputdata_dir, int noutput_channels, float stdev_factor);

iftImage *SubsampleImage(iftImage *img, int stride);

void MImageToFeatureMatrixAtRow(iftMImage *mimg, iftAdjRel *A, iftMatrix *matrix, int initRow);

void FLIMExtractFeaturesPerImage(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                 char *param_dir, char *feat_dir, char *object_dir);

void FLIMExtractFeaturesFromLayerPerImage(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                 char *param_dir, int layer_index, char *feat_dir, char *object_dir);

void FLIMExtractFeaturesPerBatch(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                 char *param_dir, char *feat_dir, char *object_dir);

void FLIMExtractFeaturesFromLayerPerBatch(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                          char *param_dir, int layer_index, char *feat_dir, char *object_dir);

iftMImage **
FLIMConvolutionalLayer(iftMImage **input, int nimages, iftImage **mask, iftFLIMArch *arch, int layer, int layer_index, int atrous_factor, char *param_dir);

iftMatrix *KernelBankByPCA(iftDataSet *Z, int *nkernels);





/* ------------ private functions --------------------------------- */

iftMatrix *KernelBankByPCA(iftDataSet *Z, int *nkernels)
{
  /* Estimate the initial kernel bank by PCA */

  iftMatrix *kern_pca  = iftRotationMatrixByPCA(Z);
  iftMatrix *kern_rot  = iftTransposeMatrix(kern_pca);
  iftDestroyMatrix(&kern_pca);
  *nkernels            = iftMin(*nkernels, kern_rot->ncols);
  iftMatrix *kern_bank = iftCreateMatrix(*nkernels, kern_rot->nrows);
  for (int col = 0; col < *nkernels; col++) {
    for (int row = 0; row < kern_rot->nrows; row++) {
      iftMatrixElem(kern_bank, col, row) = iftMatrixElem(kern_rot, col, row);
    }
  }
  iftDestroyMatrix(&kern_rot);
  
  return(kern_bank);
}

iftAdjRel *iftFLIMAdjRelFromKernel(iftFLIMLayer layer, bool dim3D) {
    iftAdjRel *A;

    if (dim3D) {
        A = iftCuboidWithDilationForConv(layer.kernel_size[0], layer.kernel_size[1], layer.kernel_size[2],
                                  layer.dilation_rate[0], layer.dilation_rate[1], layer.dilation_rate[2]);
    } else {
        A = iftRectangularWithDilationForConv(layer.kernel_size[0], layer.kernel_size[1], layer.dilation_rate[0],
                                       layer.dilation_rate[1]);
    }

    return (A);
}

iftAdjRel *iftFLIMAdaptiveAdjRelFromKernel(iftFLIMLayer layer, int atrous_factor, bool dim3D)
{
    iftAdjRel *A;

    if (dim3D) {
        A = iftCuboidWithDilationForConv(layer.kernel_size[0], layer.kernel_size[1], layer.kernel_size[2],
                                  layer.dilation_rate[0]*atrous_factor,
                                  layer.dilation_rate[1]*atrous_factor,
                                  layer.dilation_rate[2]*atrous_factor);
    } else {
        A = iftRectangularWithDilationForConv(layer.kernel_size[0], layer.kernel_size[1], layer.dilation_rate[0]*atrous_factor,
                                       layer.dilation_rate[1]*atrous_factor);
    }

    return (A);
}

void MImageToFeatureMatrixAtRow(iftMImage *mimg, iftAdjRel *A, iftMatrix *matrix, int initRow) {
#pragma omp parallel for
    for (int p = 0; p < mimg->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mimg, p);
        for (int i = 0; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);
                for (int b = 0; b < mimg->m; b++) {
                    iftMatrixElem(matrix, b + i * mimg->m, initRow + p) = mimg->val[q][b];
                }
            }
        }
    }
}

void FLIMExtractFeaturesPerBatch(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                 char *param_dir, char *feat_dir, char *object_dir) {
    iftMImage **input = NULL, **output = NULL;
    iftImage **mask = NULL;
    char filename[200];
    int nimages = last - first + 1;
    char ext[20];
    sprintf(ext, "%s", iftFileExt(fs_images->files[0]->path));
    char **basename = (char **) calloc(nimages, sizeof(char *));
    input = (iftMImage **) calloc(nimages, sizeof(iftMImage *));
    output = NULL;
    if (object_dir != NULL)
        mask = (iftImage **) calloc(nimages, sizeof(iftImage *));

    /* Load batch */

    for (int i = first, k = 0; i <= last; i++, k++) {
        sprintf(filename, "%s/%s", orig_dir, fs_images->files[i]->path);
	if (strcmp(ext,".mimg") == 0){
	  input[k] = iftReadMImage(filename);
        }else{
	  input[k] = iftReadInputMImage(filename);
        }
        basename[k] = iftFilename(fs_images->files[i]->path, ext);
        if (mask != NULL) {
	        sprintf(filename, "%s/%s%s", object_dir, basename[k],ext);
            if (iftFileExists(filename)) {
                mask[k] = iftReadImageByExt(filename);
                int stride = mask[k]->xsize / input[k]->xsize;
                if (stride > 1) {
                    iftImage *aux = SubsampleImage(mask[k], stride);
                    iftDestroyImage(&mask[k]);
                    mask[k] = aux;
                }
            }
        }
    }

    /* For each layer do */

    for (int l = 0; l < arch->nlayers; l++) {
      output = FLIMConvolutionalLayer(input, nimages, mask, arch, l, l+1, 1, param_dir);
      for (int k = 0; k < nimages; k++) {
	iftDestroyMImage(&input[k]);
	input[k] = output[k];
	output[k] = NULL;
      }
      iftFree(output);
    }

    for (int k = 0; k < nimages; k++) {
        sprintf(filename, "%s/%s.mimg", feat_dir, basename[k]);
        iftFree(basename[k]);
        iftWriteMImage(input[k], filename);
        iftDestroyMImage(&input[k]);
        if (mask != NULL)
            iftDestroyImage(&mask[k]);
    }
    iftFree(mask);
    iftFree(input);
    iftFree(basename);
}

void FLIMExtractFeaturesFromLayerPerBatch(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                 char *param_dir, int layer_index, char *feat_dir, char *object_dir) {
    iftMImage **input = NULL, **output = NULL;
    iftImage **mask = NULL;
    char filename[200];
    int nimages = last - first + 1;
    char ext[20];
    sprintf(ext, "%s", iftFileExt(fs_images->files[0]->path));
    char **basename = (char **) calloc(nimages, sizeof(char *));
    input = (iftMImage **) calloc(nimages, sizeof(iftMImage *));
    output = NULL;
    if (object_dir != NULL)
        mask = (iftImage **) calloc(nimages, sizeof(iftImage *));

    /* Load batch */

    for (int i = first, k = 0; i <= last; i++, k++) {
        sprintf(filename, "%s/%s", orig_dir, fs_images->files[i]->path);
        if (strcmp(ext,".mimg") == 0){
            input[k] = iftReadMImage(filename);
        }else{
            input[k] = iftReadInputMImage(filename);
        }
        basename[k] = iftFilename(fs_images->files[i]->path, ext);
        if (mask != NULL) {
            if (iftIs3DMImage(input[k])){
                sprintf(filename, "%s/%s.nii.gz", object_dir, basename[k]);
            } else {
                sprintf(filename, "%s/%s.png", object_dir, basename[k]);
            }
            if (iftFileExists(filename)) {
                mask[k] = iftReadImageByExt(filename);
                int stride = mask[k]->xsize / input[k]->xsize;
                if (stride > 1) {
                    iftImage *aux = SubsampleImage(mask[k], stride);
                    iftDestroyImage(&mask[k]);
                    mask[k] = aux;
                }
            }
        }
    }

    /* For each layer do */

    output = FLIMConvolutionalLayer(input, nimages, mask, arch, 0, layer_index, 1, param_dir);
    for (int k = 0; k < nimages; k++) {
        iftDestroyMImage(&input[k]);
        input[k] = output[k];
        output[k] = NULL;
    }
    iftFree(output);

    for (int k = 0; k < nimages; k++) {
        sprintf(filename, "%s/%s.mimg", feat_dir, basename[k]);
        iftFree(basename[k]);
        iftWriteMImage(input[k], filename);
        iftDestroyMImage(&input[k]);
        if (mask != NULL)
            iftDestroyImage(&mask[k]);
    }
    iftFree(mask);
    iftFree(input);
    iftFree(basename);
}

void FLIMExtractFeaturesPerImage(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                 char *param_dir, char *feat_dir, char *object_dir) {
    iftMImage **input = NULL, **output = NULL;
    iftImage **mask = NULL;
    char filename[200];

    printf("\n");

    for (int i = first; i <= last; i++) {
        input = (iftMImage **) calloc(1, sizeof(iftMImage *));     
	char ext[20];
	sprintf(ext, "%s", iftFileExt(fs_images->files[i]->path));
	sprintf(filename, "%s/%s", orig_dir, fs_images->files[i]->path);
	if (strcmp(ext,".mimg") == 0){
	  input[0] = iftReadMImage(filename);
	}else{
	  input[0] = iftReadInputMImage(filename);
        }
        char *basename = iftFilename(fs_images->files[i]->path, ext);
        if (object_dir != NULL) {
            mask = (iftImage **) calloc(1, sizeof(iftImage *));
	        sprintf(filename, "%s/%s%s", object_dir, basename, ext);
            if (iftFileExists(filename)) {
                mask[0] = iftReadImageByExt(filename);
                int stride = mask[0]->xsize / input[0]->xsize;
                if (stride > 1) {
                    iftImage *aux = SubsampleImage(mask[0], stride);
                    iftDestroyImage(&mask[0]);
                    mask[0] = aux;
                }
            }
        }

        printf("Processing file %s: image %d of %d files\n", basename, i + 1, last - first + 1);
        fflush(stdout);

        for (int l = 0; l < arch->nlayers; l++) {
	  output = FLIMConvolutionalLayer(input, 1, mask, arch, l, l+1, 1, param_dir);
	  iftDestroyMImage(&input[0]);
	  input[0] = output[0];
	  output[0] = NULL;
	  iftFree(output);
        }

        sprintf(filename, "%s/%s.mimg", feat_dir, basename);
        iftFree(basename);
        iftWriteMImage(input[0], filename);
        iftDestroyMImage(&input[0]);
        iftFree(input);
        if (mask != NULL) {
            iftDestroyImage(&mask[0]);
            iftFree(mask);
        }
    }
}

void FLIMExtractFeaturesFromLayerPerImage(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                 char *param_dir, int layer_index, char *feat_dir, char *object_dir) {
    iftMImage **input = NULL, **output = NULL;
    iftImage **mask = NULL;
    char filename[200];

    printf("\n");

    for (int i = first; i <= last; i++) {

        char ext[20];
        sprintf(ext, "%s", iftFileExt(fs_images->files[i]->path));
        sprintf(filename, "%s/%s", orig_dir, fs_images->files[i]->path);
        input = (iftMImage **) calloc(1, sizeof(iftMImage *));
        if (strcmp(ext,".mimg") == 0){
            input[0] = iftReadMImage(filename);
        }else{
            input[0] = iftReadInputMImage(filename);
        }
        char *basename = iftFilename(fs_images->files[i]->path, ext);
        if (object_dir != NULL) {
            mask = (iftImage **) calloc(1, sizeof(iftImage *));
            if (iftIs3DMImage(input[0])){
                sprintf(filename, "%s/%s.nii.gz", object_dir, basename);
            } else {
                sprintf(filename, "%s/%s.png", object_dir, basename);
            }
            if (iftFileExists(filename)) {
                mask[0] = iftReadImageByExt(filename);
                int stride = mask[0]->xsize / input[0]->xsize;
                if (stride > 1) {
                    iftImage *aux = SubsampleImage(mask[0], stride);
                    iftDestroyImage(&mask[0]);
                    mask[0] = aux;
                }
            }
        }

        printf("Processing file %s: image %d of %d files\n", basename, i + 1, last - first + 1);
        fflush(stdout);

        output = FLIMConvolutionalLayer(input, 1, mask, arch, 0, layer_index, 1, param_dir);
        iftDestroyMImage(&input[0]);
        input[0] = output[0];
        output[0] = NULL;
        iftFree(output);

        sprintf(filename, "%s/%s.mimg", feat_dir, basename);
        iftFree(basename);
        iftWriteMImage(input[0], filename);
        iftDestroyMImage(&input[0]);
        iftFree(input);
        if (mask != NULL) {
            iftDestroyImage(&mask[0]);
            iftFree(mask);
        }
    }
}

iftMImage *iftReadInputMImage(char *filename) {
    iftImage  *img     = iftReadImageByExt(filename);
    iftMImage *input   = iftImageToMImage(img, LABNorm2_CSPACE); 
    iftDestroyImage(&img);    
    return (input);
}

iftLabeledSet *LabelMarkers(iftMImage *img, iftLabeledSet *S) {
    iftLabeledSet *M = NULL, *seed = S;
    iftImage *markers = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iftAdjRel *A = NULL;

    if (iftIs3DMImage(img))
        A = iftSpheric(sqrtf(3.0));
    else
        A = iftCircular(sqrtf(2.0));

    while (seed != NULL) {
        int p = seed->elem;
        markers->val[p] = 255;
        seed = seed->next;
    }

    iftImage *lbmarkers = iftLabelComp(markers, A);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&markers);

    seed = S;
    while (seed != NULL) {
        int p = seed->elem;
        iftInsertLabeledSet(&M, p, lbmarkers->val[p]);
        seed = seed->next;
    }

    iftDestroyImage(&lbmarkers);

    return (M);
}

iftDataSet *iftComputeSeedDataSet(iftMImage *img, iftLabeledSet *S, iftAdjRel *A, int nsamples) {
  int tensor_size = img->m * A->n;
  iftDataSet *Z = iftCreateDataSet(nsamples, tensor_size);
  int ninput_channels = img->m;

  Z->nclasses = 0;
  int s       = 0;
  iftLabeledSet *seed = S;
  while (seed != NULL) {
    int p = seed->elem;
    Z->sample[s].id = s;
    Z->sample[s].truelabel = seed->label;
    if (Z->sample[s].truelabel > Z->nclasses)
      Z->nclasses = Z->sample[s].truelabel;
    iftVoxel u = iftMGetVoxelCoord(img, p);
    int j = 0;
    for (int k = 0; k < A->n; k++) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, k);
      if (iftMValidVoxel(img, v)) {
	int q = iftMGetVoxelIndex(img, v);
	for (int b = 0; b < ninput_channels; b++) {
	  Z->sample[s].feat[j] = img->val[q][b];
	  j++;
	}
      } else {
	for (int b = 0; b < img->m; b++) {
	  Z->sample[s].feat[j] = 0;
	  j++;
	}
      }
    }
    s++;
    seed = seed->next;
  }
  
  iftSetStatus(Z, IFT_TRAIN);
  iftAddStatus(Z, IFT_SUPERVISED);
  
  return (Z);
}

iftMatrix *iftComputeKernelBank(iftDataSet *Z, int *ngroups, uchar grouping_method, iftKmeansDistance distance) {
  iftMatrix *kernels = NULL;
  iftGraph *graph = NULL; /* For IOPF and IW */
  iftKnnGraph *knngraph = NULL;
  int i = 0;

  if (distance == NULL)
    distance = iftEuclideanDistance;
  
  iftRandomSeed(42);

  if (*ngroups >= Z->nsamples) { /* use all markers instead */
    *ngroups = Z->nsamples;
    kernels  = iftCreateMatrix(*ngroups, Z->nfeats);
    for (int s = 0; s < Z->nsamples; s++) {
      iftUnitNorm(Z->sample[s].feat, Z->nfeats);
      for (int j = 0; j < Z->nfeats; j++){
	iftMatrixElem(kernels, i, j) = Z->sample[s].feat[j];
      }
      i++;
    }
    return(kernels);
  }
  
  switch (grouping_method) {
    
  case 0: /* KMeans */
    kernels = iftCreateMatrix(*ngroups, Z->nfeats);
    iftKMeans(Z, *ngroups, 100, 0.001, NULL, NULL, distance);
    for (int s = 0; s < Z->nsamples; s++) {
      if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)) {
        iftUnitNorm(Z->sample[s].feat, Z->nfeats);
        for (int j = 0; j < Z->nfeats; j++){
          iftMatrixElem(kernels, i, j) = Z->sample[s].feat[j];
        }
        i++;
      }
    }
    
      
    break;

  case 1: /* IW in a complete graph */
      
    iftAddStatus(Z,IFT_TRAIN);
    graph = iftCreateGraph(Z);
    graph->is_complete = true;
    iftIteratedWatersheds(graph, *ngroups, 100, IFT_MAX, IFT_DATA_FEAT, true);
    kernels = iftCreateMatrix(*ngroups, Z->nfeats);
    for (int i = 0; i < *ngroups; i++) {
      int u = graph->centroids[i];
      int s = graph->node[u].sample;
      //iftUnitNorm(Z->sample[s].feat, Z->nfeats);
      for (int j = 0; j < Z->nfeats; j++)
	iftMatrixElem(kernels, i, j) = Z->sample[s].feat[j];
    }
    iftDestroyGraph(&graph);
    
    break;
    
  case 2: /* OPF C Clusters */
    
    iftAddStatus(Z,IFT_TRAIN);
    knngraph = iftCreateKnnGraph(Z,iftMax(Z->nsamples-1,1));
    *ngroups = iftUnsupTrainWithCClusters(knngraph, *ngroups);
    
    kernels = iftCreateMatrix(*ngroups, Z->nfeats);
    for (int s = 0; s < Z->nsamples; s++) {
      if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)) {
	//iftUnitNorm(Z->sample[s].feat, Z->nfeats);
	for (int j = 0; j < Z->nfeats; j++){
	  iftMatrixElem(kernels, i, j) = Z->sample[s].feat[j];
	}
	i++;
      }
    }
    iftDestroyKnnGraph(&knngraph);
    
    break;
    
  default:
    iftError("Grouping method must be 0 for KMeans, 1 for IW", "iftComputeKernelBank");
  }
  
  return (kernels);
}

void iftNormalizeImageByZScore(iftMImage *img, float *mean, float *stdev) {
    int ninput_channels = img->m;

#pragma omp parallel for
    for (int p = 0; p < img->n; p++) {
        for (int b = 0; b < ninput_channels; b++) {
            img->val[p][b] = (img->val[p][b] - mean[b]) / stdev[b];
        }
    }
}

void NormalizeImagePerBand(iftMImage *img, iftImage *label) {
    if (label == NULL) {
        for (int b = 0; b < img->m; b++) {
            float Imin = IFT_INFINITY_FLT, Imax = IFT_INFINITY_FLT_NEG;
            for (int p = 0; p < img->n; p++) {
                if (img->val[p][b] > Imax)
                    Imax = img->val[p][b];
                if ((img->val[p][b] < Imin) && (img->val[p][b] > 0))
                    Imin = img->val[p][b];
            }
            if (Imin < Imax)
                for (int p = 0; p < img->n; p++) {
                    img->val[p][b] = (img->val[p][b] - Imin) / (Imax - Imin);
                    if (img->val[p][b] < 0)
                        img->val[p][b] = 0;
                }
        }
    } else {
        for (int b = 0; b < img->m; b++) {
            float Imin = IFT_INFINITY_FLT, Imax = IFT_INFINITY_FLT_NEG;
            for (int p = 0; p < img->n; p++) {
                if (label->val[p] != 0) {
                    if (img->val[p][b] > Imax)
                        Imax = img->val[p][b];
                    if ((img->val[p][b] < Imin) && (img->val[p][b] > 0))
                        Imin = img->val[p][b];
                }
            }
            if (Imin < Imax)
                for (int p = 0; p < img->n; p++) {
                    if (label->val[p] != 0) {
                        img->val[p][b] = (img->val[p][b] - Imin) / (Imax - Imin);
                        if (img->val[p][b] < 0)
                            img->val[p][b] = 0;
                    } else {
                        img->val[p][b] = 0;
                    }
                }
        }
    }
}

iftMatrix *iftSelectRelevantKernelsByPCA(iftMatrix *kernels, int number_of_kernels,float stdev_factor) {
    iftMatrix *kern_t = iftTransposeMatrix(kernels);
    iftDataSet *Z = iftFeatureMatrixToDataSet(kern_t);
    iftDestroyMatrix(&kern_t);

    iftSetStatus(Z, IFT_TRAIN);
    iftNormalizeDataSetByZScoreInPlace(Z,NULL,stdev_factor);

    iftMatrix *kern_pca = iftRotationMatrixByPCA(Z);
    iftMatrix *kern_rot = iftTransposeMatrix(kern_pca);
    iftDestroyMatrix(&kern_pca);

    number_of_kernels = iftMin(number_of_kernels, kern_rot->ncols);
    iftMatrix *M = iftCreateMatrix(number_of_kernels, kern_rot->nrows);

    for (int k = 0; k < number_of_kernels; k++) {
        for (int j = 0; j < kern_rot->nrows; j++) {
            iftMatrixElem(M, k, j) = iftMatrixElem(kern_rot, k, j);
        }
    }

    iftDestroyMatrix(&kern_rot);
    iftDestroyDataSet(&Z);
    
    return (M);
}

iftMatrix *iftSelectRelevantKernelsByKmeans(iftMatrix *K, int ngroups, iftKmeansDistance distance) {
    iftMatrix *Kt = iftTransposeMatrix(K);
    iftDataSet *Z = iftFeatureMatrixToDataSet(Kt);
    iftDestroyMatrix(&Kt);
    
    if (distance == NULL)
    distance = iftEuclideanDistance;

    iftRandomSeed(42);


    iftMatrix *kernels = iftCreateMatrix(ngroups, Z->nfeats);

    iftKMeans(Z, ngroups, 100, 0.001, NULL, NULL, distance);
    int i = 0;
    for (int s = 0; s < Z->nsamples; s++) {
        if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)) {
	  //iftUnitNorm(Z->sample[s].feat, Z->nfeats);
            for (int j = 0; j < Z->nfeats; j++)
                iftMatrixElem(kernels, i, j) = Z->sample[s].feat[j];
            i++;
        }
    }
    iftDestroyDataSet(&Z);
    return (kernels);
}

iftMatrix *LearnKernelBank(iftMImage *input, iftLabeledSet *M, iftAdjRel *A, int nsamples, int nkernels_per_image, int nkernels_per_marker) {
    int tensor_size     = input->m * A->n;
    iftDataSet *Z       = iftComputeSeedDataSet(input, M, A, nsamples);
    iftMatrix **kernels = (iftMatrix **) calloc(Z->nclasses + 1, sizeof(iftMatrix *));
    int total_nkernels  = 0;
    
    for (int c = 1; c <= Z->nclasses; c++) {
        iftDataSet *Z1  = iftExtractSamplesFromClass(Z, c);	
        int nkernels    = nkernels_per_marker;
	/* 0: kmeans, 1: iterated watershed and 2: OPF c clusters */
	//kernels[c]      = iftComputeKernelBank(Z1, &nkernels, 0, iftCosineDistance2);
	kernels[c]      = iftComputeKernelBank(Z1, &nkernels, 0, iftEuclideanDistance);
        total_nkernels += nkernels;
        iftDestroyDataSet(&Z1);
    }

    iftMatrix *kernelbank = iftCreateMatrix(total_nkernels, tensor_size);

    int k = 0;
    for (int c = 1; c <= Z->nclasses; c++) {
        for (int col = 0; col < kernels[c]->ncols; col++, k++) {
            for (int row = 0; row < kernels[c]->nrows; row++) {
                iftMatrixElem(kernelbank, k, row) = iftMatrixElem(kernels[c], col, row);
            }
        }
    }

    for (int c = 0; c <= Z->nclasses; c++) {
        iftDestroyMatrix(&kernels[c]);
    }
    iftFree(kernels);
    iftDestroyDataSet(&Z);
	   
    if (kernelbank->ncols > nkernels_per_image) { /* force a number of kernels per image */
      //      iftMatrix *Rkernels = iftSelectRelevantKernelsByKmeans(kernelbank, nkernels_per_image, iftCosineDistance2);
      iftMatrix *Rkernels = iftSelectRelevantKernelsByKmeans(kernelbank, nkernels_per_image, iftEuclideanDistance);
      iftDestroyMatrix(&kernelbank);
      kernelbank = Rkernels;
    }
    
   return (kernelbank);
}

iftMImage **
FLIMConvolutionalLayer(iftMImage **input, int nimages, iftImage **mask, iftFLIMArch *arch, int layer, int layer_index, int atrous_factor, char *param_dir) {
    /* Set input parameters */

    iftAdjRel *A = iftFLIMAdaptiveAdjRelFromKernel(arch->layer[layer], atrous_factor, iftIs3DMImage(input[0]));
    int ninput_channels = input[0]->m;
    char *basename = NULL;
    iftMImage **output = (iftMImage **) calloc(nimages, sizeof(iftMImage *));

    /* Read consensus parameters of the current layer */

    char filename[200];
    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftMatrix *kernelbank = iftReadMatrix(filename);
    float *mean  = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftReadMeanStdev(filename, mean, stdev, ninput_channels);
    /* BIAS: use environment variable to use the bias idea */
    float *bias = NULL;
    char *use_bias = getenv("USE_BIAS");

    if (use_bias!=NULL){
    	bias = iftReadBias(filename);
    } 
    
    /* Apply convolutional layer */
    for (int i = 0; i < nimages; i++) {

        /* marker-based normalization */

    /* BIAS: bias do not need normalization, but fill the matrix with mean */
	iftMatrix *imgM;    
	if (use_bias==NULL){
	  iftNormalizeImageByZScore(input[i], mean, stdev);
	  imgM = iftMImageToFeatureMatrix(input[i], A, NULL);
	}else{
          imgM = iftMImageToFeatureMatrix(input[i], A, mean);
	}

	/* convolution */

	iftMatrix *conv = iftMultMatrices(imgM, kernelbank);
	iftDestroyMatrix(&imgM);

	iftMImage *activ = iftMatrixToMImage(conv, input[i]->xsize, input[i]->ysize, input[i]->zsize, kernelbank->ncols, 'c');

	
        iftDestroyMatrix(&conv);
	
        if (mask != NULL) { /* apply mask whenever it is the case */		
	  int stride = mask[i]->xsize / activ->xsize;
	  if (stride > 1) {
	    iftImage *aux = SubsampleImage(mask[i], stride);
	    iftDestroyImage(&mask[i]);
	    mask[i] = aux;
	  }
	  
#pragma omp parallel for
	  for (int p = 0; p < activ->n; p++) {
	    if (mask[i]->val[p] == 0)
	      for (int b = 0; b < activ->m; b++) {
		activ->val[p][b] = 0;
	      }
	  }
        }

        /* activation in place */
	
        if (arch->layer[layer].relu) { /* ReLU in place */
#pragma omp parallel for
	  for (int p = 0; p < activ->n; p++) {
	    for (int b = 0; b < activ->m; b++) {
	      /* BIAS: check for the environment variable */
	      if (use_bias!=NULL){
		      activ->val[p][b] += bias[b];
	      }
	      if (activ->val[p][b] < 0)
		activ->val[p][b] = 0;
	    }
	  }
        }
        
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
    /* BIAS: free bias array */
    if (use_bias!=NULL){
	    iftFree(bias);
    }
    return output;
}

iftMatrix *ConsensusKernelbank(iftFileSet *fs_seeds, char *inputdata_dir, int noutput_channels, float stdev_factor) {
    int nkernels = fs_seeds->n;
    iftMatrix **kernels = (iftMatrix **) calloc(nkernels, sizeof(iftMatrix *));
    int ncols = 0;
    int nrows = 0;
    char filename[200];

    /* Load kernels from the training images */
    for (int i = 0; i < nkernels; i++) {
        char *basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s-kernels.npy", inputdata_dir, basename);
        kernels[i] = iftReadMatrix(filename);
        ncols += kernels[i]->ncols;
        iftFree(basename);
    }
    nrows = kernels[0]->nrows;

    /* Copy all kernels into a single matrix */

    iftMatrix *Mkernels = iftCreateMatrix(ncols, nrows);
    int k = 0;
    for (int i = 0; i < nkernels; i++) {
        for (int col = 0; col < kernels[i]->ncols; col++, k++) {
            for (int row = 0; row < kernels[i]->nrows; row++) {
                iftMatrixElem(Mkernels, k, row) = iftMatrixElem(kernels[i], col, row);
            }
        }
    }

    for (int i = 0; i < nkernels; i++) {
        iftDestroyMatrix(&kernels[i]);
    }
    iftFree(kernels);

    /* Reduce the number of kernels into the desired number of output channels */

    iftMatrix *Rkernels = NULL;
    
    if (Mkernels->ncols <= noutput_channels) {
        Rkernels = iftCopyMatrix(Mkernels);
    } else {
      printf("number of kernels %d, feature space dimension %d\n",Mkernels->ncols,Mkernels->nrows);
      if (Mkernels->ncols > Mkernels->nrows){
      //if (0){
      	Rkernels = iftSelectRelevantKernelsByPCA(Mkernels, noutput_channels, stdev_factor);
      	printf("\n By PCA\n");
      }else{
	//	Rkernels = iftSelectRelevantKernelsByKmeans(Mkernels, noutput_channels, iftCosineDistance2);
	Rkernels = iftSelectRelevantKernelsByKmeans(Mkernels, noutput_channels, iftEuclideanDistance);
	printf("\n By Kmeans\n");
      }
    }

    iftDestroyMatrix(&Mkernels);

    return (Rkernels);
}

void iftStatisticsFromAllSeeds(iftFileSet *fs_seeds, char *inputdata_dir, float *mean, float *stdev, float stdev_factor) {
    int nseeds = 0, ninput_channels = 0;
    char *basename = NULL;
    char filename[200];
    iftMImage *input = NULL;

    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s.mimg", inputdata_dir, basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            nseeds += 1;
            for (int b = 0; b < ninput_channels; b++) {
                mean[b] += input->val[p][b];
            }
        }
        iftDestroyMImage(&input);
    }

    for (int b = 0; b < ninput_channels; b++) {
        mean[b] = mean[b] / nseeds;
    }

    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s.mimg", inputdata_dir, basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            for (int b = 0; b < ninput_channels; b++) {
                stdev[b] += (input->val[p][b] - mean[b]) * (input->val[p][b] - mean[b]);
            }
        }
        iftDestroyMImage(&input);
    }

    for (int b = 0; b < ninput_channels; b++) {
        stdev[b] = sqrtf(stdev[b] / nseeds) + stdev_factor;
    }

}

void iftStatisticsFromAllSeedsPerImage(iftFileSet *fs_seeds, char *inputdata_dir, float *mean, float *stdev, float stdev_factor) {
    int ninput_channels = 0;
    char *basename = NULL;
    char filename[200];
    iftMImage *input = NULL;

    int *nseeds = iftAllocIntArray(fs_seeds->n);

    int index = 0;
    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s.mimg", inputdata_dir, basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            nseeds[i] += 1;
            for (int b = 0; b < ninput_channels; b++) {
                index = ninput_channels*i + b;
                mean[index] += input->val[p][b];
            }
        }
        iftDestroyMImage(&input);
    }

    index = 0;
    for (int i = 0; i < fs_seeds->n; i++) {
        for (int b = 0; b < ninput_channels; b++) {
            index = ninput_channels*i + b;
            mean[index] = mean[index] / nseeds[i];
        }
    }

    index=0;
    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s.mimg", inputdata_dir, basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            for (int b = 0; b < ninput_channels; b++) {
                index = ninput_channels*i + b;
                stdev[index] += (input->val[p][b] - mean[index]) * (input->val[p][b] - mean[index]);
            }
        }
        iftDestroyMImage(&input);
    }

    index = 0;
    for (int i = 0; i < fs_seeds->n; i++) {
        for (int b = 0; b < ninput_channels; b++) {
            index = ninput_channels*i + b;
            stdev[index] = sqrtf(stdev[index] / nseeds[i]) + stdev_factor;
        }
    }

}

void iftReadMeanStdev(char *basepath, float *mean, float *stdev, int ninput_channels) {
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

void iftWriteMeanStdev(char *basepath, float *mean, float *stdev, int ninput_channels) {
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

float *iftReadBias(char *basepath) {
    char filename[200];
    FILE *fp;
    int number_of_kernels;
    
    sprintf(filename, "%s-bias.txt", basepath);
    fp = fopen(filename, "r");
    if (fscanf(fp, "%d", &number_of_kernels)!=1);
    float *bias = iftAllocFloatArray(number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++) {
      if (fscanf(fp, "%f", &bias[k])!=1);
    }
    fclose(fp);

    return(bias);
}

void iftWriteBias(char *basepath, float *bias, int number_of_kernels) {
    char filename[200];
    FILE *fp;

    sprintf(filename, "%s-bias.txt", basepath);
    fp = fopen(filename, "w");
    fprintf(fp, "%d\n", number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++) {
        fprintf(fp, "%f ", bias[k]);
    }
    fclose(fp);
}

iftImage *SubsampleImage(iftImage *img, int stride) {
    iftImage *simg = iftCreateImage(ceilf(img->xsize / (float) stride), ceilf(img->ysize / (float) stride),
                                    iftMax(ceilf(img->zsize / (float) stride), 1));
    int q = 0;
    iftVoxel u;
    for (u.z = 0; u.z < img->zsize; u.z = u.z + stride) {
        for (u.y = 0; u.y < img->ysize; u.y = u.y + stride) {
            for (u.x = 0; u.x < img->xsize; u.x = u.x + stride) {
                int p = iftGetVoxelIndex(img, u);
                simg->val[q] = img->val[p];
                q++;
            }
        }
    }

    return (simg);
}

/* --------------------- public functions ----------------------------- */

iftFLIMArch *iftReadFLIMArch(char *filename) {
    iftDict *arch_dict = iftReadJson(filename);
    int nlayers = iftGetLongValFromDict("nlayers", arch_dict);
    float stdev_factor = iftGetDblValFromDict("stdev_factor", arch_dict);
    bool intrinsic_atrous = FALSE;
    if (iftDictContainKey("apply_intrinsic_atrous", arch_dict, NULL)) {
        intrinsic_atrous = iftGetBoolValFromDict("apply_intrinsic_atrous", arch_dict);
    }
    iftFLIMArch *arch = (iftFLIMArch *) calloc(1, sizeof(iftFLIMArch));

    arch->nlayers = nlayers;
    arch->stdev_factor = stdev_factor;
    arch->layer = (iftFLIMLayer *) calloc(arch->nlayers, sizeof(iftFLIMLayer));
    arch->apply_intrinsic_atrous = intrinsic_atrous;

    for (int l = 0; l < arch->nlayers; l++) {

        char name[50];
        sprintf(name, "layer%d", l + 1);
        iftDict *layer_dict = iftGetDictFromDict(name, arch_dict);
        iftDict *conv_dict = iftGetDictFromDict("conv", layer_dict);
        iftIntArray *input = iftGetIntArrayFromDict("kernel_size", conv_dict);
        for (int i = 0; i < 3; i++)
            arch->layer[l].kernel_size[i] = input->val[i];
        iftFree(input);
        input = iftGetIntArrayFromDict("dilation_rate", conv_dict);
        for (int i = 0; i < 3; i++)
            arch->layer[l].dilation_rate[i] = input->val[i];
        iftFree(input);
        arch->layer[l].nkernels_per_image = iftGetLongValFromDict("nkernels_per_image", conv_dict);
        arch->layer[l].nkernels_per_marker = iftGetLongValFromDict("nkernels_per_marker", conv_dict);
        arch->layer[l].noutput_channels = iftGetLongValFromDict("noutput_channels", conv_dict);
        arch->layer[l].relu = iftGetBoolValFromDict("relu", layer_dict);
        iftDict *pool_dict = iftGetDictFromDict("pooling", layer_dict);
        arch->layer[l].pool_type = iftGetStrValFromDict("type", pool_dict);
        input = iftGetIntArrayFromDict("size", pool_dict);
        for (int i = 0; i < 3; i++)
            arch->layer[l].pool_size[i] = input->val[i];
        iftFree(input);
        arch->layer[l].pool_stride = iftGetLongValFromDict("stride", pool_dict);

        iftDestroyDict(&conv_dict);
        iftDestroyDict(&pool_dict);

        iftDestroyDict(&layer_dict);
    }

    iftDestroyDict(&arch_dict);

    return (arch);
}

void iftWriteFLIMArch(iftFLIMArch *arch, char *filename)
{
    iftDict *new_arch = iftCreateDict();

    /* Dictionary does not copy nested dicts, and therefore all nested dicts must be allocated first */
    iftDict *layer[arch->nlayers];
    iftDict *conv[arch->nlayers];
    iftDict *pooling[arch->nlayers];

    for (int l = 0; l < arch->nlayers; l++) {
        layer[l] = iftCreateDict();
        conv[l] = iftCreateDict();
        pooling[l] = iftCreateDict();
    }

    iftInsertIntoDict("stdev_factor",arch->stdev_factor, new_arch);
    iftInsertIntoDict("nlayers",arch->nlayers, new_arch);
    iftInsertIntoDict("apply_intrinsic_atrous",arch->apply_intrinsic_atrous, new_arch);

    for (int l = 0; l < arch->nlayers; l++){

        char layername[100];
        sprintf(layername,"layer%d",l+1);

        iftIntArray *kernel_size = iftCreateIntArray(3);
        for (int i = 0; i < 3; i++) kernel_size->val[i] = arch->layer[l].kernel_size[i];
        long nkernels_per_marker = arch->layer[l].nkernels_per_marker;
        iftIntArray *dilation_rate = iftCreateIntArray(3);
        for (int i = 0; i < 3; i++) dilation_rate->val[i] = arch->layer[l].dilation_rate[i];
        long nkernels_per_image = arch->layer[l].nkernels_per_image;
        long noutput_channels = arch->layer[l].noutput_channels;
        iftInsertIntoDict("kernel_size",kernel_size,conv[l]);
        iftInsertIntoDict("nkernels_per_marker",nkernels_per_marker,conv[l]);
        iftInsertIntoDict("dilation_rate",dilation_rate,conv[l]);
        iftInsertIntoDict("nkernels_per_image",nkernels_per_image,conv[l]);
        iftInsertIntoDict("noutput_channels",noutput_channels,conv[l]);
        iftInsertIntoDict("conv",conv[l],layer[l]);

        bool relu = TRUE;
        iftInsertIntoDict("relu",relu,layer[l]);

        char *pool_type = arch->layer[l].pool_type;
        iftIntArray *pool_size = iftCreateIntArray(3);
	for (int i = 0; i < 3; i++) pool_size->val[i] = arch->layer[l].pool_size[i];
        long pool_stride = arch->layer[l].pool_stride;
        iftInsertIntoDict("type",pool_type,pooling[l]);
        iftInsertIntoDict("size",pool_size,pooling[l]);
        iftInsertIntoDict("stride",pool_stride,pooling[l]);
        iftInsertIntoDict("pooling",pooling[l],layer[l]);

        iftInsertIntoDict(layername,layer[l],new_arch);
    }

    iftWriteJson(new_arch,filename);
    for (int l = 0; l < arch->nlayers; l++) {
        iftDestroyDict(&layer[l]);
        iftDestroyDict(&conv[l]);
        iftDestroyDict(&pooling[l]);
    }
    iftDestroyDict(&new_arch);
}

void iftDestroyFLIMArch(iftFLIMArch **arch) {
    iftFLIMArch *aux = *arch;

    if (aux != NULL) {
        for (int l = 0; l < aux->nlayers; l++) {
            iftFree(aux->layer[l].pool_type);
        }
        iftFree(aux->layer);
        iftFree(aux);
        *arch = NULL;
    }
}

void iftFLIMTrain(int layer, char *param_dir, char *orig_dir, char *seeds_dir)
{

   char arch_file[255], layer_output[255], layer_input[255], arch_layer_file[255];

   if (layer == 1)
   {
      sprintf(layer_input, "%s", orig_dir);
   }
   else
   {
      char file_kernels[255], file_mean[255], file_stdev[255];

      sprintf(file_kernels, "%s/conv%d-kernels.npy", param_dir, layer);
      sprintf(file_mean, "%s/conv%d-mean.txt", param_dir, layer);
      sprintf(file_stdev, "%s/conv%d-stdev.txt", param_dir, layer);

      if (!iftFileExists(file_kernels) && !iftFileExists(file_mean) && !iftFileExists(file_stdev))
         iftFLIMTrain(layer - 1, param_dir, orig_dir, seeds_dir);

      sprintf(layer_input, "%s/layer%d/", param_dir, layer - 1);
   }
   sprintf(layer_output, "%s/layer%d/", param_dir, layer);

   if (iftDirExists(layer_output))
   {
      iftRemoveDir(layer_output);
      iftMakeDir(layer_output);
   }
   else
   {
      iftMakeDir(layer_output);
   }

   // Reading architecture
   sprintf(arch_file, "%s/arch.json", param_dir);
   sprintf(arch_layer_file, "%s/arch_layer%d.json", param_dir, layer);

   // save arch on arch_layer%d.json
   iftFLIMArch *arch = iftReadFLIMArch(arch_file);

   iftFLIMArch *save_arch = (iftFLIMArch *)calloc(1, sizeof(iftFLIMArch));
   save_arch->nlayers = 1;
   save_arch->layer = (iftFLIMLayer *)calloc(1, sizeof(iftFLIMLayer));
   save_arch->layer[0].pool_type = (char *)malloc((strlen(arch->layer[layer - 1].pool_type) + 1) * sizeof(char));

   for (int i = 0; i < 3; i++)
   {
      save_arch->layer[0].kernel_size[i] = arch->layer[layer - 1].kernel_size[i];
      save_arch->layer[0].dilation_rate[i] = arch->layer[layer - 1].dilation_rate[i];
      save_arch->layer[0].pool_size[i] = arch->layer[layer - 1].pool_size[i];
   }

   save_arch->stdev_factor = arch->stdev_factor;
   save_arch->apply_intrinsic_atrous = arch->apply_intrinsic_atrous;
   save_arch->layer[0].nkernels_per_image = arch->layer[layer - 1].nkernels_per_image;
   save_arch->layer[0].nkernels_per_marker = arch->layer[layer - 1].nkernels_per_marker;
   save_arch->layer[0].noutput_channels = arch->layer[layer - 1].noutput_channels;
   save_arch->layer[0].relu = arch->layer[layer - 1].relu;
   save_arch->layer[0].topology = arch->layer[layer - 1].topology;
   strcpy(save_arch->layer[0].pool_type, arch->layer[layer - 1].pool_type);
   save_arch->layer[0].pool_stride = arch->layer[layer - 1].pool_stride;

   iftDestroyFLIMArch(&arch);
   iftWriteFLIMArch(save_arch, arch_layer_file);

   iftFLIMLearnLayer(layer_input, seeds_dir, param_dir, layer, save_arch, layer_output);

   /* Writting architecture in case it changed */
   arch = iftReadFLIMArch(arch_file);
   arch->layer[layer - 1].noutput_channels = save_arch->layer[0].noutput_channels;
   iftWriteFLIMArch(arch, arch_file);

   iftDestroyFLIMArch(&arch);
   iftDestroyFLIMArch(&save_arch);
}

void iftFLIMLearnModel(char *orig_dir, char *markers_dir, char *param_dir, iftFLIMArch *arch) {

    /* Set input parameters */

    iftMakeDir("tmp");

    iftFileSet *fs_orig  = iftLoadFileSetFromDirOrCSV(orig_dir, 1, 1);
    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds.txt", 1);
    iftAdjRel  *A        = NULL;
    iftMImage **output   = NULL;
    iftMImage *input     = NULL;
    int ninput_channels  = 0;
    int nimages          = fs_seeds->n;
    int atrous_factor    = 1;
    char *basename       = NULL;
    char filename[200], ext[10];
    
    sprintf(ext, "%s", iftFileExt(fs_orig->files[0]->path));
    printf("%s\n",ext);
    
    /* Generate input layer */
    for (int i = 0; i < nimages; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s%s", orig_dir, basename, ext);
        if (strcmp(ext,".mimg")==0){
            input = iftReadMImage(filename);
        } else {
            input = iftReadInputMImage(filename);
        }	
        sprintf(filename, "tmp/%s.mimg", basename);
        iftWriteMImage(input, filename);
        iftFree(basename);
        iftDestroyMImage(&input);
    }

    /* For each layer do */

    for (int l = 0; l < arch->nlayers; l++) {
        basename        = iftFilename(fs_seeds->files[0]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input           = iftReadMImage(filename);
        if (l == 0) {
            A = iftFLIMAdjRelFromKernel(arch->layer[l], iftIs3DMImage(input));
        } else {
            if (arch->apply_intrinsic_atrous) {
                atrous_factor *= arch->layer[l - 1].pool_stride;
                printf("Updating atrous factor\n");
                fflush(stdout);
            }
	    A = iftFLIMAdaptiveAdjRelFromKernel(arch->layer[l], atrous_factor, iftIs3DMImage(input));
        }
        ninput_channels = input->m;
        iftDestroyMImage(&input);

        /* Learn and save marker-based normalization parameters and kernels from each training image */

        float *mean  = iftAllocFloatArray(ninput_channels);
        float *stdev = iftAllocFloatArray(ninput_channels);
        iftStatisticsFromAllSeeds(fs_seeds, "tmp", mean, stdev, arch->stdev_factor);	
	printf("\nLayer %d\n", l + 1);
        fflush(stdout);

        for (int i = 0; i < nimages; i++) {

            basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
            sprintf(filename, "tmp/%s.mimg", basename);
            input    = iftReadMImage(filename);
	    
            printf("Processing file %s: %d of %d files\r", basename, i + 1, nimages);
            fflush(stdout);

            /* Apply marker-based image normalization */

            iftNormalizeImageByZScore(input, mean, stdev);
	    sprintf(filename, "tmp/%s-norm.mimg", basename);
	    iftWriteMImage(input,filename);

            /* Assign a distinct label to each marker */

            iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
            iftLabeledSet *M = LabelMarkers(input, S);
            iftDestroyLabeledSet(&S);

            /* Learn and save kernel bank */

            int nsamples          = iftLabeledSetSize(M);
            iftMatrix *kernelbank = LearnKernelBank(input, M, A, nsamples, arch->layer[l].nkernels_per_image, arch->layer[l].nkernels_per_marker);
            iftDestroyLabeledSet(&M);	   
            sprintf(filename, "tmp/%s-kernels.npy", basename);
            iftWriteMatrix(kernelbank, filename);
	    iftDestroyMatrix(&kernelbank);	    
	    iftDestroyMImage(&input);
	    iftFree(basename);
        }

        /* Create a consensus layer (i.e., merge kernels) and save
	   final kernel bank for layer l */

        iftMatrix *kernelbank = ConsensusKernelbank(fs_seeds, "tmp", arch->layer[l].noutput_channels, arch->stdev_factor);

	printf("\n nkernels %d\n", kernelbank->ncols);	
	
	/* BIAS: estimate bias array */
	char * use_bias = getenv("USE_BIAS");
	float * bias = NULL;
	if (use_bias!=NULL){
	  bias = iftAllocFloatArray(kernelbank->ncols);        
	  for (int col=0; col < kernelbank->ncols; col++){ 
	    int row = 0; 
	    for (int adj=0; adj < A->n; adj++){ 
	      for (int ch=0; ch < ninput_channels; ch++){ 
		iftMatrixElem(kernelbank,col,row) = 
		  iftMatrixElem(kernelbank,col,row) / stdev[ch]; 
		bias[col] -= (mean[ch]*iftMatrixElem(kernelbank,col,row)); 
		row++; 
	      } 
	    } 
	  }
	  
	  sprintf(filename, "%s/conv%d", param_dir, l+1);
	  iftWriteBias(filename, bias, kernelbank->ncols); 
	  iftFree(bias); 
	}

	// updating number of kernels on the architecture
        arch->layer[l].noutput_channels = kernelbank->ncols;
        sprintf(filename, "%s/conv%d-kernels.npy", param_dir, l+1);
        iftWriteMatrix(kernelbank, filename);
        iftDestroyMatrix(&kernelbank);
        sprintf(filename, "%s/conv%d", param_dir, l+1);
        iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
        // making a copy of mean and stdev for the python implementation
        sprintf(filename, "%s/m-norm%d", param_dir, l+1);
        iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
        iftFree(mean);
        iftFree(stdev);
    	iftDestroyAdjRel(&A);

        /* Apply convolutional layer using the consensus kernel bank
	   and the statistics from all markers */
        int pool_stride            = arch->layer[l].pool_stride;
        arch->layer[l].pool_stride = 1;
        for (int i = 0; i < nimages; i++) {
            basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
            sprintf(filename, "tmp/%s.mimg", basename);
            input = iftReadMImage(filename);
            iftFree(basename);
            output = FLIMConvolutionalLayer(&input, 1, NULL, arch, l, l+1, atrous_factor, param_dir);
            iftDestroyMImage(&input);
	    
            iftWriteMImage(output[0], filename);
            iftDestroyMImage(&output[0]);
            iftFree(output);
        }
        arch->layer[l].pool_stride = pool_stride;
    }
    
    iftRemoveDir("tmp");
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFileSet(&fs_orig);
}

void iftFLIMLearnModelPCA(char *orig_dir, char *markers_dir, char *param_dir, iftFLIMArch *arch) {

    /* Set input parameters */

    iftMakeDir("tmp");

    iftFileSet *fs_orig  = iftLoadFileSetFromDirOrCSV(orig_dir, 1, 1);
    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds.txt", 1);
    iftAdjRel  *A        = NULL;
    iftMImage **output   = NULL;
    iftMImage *input     = NULL;
    int ninput_channels  = 0;
    int nimages          = fs_seeds->n;
    int atrous_factor    = 1;
    char *basename       = NULL;
    char filename[200], ext[10];
    
    sprintf(ext, "%s", iftFileExt(fs_orig->files[0]->path));
    printf("%s\n",ext);
    
    /* Generate input layer */
    for (int i = 0; i < nimages; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s%s", orig_dir, basename, ext);
        if (strcmp(ext,".mimg")==0){
            input = iftReadMImage(filename);
        } else {
            input = iftReadInputMImage(filename);
        }	
        sprintf(filename, "tmp/%s.mimg", basename);
        iftWriteMImage(input, filename);
        iftFree(basename);
        iftDestroyMImage(&input);
    }

    /* For each layer do */

    for (int l = 0; l < arch->nlayers; l++) {
        basename        = iftFilename(fs_seeds->files[0]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input           = iftReadMImage(filename);
        if (l == 0) {
            A = iftFLIMAdjRelFromKernel(arch->layer[l], iftIs3DMImage(input));
        } else {
            if (arch->apply_intrinsic_atrous) {
                atrous_factor *= arch->layer[l - 1].pool_stride;
                printf("Updating atrous factor\n");
                fflush(stdout);
            }
	    A = iftFLIMAdaptiveAdjRelFromKernel(arch->layer[l], atrous_factor, iftIs3DMImage(input));
        }
        ninput_channels = input->m;
        iftDestroyMImage(&input);

        /* Compute mean and standard deviation for marker-based
	   normalization */

        float *mean  = iftAllocFloatArray(ninput_channels);
        float *stdev = iftAllocFloatArray(ninput_channels);	
        iftStatisticsFromAllSeeds(fs_seeds, "tmp", mean, stdev, arch->stdev_factor);
	/* Create and normalize the patch dataset from all seeds
	   (using all training images) */
	
	iftDataSet *Z = iftDataSetFromAllSeeds(markers_dir,"tmp",A);
	iftNormalizeDataSetByZScoreInPlace(Z, NULL, arch->stdev_factor);

	/* Compute kernelbank for the current layer */

	int number_of_kernels = arch->layer[l].noutput_channels;
	iftMatrix *kernelbank = KernelBankByPCA(Z,&number_of_kernels);
	arch->layer[l].noutput_channels = number_of_kernels;

	printf("\n Layer %d\n", l + 1);
	printf("\n Number of kernels=%d\n",arch->layer[l].noutput_channels);
        fflush(stdout);

	/* BIAS: read bias array */
    char * use_bias = getenv("USE_BIAS");
	float * bias = NULL;
	if (use_bias!=NULL){
        bias = iftAllocFloatArray(kernelbank->ncols);        
        for (int col=0; col < kernelbank->ncols; col++){ 
        int row = 0; 
        for (int adj=0; adj < A->n; adj++){ 
            for (int ch=0; ch < ninput_channels; ch++){ 
            iftMatrixElem(kernelbank,col,row) = 
                    iftMatrixElem(kernelbank,col,row) / stdev[ch]; 
            bias[col] -= (mean[ch]*iftMatrixElem(kernelbank,col,row)); 
            row++; 
            } 
        } 
        } 
	}
	
	sprintf(filename, "%s/conv%d", param_dir, l+1);
        iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
        iftFree(mean);
        iftFree(stdev);

	/* BIAS: write and free bias array */
    if (use_bias !=NULL){
		iftWriteBias(filename, bias, kernelbank->ncols); 
        iftFree(bias);
	}
	
	iftDestroyAdjRel(&A);

        sprintf(filename, "%s/conv%d-kernels.npy", param_dir, l+1);
        iftWriteMatrix(kernelbank, filename);
	iftDestroyMatrix(&kernelbank);
	iftDestroyDataSet(&Z);

        int pool_stride            = arch->layer[l].pool_stride;
        arch->layer[l].pool_stride = 1;
        for (int i = 0; i < nimages; i++) {
            basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
            sprintf(filename, "tmp/%s.mimg", basename);
            input = iftReadMImage(filename);
            iftFree(basename);
            output = FLIMConvolutionalLayer(&input, 1, NULL, arch, l, l+1, atrous_factor, param_dir);
            iftDestroyMImage(&input);
	    
            iftWriteMImage(output[0], filename);
            iftDestroyMImage(&output[0]);
            iftFree(output);
        }
        arch->layer[l].pool_stride = pool_stride;
    }
    
    //iftRemoveDir("tmp");
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFileSet(&fs_orig);
}

void iftFLIMLearnLayer(char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftFLIMArch *arch, char *output_dir)
{
    /* Set input parameters */

    iftMakeDir("tmp");

    iftFileSet *fs_activ = iftLoadFileSetFromDirOrCSV(activ_dir, 1, 1);
    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds.txt", 1);
    iftAdjRel *A         = NULL;
    iftMImage **output   = NULL;
    iftMImage *input     = NULL;
    int ninput_channels  = 0;
    int nimages          = fs_seeds->n;
    int atrous_factor    = 1;
    char *basename = NULL;
    char filename[200], ext[10];
    
    sprintf(ext, "%s", iftFileExt(fs_activ->files[0]->path));

    /* Generate input layer */
    for (int i = 0; i < nimages; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s%s", activ_dir, basename, ext);
        if (strcmp(ext,".mimg")==0){
            input = iftReadMImage(filename);
        } else {
            input = iftReadInputMImage(filename);
        }
        sprintf(filename, "tmp/%s.mimg", basename);
        iftWriteMImage(input, filename);
        iftFree(basename);
        iftDestroyMImage(&input);
    }

    /* For a specific layer do */

    basename = iftFilename(fs_seeds->files[0]->path, "-seeds.txt");
    sprintf(filename, "tmp/%s.mimg", basename);
    input           = iftReadMImage(filename);
    if (layer_index == 1) {
        A = iftFLIMAdjRelFromKernel(arch->layer[0], iftIs3DMImage(input));
    } else {
        // reading atrous factor already updated from previous layers
        sprintf(filename,"%s/intrinsic_atrous.txt",param_dir);
        if (!iftFileExists(filename)){
            FILE *fp = fopen(filename, "w");
            fprintf(fp, "%d", 1);
            fclose(fp);
        }
        char *data = iftReadFileAsString(filename);
        atrous_factor = atoi(data);
        iftFree(data);
        A = iftFLIMAdaptiveAdjRelFromKernel(arch->layer[0], atrous_factor, iftIs3DMImage(input));
    }
    ninput_channels = input->m;
    iftDestroyMImage(&input);

    /* Learn and save marker-based normalization parameters and kernels from each training image */

    float *mean  = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);
    iftStatisticsFromAllSeeds(fs_seeds, "tmp", mean, stdev, arch->stdev_factor);
    
    printf("\nLayer %d\n", layer_index);
    fflush(stdout);

    for (int i = 0; i < nimages; i++) {

        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input = iftReadMImage(filename);
	
        printf("Processing file %s: %d of %d files\r", basename, i + 1, nimages);
        fflush(stdout);

        /* Learn mean and standard deviation */

        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        iftLabeledSet *M = LabelMarkers(input, S);
        iftDestroyLabeledSet(&S);

        /* Apply marker-based image normalization */

        iftNormalizeImageByZScore(input, mean, stdev);
        sprintf(filename, "tmp/%s-norm.mimg", basename);
        iftWriteMImage(input,filename);

        /* Learn and save kernel bank */

        int nsamples          = iftLabeledSetSize(M);
        iftMatrix *kernelbank = LearnKernelBank(input, M, A, nsamples, arch->layer[0].nkernels_per_image, arch->layer[0].nkernels_per_marker);
	
	iftDestroyMImage(&input);
        iftDestroyLabeledSet(&M);
        sprintf(filename, "tmp/%s-kernels.npy", basename);
        iftWriteMatrix(kernelbank, filename);
	iftDestroyMatrix(&kernelbank);
        iftFree(basename);
    }

    /* Create a consensus layer (i.e., merge kernels) and save final kernel bank for layer l */

    iftMatrix *kernelbank = ConsensusKernelbank(fs_seeds, "tmp", arch->layer[0].noutput_channels, arch->stdev_factor);


    /* BIAS: read bias array */
    char * use_bias = getenv("USE_BIAS");
	float * bias = NULL;
	if (use_bias!=NULL){
        bias = iftAllocFloatArray(kernelbank->ncols);        
        for (int col=0; col < kernelbank->ncols; col++){ 
        int row = 0; 
        for (int adj=0; adj < A->n; adj++){ 
            for (int ch=0; ch < ninput_channels; ch++){ 
            iftMatrixElem(kernelbank,col,row) = 
                    iftMatrixElem(kernelbank,col,row) / stdev[ch]; 
            bias[col] -= (mean[ch]*iftMatrixElem(kernelbank,col,row)); 
            row++; 
            } 
        } 
        }
        sprintf(filename, "%s/conv%d", param_dir, layer_index);
        iftWriteBias(filename, bias, kernelbank->ncols); 
        iftFree(bias);
	}



    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftWriteMatrix(kernelbank, filename);
    iftDestroyMatrix(&kernelbank);
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
    // making a copy of mean and stdev for the python implementation
    sprintf(filename, "%s/m-norm%d", param_dir, layer_index);
    iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
    iftFree(mean);
    iftFree(stdev);
    iftDestroyAdjRel(&A);

    /* Apply convolutional layer using the consensus kernel bank and the statistics from all markers */
    int pool_stride = arch->layer[0].pool_stride;
    arch->layer[0].pool_stride = 1;
    for (int i = 0; i < nimages; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input = iftReadMImage(filename);

        output = FLIMConvolutionalLayer(&input, 1, NULL, arch, 0, layer_index, atrous_factor, param_dir);
        iftDestroyMImage(&input);

        sprintf(filename, "%s/%s.mimg", output_dir, basename);
        iftWriteMImage(output[0], filename);
        iftDestroyMImage(&output[0]);
        iftFree(output);
        iftFree(basename);
    }
    arch->layer[0].pool_stride = pool_stride;

    // updating atrous factor with pooling of current layer
    if (arch->apply_intrinsic_atrous) {
        atrous_factor *= arch->layer[0].pool_stride;
        printf("Updating atrous factor for next layer\n");
        fflush(stdout);
    }
    // writing atrous factor on file
    sprintf(filename,"%s/intrinsic_atrous.txt",param_dir);
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d", atrous_factor);
    fclose(fp);
    
    iftRemoveDir("tmp");
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFileSet(&fs_activ);
}


iftMatrix * ConsensusBias(iftFileSet *fs_seeds, char *inputdata_dir, int ncols, float * final_bias){

    int nkernels = fs_seeds->n;
    iftMatrix **kernels = (iftMatrix **) calloc(nkernels, sizeof(iftMatrix *));

    char filename[200];
    float **all_bias = (float **) calloc(nkernels, sizeof(float *));
    /* Load kernels from the training images */
    for (int i = 0; i < nkernels; i++) {
        char *basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s-kernels.npy", inputdata_dir, basename);
        kernels[i] = iftReadMatrix(filename);

        sprintf(filename, "%s/%s-bias.txt", inputdata_dir, basename);
        all_bias[i] = iftReadBias(filename);
        iftFree(basename);
    }
    int nrows = kernels[0]->nrows;

    /* Copy all kernels into a single matrix */
    iftMatrix *Mkernels = iftCreateMatrix(ncols, nrows);
    int k = 0;
    for (int i = 0; i < nkernels; i++) {
        for (int col = 0; col < kernels[i]->ncols; col++, k++) {
            final_bias[k] = all_bias[i][col];
            // *(final_bias[k]) = 0;
            // new_final_bias[k] = all_bias[i][col];
            for (int row = 0; row < kernels[i]->nrows; row++) {
                iftMatrixElem(Mkernels, k, row) = iftMatrixElem(kernels[i], col, row);
            }
        }
    }

    for (int i = 0; i < nkernels; i++) {
        iftDestroyMatrix(&kernels[i]);
        iftFree(all_bias[i]);
    }
    iftFree(kernels);
    iftFree(all_bias);


    return Mkernels;
}


void iftFLIMLearnBiasedLayer(char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftFLIMArch *arch, char *output_dir)
{
    /* Set input parameters */

    iftMakeDir("tmp");

    iftFileSet *fs_activ = iftLoadFileSetFromDirOrCSV(activ_dir, 1, 1);
    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds.txt", 1);
    iftAdjRel *A         = NULL;
    iftMImage **output   = NULL;
    iftMImage *input     = NULL;
    int ninput_channels  = 0;
    int nimages          = fs_seeds->n;
    int atrous_factor    = 1;
    char *basename = NULL;
    char filename[200], ext[10];


    /* Compute the number of filters per image according to the desired filters 'C'*/
    int desired_filters = arch->layer[0].noutput_channels;
    int number_of_images = fs_seeds->n;
    int *n_of_filters_per_image = iftAllocIntArray(number_of_images);
    int nf = desired_filters/number_of_images;
    int rest = desired_filters%number_of_images;
    for(int i=0;i<number_of_images;i++){
        n_of_filters_per_image[i]=nf;
        if(rest>0){
            n_of_filters_per_image[i]++;
            rest--;
        }
    }

    
    sprintf(ext, "%s", iftFileExt(fs_activ->files[0]->path));

    /* Generate input layer */
    for (int i = 0; i < nimages; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s%s", activ_dir, basename, ext);
        if (strcmp(ext,".mimg")==0){
            input = iftReadMImage(filename);
        } else {
            input = iftReadInputMImage(filename);
        }
        sprintf(filename, "tmp/%s.mimg", basename);
        iftWriteMImage(input, filename);
        iftFree(basename);
        iftDestroyMImage(&input);
    }

    /* For a specific layer compute the atros convolution factor and adj region */

    basename = iftFilename(fs_seeds->files[0]->path, "-seeds.txt");
    sprintf(filename, "tmp/%s.mimg", basename);
    input           = iftReadMImage(filename);
    if (layer_index == 1) {
        A = iftFLIMAdjRelFromKernel(arch->layer[0], iftIs3DMImage(input));
    } else {
        // reading atrous factor already updated from previous layers
        sprintf(filename,"%s/intrinsic_atrous.txt",param_dir);
        if (!iftFileExists(filename)){
            FILE *fp = fopen(filename, "w");
            fprintf(fp, "%d", 1);
            fclose(fp);
        }
        char *data = iftReadFileAsString(filename);
        atrous_factor = atoi(data);
        iftFree(data);
        A = iftFLIMAdaptiveAdjRelFromKernel(arch->layer[0], atrous_factor, iftIs3DMImage(input));
    }
    ninput_channels = input->m;
    iftDestroyMImage(&input);

    /* Learn and save mean and stdev parameters and kernels from each training image
        this differs from the standard FLIM that computes the mean and stdev for each image
     */

    float *mean  = iftAllocFloatArray(ninput_channels*nimages);
    float *stdev = iftAllocFloatArray(ninput_channels*nimages);
    iftStatisticsFromAllSeedsPerImage(fs_seeds, "tmp", mean, stdev, arch->stdev_factor);
    
    printf("\nLayer %d\n", layer_index);
    fflush(stdout);

    int ncols = 0;

    for (int i = 0; i < nimages; i++) {

        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input = iftReadMImage(filename);
	
        printf("Processing file %s: %d of %d files\n", basename, i + 1, nimages);
        fflush(stdout);

        /* Learn mean and standard deviation */

        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        iftLabeledSet *M = LabelMarkers(input, S);
        iftDestroyLabeledSet(&S);

        /* Apply marker-based image normalization */

        iftNormalizeImageByZScore(input, &(mean[i*ninput_channels]), &(stdev[i*ninput_channels]));
        sprintf(filename, "tmp/%s-norm.mimg", basename);
        iftWriteMImage(input,filename);

        /* Learn and save kernel bank */

        int nsamples          = iftLabeledSetSize(M);
	
        // iftMatrix *kernelbank = LearnKernelBank(input, M, A, nsamples,  arch->layer[0].nkernels_per_image, arch->layer[0].nkernels_per_marker);
        iftMatrix *kernelbank = LearnKernelBank(input, M, A, nsamples,  n_of_filters_per_image[i], arch->layer[0].nkernels_per_marker);

	printf("\n nkernels %d \n",kernelbank->ncols);
	
         /* BIAS: read bias array */
        float *bias = iftAllocFloatArray(kernelbank->ncols);
        ncols+= kernelbank->ncols;    
        for (int col=0; col < kernelbank->ncols; col++){ 
            int row = 0; 
            for (int adj=0; adj < A->n; adj++){ 
                for (int ch=0; ch < ninput_channels; ch++){ 
                    iftMatrixElem(kernelbank,col,row) = 
                            iftMatrixElem(kernelbank,col,row) / stdev[i*ninput_channels+ch]; 
                    bias[col] -= (mean[i*ninput_channels+ch]*iftMatrixElem(kernelbank,col,row)); 
                    row++; 
                } 
            } 
        }
        sprintf(filename, "tmp/%s-bias.txt", basename);
        iftWriteBias(filename, bias, kernelbank->ncols); 

        sprintf(filename, "tmp/%s-kernels.npy", basename);
        iftWriteMatrix(kernelbank, filename);
        
	    iftDestroyMImage(&input);
        iftDestroyLabeledSet(&M);
	    iftDestroyMatrix(&kernelbank);
        iftFree(basename);
        iftFree(bias);
    }

    /* combining to final kernelbank*/
    float * final_bias = iftAllocFloatArray(ncols);
    iftMatrix * final_kernelbank = ConsensusBias(fs_seeds, "tmp", ncols, final_bias);


    iftFree(n_of_filters_per_image);

    /* write files*/
    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftWriteMatrix(final_kernelbank, filename);
    iftDestroyMatrix(&final_kernelbank);
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftWriteBias(filename, final_bias, ncols); 

    iftFree(mean);
    iftFree(stdev);

    mean = iftAllocFloatArray(ninput_channels);
    stdev = iftAllocFloatArray(ninput_channels);

    for(int b=0;b<ninput_channels;b++){
        mean[b]=0;
        stdev[b]=1.0;
    }
    
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
    // making a copy of mean and stdev for the python implementation
    sprintf(filename, "%s/m-norm%d", param_dir, layer_index);
    iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
    iftDestroyAdjRel(&A);
    iftFree(mean);
    iftFree(stdev);

    /* Apply convolutional layer using the consensus kernel bank and the statistics from all markers */
    arch->layer[0].pool_stride = 1;
    for (int i = 0; i < nimages; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input = iftReadMImage(filename);

        output = FLIMConvolutionalLayer(&input, 1, NULL, arch, 0, layer_index, atrous_factor, param_dir);
        iftDestroyMImage(&input);

        sprintf(filename, "%s/%s.mimg", output_dir, basename);
        iftWriteMImage(output[0], filename);
        iftDestroyMImage(&output[0]);
        iftFree(output);
        iftFree(basename);
    }
    // arch->layer[0].pool_stride = pool_stride;

    // updating atrous factor with pooling of current layer
    if (arch->apply_intrinsic_atrous) {
        atrous_factor *= arch->layer[0].pool_stride;
        printf("Updating atrous factor for next layer\n");
        fflush(stdout);
    }
    // writing atrous factor on file
    sprintf(filename,"%s/intrinsic_atrous.txt",param_dir);
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d", atrous_factor);
    fclose(fp);
    
    iftRemoveDir("tmp");
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFileSet(&fs_activ);
}

void iftFLIMExtractFeatures(char *orig_dir, char *image_list, iftFLIMArch *arch, char *param_dir,
                            char *feat_dir, char *object_dir, int device) {
    iftFileSet *fs_images = iftLoadFileSetFromDirOrCSV(image_list, 1, 1);
    int nimages = fs_images->n;
    int batchsize = 0;
    int nbatches = 0;
    int nremaining_images = nimages;
    iftMImage *input = NULL;
    char filename[200], ext[10];
    bool batch_process = true;
    int ninput_channels = 0;
    int input_image_size = 0;
    sprintf(ext, "%s", iftFileExt(fs_images->files[0]->path));

    /* Verify if all images have the same dimension for batch processing */

    sprintf(filename, "%s/%s", orig_dir, fs_images->files[0]->path);
    if (strcmp(ext,".mimg")==0){
      input = iftReadMImage(filename);
    } else {
      input = iftReadInputMImage(filename);
    }	
    int xsize = input->xsize, ysize = input->ysize, zsize = input->zsize;
    ninput_channels = input->m;
    input_image_size = input->n;
    iftDestroyMImage(&input);
    for (int i = 1; i < nimages; i++) {
        sprintf(filename, "%s/%s", orig_dir, fs_images->files[i]->path);
        if (strcmp(ext,".mimg")==0){
            input = iftReadMImage(filename);
        } else {
            input = iftReadInputMImage(filename);
        }	
        if ((input->xsize != xsize) ||
            (input->ysize != ysize) ||
            (input->zsize != zsize)) {
            batch_process = false;
            iftDestroyMImage(&input);
            break;
        }
    }

    /* Select batch size for GPU/CPU processing */

  if (batch_process) {  /* process in batch */
#ifdef IFT_GPU
    int ndevices = iftNumberOfDevices();
    if (ndevices > 0 && iftStartDevice(device)){
      batchsize         = iftFLIMBatchSizeGPU(arch,input_image_size,ninput_channels,device);
      batchsize         = iftMin(batchsize,nimages);
      nbatches          = nimages/batchsize;
      nremaining_images = nremaining_images - nbatches*batchsize;
      for (int batch=0; batch < nbatches; batch++){
	int first = batch*batchsize, last  = first + batchsize - 1;
	printf("Processing batch %d of %d batches\n", batch+1, nbatches);
	fflush(stdout);
	FLIMExtractFeaturesPerBatch(fs_images, first, last, orig_dir, arch, param_dir, feat_dir, object_dir);
      }
      int first = nimages - nremaining_images, last  = nimages - 1;
      if (first <= last){
	printf("Processing remaining %d images\n", last-first+1);
	fflush(stdout);
	FLIMExtractFeaturesPerBatch(fs_images, first, last, orig_dir, arch, param_dir, feat_dir, object_dir);
      }
      iftStopDevice(device);
    } else {
      int first = 0, last  = nimages - 1;
      FLIMExtractFeaturesPerImage(fs_images, first, last, orig_dir, arch, param_dir, feat_dir, object_dir);
    }
#else
    batchsize = iftFLIMBatchSizeCPU(arch, input_image_size, ninput_channels);
    batchsize = iftMin(batchsize, nimages);
    nbatches = nimages / batchsize;
    nremaining_images = nremaining_images - nbatches * batchsize;
    for (int batch = 0; batch < nbatches; batch++) {
      int first = batch * batchsize, last = first + batchsize - 1;
      printf("Processing batch %d of %d batches\n", batch + 1, nbatches);
      fflush(stdout);
      FLIMExtractFeaturesPerBatch(fs_images, first, last, orig_dir, arch, param_dir, feat_dir, object_dir);
    }
    int first = nimages - nremaining_images, last = nimages - 1;
    if (first <= last) {
      printf("Processing remaining %d images\n", last - first + 1);
      fflush(stdout);
      FLIMExtractFeaturesPerBatch(fs_images, first, last, orig_dir, arch, param_dir, feat_dir, object_dir);
    }
#endif
  } else { /* process per image */
    int first = 0, last = nimages - 1;
    FLIMExtractFeaturesPerImage(fs_images, first, last, orig_dir, arch, param_dir, feat_dir, object_dir);
  }
  
  iftDestroyFileSet(&fs_images);
}

void FLIMExtractFeaturesFromLayer(int layer, char *param_dir, char *orig_dir, char *seeds_dir, bool write_img)
{

  char arch_file[255], layer_input[255], graph_list[255], activation_dir[255];
  //char kernels_img_dir[255];
  char file_kernels[255], file_mean[255], file_stdev[255];

   sprintf(file_kernels, "%s/conv%d-kernels.npy", param_dir, layer);
   sprintf(file_mean, "%s/conv%d-mean.txt", param_dir, layer);
   sprintf(file_stdev, "%s/conv%d-stdev.txt", param_dir, layer);

   if (!iftFileExists(file_kernels) && !iftFileExists(file_mean) && !iftFileExists(file_stdev))
      iftFLIMTrain(layer, param_dir, orig_dir, seeds_dir);

   sprintf(activation_dir, "%s/layer%d/", param_dir, layer);

   if (layer == 1)
   {
      sprintf(layer_input, "%s", orig_dir);
      sprintf(graph_list, "%s/input.csv", param_dir);
   }
   else
   {
      sprintf(layer_input, "%s/layer%d/", param_dir, layer - 1);
      sprintf(graph_list, "%s/layer%d.csv", param_dir, layer - 1);
   }

   iftFileSet *list = iftLoadFileSetFromDir(layer_input, 1);
   iftFileSet *list_orig = iftLoadFileSetFromDir(orig_dir, 1);

   if (list->n < list_orig->n)
   {
      FLIMExtractFeaturesFromLayer(layer - 1, param_dir, orig_dir, seeds_dir, write_img);
   }

   /* // Creating list of graphs */
   /*    writeCSVFiles(graph_list, layer_input); */

   // Creating output dir for layer
   if (iftDirExists(activation_dir))
   {
      iftRemoveDir(activation_dir);
      iftMakeDir(activation_dir);
   }
   else
   {
      iftMakeDir(activation_dir);
   }

   // Reading architecture
   sprintf(arch_file, "%s/arch_layer%d.json", param_dir, layer);
   iftFLIMArch *tmp_arch = iftReadFLIMArch(arch_file);

   iftFLIMExtractFeaturesFromLayer(layer_input, graph_list, tmp_arch, param_dir, layer,
                                   activation_dir, NULL, -1);

   iftDestroyFLIMArch(&tmp_arch);

   // visualization

   /* Verify if layer<n> exists, which is necessary to visualize activations */
   char layer_output[255];
   sprintf(layer_output, "%s/layer%d/", param_dir, layer);
   if (!iftDirExists(layer_output))
   {
      iftMakeDir(layer_output);
   }

   /* if (write_img) */
   /* { */
   /*    char ext[10]; */
   /*    sprintf(ext, "pgm"); */
   /*    sprintf(kernels_img_dir, "%s/img_activations/layer%d/", param_dir, layer); */
   /*    iftRemoveDir(kernels_img_dir); */
   /*    iftMakeDir(kernels_img_dir); */
   /*    writeActivationsImg(activation_dir, kernels_img_dir, ext, NULL); */
   /* } */
}

void iftFLIMExtractFeaturesFromLayer(char *orig_dir, char *image_list, iftFLIMArch *arch, char *param_dir, int layer_index,
                                     char *feat_dir, char *object_dir, int device) {
    iftFileSet *fs_images = iftLoadFileSetFromDirOrCSV(image_list, 1, 1);
    int nimages = fs_images->n;
    int batchsize = 0;
    int nbatches = 0;
    int nremaining_images = nimages;
    iftMImage *input = NULL;
    char filename[200], ext[10];
    bool batch_process = true;
    int ninput_channels = 0;
    int input_image_size = 0;
    
    sprintf(ext, "%s", iftFileExt(fs_images->files[0]->path));

    /* Verify if all images have the same dimension for batch processing */

    sprintf(filename, "%s/%s", orig_dir, fs_images->files[0]->path);
    if (strcmp(ext,".mimg")==0){
        input = iftReadMImage(filename);
    }else{
        input = iftReadInputMImage(filename);
    }
    int xsize = input->xsize, ysize = input->ysize, zsize = input->zsize;
    ninput_channels = input->m;
    input_image_size = input->n;
    iftDestroyMImage(&input);
    for (int i = 1; i < nimages; i++) {
        sprintf(filename, "%s/%s", orig_dir, fs_images->files[i]->path);
        if (strcmp(ext,".mimg")==0){
            input = iftReadMImage(filename);
        } else {
            input = iftReadInputMImage(filename);
        }
        if ((input->xsize != xsize) ||
            (input->ysize != ysize) ||
            (input->zsize != zsize)) {
            batch_process = false;
            iftDestroyMImage(&input);
            break;
        }
    }

    /* Select batch size for GPU/CPU processing */

    if (batch_process) {  /* process in batch */
#ifdef IFT_GPU
  int ndevices = iftNumberOfDevices();
  if (ndevices > 0 && iftStartDevice(device)){
    batchsize         = iftFLIMBatchSizeGPU(arch,input_image_size,ninput_channels,device);
    batchsize         = iftMin(batchsize,nimages);
    nbatches          = nimages/batchsize;
    nremaining_images = nremaining_images - nbatches*batchsize;
    for (int batch=0; batch < nbatches; batch++){
      int first = batch*batchsize, last  = first + batchsize - 1;
      printf("Processing batch %d of %d batches\n", batch+1, nbatches);
      fflush(stdout);
      FLIMExtractFeaturesFromLayerPerBatch(fs_images, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
    }
    int first = nimages - nremaining_images, last  = nimages - 1;
    if (first <= last){
      printf("Processing remaining %d images\n", last-first+1);
      fflush(stdout);
      FLIMExtractFeaturesFromLayerPerBatch(fs_images, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
    }
    iftStopDevice(device);
      } else {
    int first = 0, last  = nimages - 1;
    FLIMExtractFeaturesFromLayerPerImage(fs_images, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
      }
#else
        batchsize = iftFLIMBatchSizeCPU(arch, input_image_size, ninput_channels);
        batchsize = iftMin(batchsize, nimages);
        nbatches = nimages / batchsize;
        nremaining_images = nremaining_images - nbatches * batchsize;
        for (int batch = 0; batch < nbatches; batch++) {
            int first = batch * batchsize, last = first + batchsize - 1;
            printf("Processing batch %d of %d batches\n", batch + 1, nbatches);
            fflush(stdout);
            FLIMExtractFeaturesFromLayerPerBatch(fs_images, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
        }
        int first = nimages - nremaining_images, last = nimages - 1;
        if (first <= last) {
            printf("Processing remaining %d images\n", last - first + 1);
            fflush(stdout);
            FLIMExtractFeaturesFromLayerPerBatch(fs_images, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
        }
#endif
    } else { /* process per image */
        int first = 0, last = nimages - 1;
        FLIMExtractFeaturesFromLayerPerImage(fs_images, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
    }

    iftDestroyFileSet(&fs_images);
}

int iftFLIMBatchSizeCPU(iftFLIMArch *arch, int input_image_nvoxels, int input_image_nchannels) {
    float freeMemoryMb, percMemory = 0.85;

    freeMemoryMb = (float) iftGetFreePhysicalSystemMemory() / 1024.0 / 1024.0;
    printf("CPU: %.0fGB of free memory.\n", freeMemoryMb / 1024.0);

    float nMbytesPerDouble = sizeof(double) / 1024.0 / 1024.0;
    float ninput_channels = input_image_nchannels;
    float nMbytesInputImage = input_image_nvoxels * ninput_channels * nMbytesPerDouble;
    float max_requiredMemory = 0.0;
    for (int l = 0; l < arch->nlayers; l++) {
        int KS2 = 1;
        if (arch->layer[l].kernel_size[2] != 0)
            KS2 = arch->layer[l].kernel_size[2];
        float kernelsize = arch->layer[l].kernel_size[0] * arch->layer[l].kernel_size[1] * KS2;
        float nMbytesKernels = arch->layer[l].noutput_channels * kernelsize * ninput_channels * nMbytesPerDouble;
        float nMbytesOutputImage = input_image_nvoxels * arch->layer[l].noutput_channels * nMbytesPerDouble;
        float requiredMemory = nMbytesKernels + nMbytesOutputImage + nMbytesInputImage;

        if (requiredMemory > max_requiredMemory) {
            max_requiredMemory = requiredMemory;
        }
        ninput_channels = arch->layer[l].noutput_channels;
        nMbytesInputImage = nMbytesOutputImage / arch->layer[l].pool_stride;
        input_image_nvoxels = input_image_nvoxels / arch->layer[l].pool_stride;
    }

    int batchsize = (int) floor(percMemory * freeMemoryMb / max_requiredMemory);

    return (batchsize);
}


iftMImage *iftFLIMAtrousAveragePooling(iftMImage *mimg, int width, int height, int depth, int atrous_factor, int stride) {

    iftAdjRel *A;

    if (iftIs3DMImage(mimg)) {
        A = iftCuboidWithDilation(width, height, depth, atrous_factor, atrous_factor, atrous_factor);
    } else {
        A = iftRectangularWithDilation(width, height, atrous_factor, atrous_factor);
    }

    iftMImage *pool = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);

#pragma omp parallel for
    for (int p = 0; p < mimg->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mimg, p);
        for (int i = 0; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);
                for (int b = 0; b < mimg->m; b++)
                    pool->val[p][b] += mimg->val[q][b];
            }
        }
        for (int b = 0; b < mimg->m; b++)
            pool->val[p][b] /= A->n;
    }

    iftDestroyAdjRel(&A);
    if (stride > 1) {
        iftMImage *mimgRet = iftCreateMImage(ceilf(mimg->xsize / (float) stride), ceilf(mimg->ysize / (float) stride),
                                             iftMax(ceilf(mimg->zsize / (float) stride), 1), mimg->m);
        int q = 0;
        iftVoxel u;
        for (u.z = 0; u.z < pool->zsize; u.z = u.z + stride) {
            for (u.y = 0; u.y < pool->ysize; u.y = u.y + stride) {
                for (u.x = 0; u.x < pool->xsize; u.x = u.x + stride) {
                    int p = iftMGetVoxelIndex(pool, u);
                    for (int b = 0; b < pool->m; b++)
                        mimgRet->val[q][b] = pool->val[p][b];
                    q++;
                }
            }
        }

        iftDestroyMImage(&pool);
        return mimgRet;
    }
    return (pool);
}


/*iftMImage *iftFLIMAveragePooling(iftMImage *mimg, int width, int height, int depth, int stride) {
    iftAdjRel *A;

    if (iftIs3DMImage(mimg)) {
        if ((width == height) && (height == depth))
            A = iftSpheric(width / 2.0);
        else
            A = iftCuboid(width, height, depth);
    } else {
        if (width == height)
            A = iftCircular(width / 2.0);
        else
            A = iftRectangular(width, height);
    }

    iftMImage *pool = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);

#pragma omp parallel for
    for (int p = 0; p < mimg->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mimg, p);
        for (int i = 0; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);
                for (int b = 0; b < mimg->m; b++)
                    pool->val[p][b] += mimg->val[q][b];
            }
        }
        for (int b = 0; b < mimg->m; b++)
            pool->val[p][b] /= A->n;
    }

    iftDestroyAdjRel(&A);
    if (stride > 1) {
        iftMImage *mimgRet = iftCreateMImage(ceilf(mimg->xsize / (float) stride), ceilf(mimg->ysize / (float) stride),
                                             iftMax(ceilf(mimg->zsize / (float) stride), 1), mimg->m);
        int q = 0;
        iftVoxel u;
        for (u.z = 0; u.z < pool->zsize; u.z = u.z + stride) {
            for (u.y = 0; u.y < pool->ysize; u.y = u.y + stride) {
                for (u.x = 0; u.x < pool->xsize; u.x = u.x + stride) {
                    int p = iftMGetVoxelIndex(pool, u);
                    for (int b = 0; b < pool->m; b++)
                        mimgRet->val[q][b] = pool->val[p][b];
                    q++;
                }
            }
        }

        iftDestroyMImage(&pool);
        return mimgRet;
    }
    return (pool);
}*/

iftMImage *iftFLIMAtrousMaxPooling(iftMImage *mimg, int width, int height, int depth, int atrous_factor, int stride) {
    iftAdjRel *A;

    if (iftIs3DMImage(mimg)) {
        A = iftCuboidWithDilation(width, height, depth, atrous_factor, atrous_factor, atrous_factor);
    } else {
        A = iftRectangularWithDilation(width, height, atrous_factor, atrous_factor);
    }

    iftMImage *pool = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);
    iftSetMImage(pool, IFT_INFINITY_FLT_NEG);

#pragma omp parallel for
    for (int p = 0; p < mimg->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mimg, p);
        for (int i = 0; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);
                for (int b = 0; b < mimg->m; b++)
                    if (mimg->val[q][b] > pool->val[p][b])
                        pool->val[p][b] = mimg->val[q][b];
            }
        }
    }

    iftDestroyAdjRel(&A);

    if (stride > 1) {
        iftMImage *mimgRet = iftCreateMImage(ceilf(mimg->xsize / (float) stride), ceilf(mimg->ysize / (float) stride),
                                             iftMax(ceilf(mimg->zsize / (float) stride), 1), mimg->m);
        int q = 0;
        iftVoxel u;
        for (u.z = 0; u.z < pool->zsize; u.z = u.z + stride) {
            for (u.y = 0; u.y < pool->ysize; u.y = u.y + stride) {
                for (u.x = 0; u.x < pool->xsize; u.x = u.x + stride) {
                    int p = iftMGetVoxelIndex(pool, u);
                    for (int b = 0; b < pool->m; b++)
                        mimgRet->val[q][b] = pool->val[p][b];
                    q++;
                }
            }
        }

        iftDestroyMImage(&pool);

        return mimgRet;
    }

    return pool;
}

/*
iftMImage *iftFLIMMaxPooling(iftMImage *mimg, int width, int height, int depth, int stride) {
    iftAdjRel *A;

    if (iftIs3DMImage(mimg)) {
        if ((width == height) && (height == depth))
            A = iftSpheric(width / 2.0);
        else
            A = iftCuboid(width, height, depth);
    } else {
        if (width == height)
            A = iftCircular(width / 2.0);
        else
            A = iftRectangular(width, height);
    }

    iftMImage *pool = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);
    iftSetMImage(pool, IFT_INFINITY_FLT_NEG);

#pragma omp parallel for
    for (int p = 0; p < mimg->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mimg, p);
        for (int i = 0; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);
                for (int b = 0; b < mimg->m; b++)
                    if (mimg->val[q][b] > pool->val[p][b])
                        pool->val[p][b] = mimg->val[q][b];
            }
        }
    }

    iftDestroyAdjRel(&A);

    if (stride > 1) {
        iftMImage *mimgRet = iftCreateMImage(ceilf(mimg->xsize / (float) stride), ceilf(mimg->ysize / (float) stride),
                                             iftMax(ceilf(mimg->zsize / (float) stride), 1), mimg->m);
        int q = 0;
        iftVoxel u;
        for (u.z = 0; u.z < pool->zsize; u.z = u.z + stride) {
            for (u.y = 0; u.y < pool->ysize; u.y = u.y + stride) {
                for (u.x = 0; u.x < pool->xsize; u.x = u.x + stride) {
                    int p = iftMGetVoxelIndex(pool, u);
                    for (int b = 0; b < pool->m; b++)
                        mimgRet->val[q][b] = pool->val[p][b];
                    q++;
                }
            }
        }

        iftDestroyMImage(&pool);

        return mimgRet;
    }

    return pool;
}*/

iftMatrix *iftFLIMSelectKernelsManual(char *kernel_bank_path, char *selected_kernels_path)
{
    iftMatrix *input_kernels   = iftReadMatrix(kernel_bank_path);
    iftDict    *json           = iftReadJson(selected_kernels_path);
    iftIntArray *selKernels    = iftGetIntArrayFromDict("selected_kernels", json);
    int        nKernels        = selKernels->n;
    iftMatrix *output_kernels  = iftCreateMatrix(nKernels,input_kernels->nrows);
    char *use_bias             = getenv("USE_BIAS");

    if (use_bias!=NULL){
      float *bias      = NULL;
      char *model_path = iftDirname(kernel_bank_path);
      char *prefix     = iftFilename(kernel_bank_path, "-kernels.npy");
      char basename[300];
      sprintf(basename,"%s/%s",model_path,prefix);
      printf("%s \n",basename);
      bias             = iftReadBias(basename);
      float *newbias   = iftAllocFloatArray(nKernels);
      /* perform kernel and bias selection */
      for(int c = 0; c < nKernels; c++) {
        int k      = selKernels->val[c];
	printf("k %d\n",k);
	newbias[c] = bias[abs(k)];
        for (int r=0; r < input_kernels->nrows; r++){
	  iftMatrixElem(output_kernels, c, r) = iftSign(k)*iftMatrixElem(input_kernels, abs(k), r);
        }
      }
      iftWriteBias(basename, newbias, nKernels); 
      iftFree(bias);
      iftFree(newbias);
      iftFree(model_path);
      iftFree(prefix);
    } else {
      /* perform kernel selection */
      for(int c = 0; c < nKernels; c++) {
        int k = selKernels->val[c];
        for (int r=0; r < input_kernels->nrows; r++){
	  iftMatrixElem(output_kernels, c, r) = iftSign(k)*iftMatrixElem(input_kernels, abs(k), r);
        }
      }
    }
    
    iftDestroyMatrix(&input_kernels);
    return output_kernels;
}

void iftFLIMConvertModel2BIAS(iftFLIMArch *arch, int ninput_channels, char * param_dir, char * output_dir){
    
    iftMakeDir(output_dir);
    
    for(int l=0; l<arch->nlayers; l++){
        int layer_index = l+1;

        //read kernel, mean and std
        char filename[200];
        sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
        iftMatrix *kernelbank = iftReadMatrix(filename);
        float *mean  = iftAllocFloatArray(ninput_channels);
        float *stdev = iftAllocFloatArray(ninput_channels);
        sprintf(filename, "%s/conv%d", param_dir, layer_index);
        iftReadMeanStdev(filename, mean, stdev, ninput_channels);

        //convert parameters
        float *bias = iftAllocFloatArray(kernelbank->ncols);        
        for (int col=0; col < kernelbank->ncols; col++){ 
            int row = 0; 
            int adj_size = kernelbank->nrows/ninput_channels;
            for (int adj=0; adj < adj_size; adj++){ 
                for (int ch=0; ch < ninput_channels; ch++){ 
                iftMatrixElem(kernelbank,col,row) = 
                        iftMatrixElem(kernelbank,col,row) / stdev[ch]; 
                bias[col] -= (mean[ch]*iftMatrixElem(kernelbank,col,row)); 
                row++; 
                } 
            } 
        }

        //save parameters to dir
        sprintf(filename, "%s/conv%d", output_dir, l+1);
        iftWriteBias(filename, bias, kernelbank->ncols);

        sprintf(filename, "%s/conv%d-kernels.npy", output_dir, l+1);
        iftWriteMatrix(kernelbank, filename);

        ninput_channels = kernelbank->ncols;

        iftFree(bias);
        iftFree(mean);
        iftFree(stdev);
        iftDestroyMatrix(&kernelbank);
        
    }

}
