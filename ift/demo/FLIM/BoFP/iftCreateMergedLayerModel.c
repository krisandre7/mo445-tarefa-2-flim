#include "ift.h"

/* 
   Author: Alexandre Xavier Falcão (August 23rd 2024)

   Description: Creates one model from all training images, by
   estimating one kernel per feature point using the patch shape
   provided in arch.json for the current layer. For each kernel, it
   uses the marker normalization parameters to estimate one bias per
   kernel. The model (kernels and biases) is saved in a given model
   folder. It also saves a weight file with one weight per kernel for
   possible decoding of the layer's output. It assigns weight 1 for
   object kernels (i.e., kernels from object feature points) and -1
   for background ones.

*/

iftVoxel GetVoxelCoord(int xsize, int ysize, int zsize, int p)
{
  iftVoxel u;
  div_t res1 = div(p, xsize * ysize);
  div_t res2 = div(res1.rem, xsize);

    u.x = res2.rem;
    u.y = res2.quot;
    u.z = res1.quot;
    u.t = 0;
    return(u);
}

iftAdjRel *GetPatchAdjacency(iftMImage *mimg, iftFLIMLayer layer)
{
  iftAdjRel *A;

  if (iftIs3DMImage(mimg)){
    A = iftCuboidWithDilationForConv(layer.kernel_size[0],
				     layer.kernel_size[1],
				     layer.kernel_size[2],
				     layer.dilation_rate[0],
				     layer.dilation_rate[1],
				     layer.dilation_rate[2]);
  }else{
    A = iftRectangularWithDilationForConv(layer.kernel_size[0],
					  layer.kernel_size[1],
					  layer.dilation_rate[0],
					  layer.dilation_rate[1]);    
  }

  return(A);
}

void AddImagePatches(iftDataSet *Z, int *s, float *scale, int dim, iftMImage *mimg, iftLabeledSet *S, iftAdjRel *A)
{
  iftLabeledSet *Saux = S;
  while (Saux != NULL) {
    int p                  = Saux->elem;
    Z->sample[*s].id        = Saux->marker;
    Z->sample[*s].truelabel = Saux->label + 1; /* truelabels vary from 1 to c */
    if (Z->sample[*s].truelabel > Z->nclasses)
      Z->nclasses = Z->sample[*s].truelabel;
    iftVoxel u;
    if (dim==3){
      u = GetVoxelCoord((int)(mimg->xsize*scale[0]),
			(int)(mimg->ysize*scale[1]),
			(int)(mimg->zsize*scale[2]), p);
    } else {
      u = GetVoxelCoord((int)(mimg->xsize*scale[0]),
			(int)(mimg->ysize*scale[1]), 1, p);
    }	
    u.x = u.x / scale[0];
    u.y = u.y / scale[1];
    u.z = u.z / scale[2];
    
    int j      = 0; 
    for (int k = 0; k < A->n; k++) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, k);
      if (iftMValidVoxel(mimg, v)) {
	int q = iftMGetVoxelIndex(mimg, v);
	for (int b = 0; b < mimg->m; b++) {
	  Z->sample[*s].feat[j] = mimg->val[q][b];
	  j++;
	}
      } else {
	for (int b = 0; b < mimg->m; b++) {
	    Z->sample[*s].feat[j] = 0;
	    j++;
	  }
      }
    }
    (*s)++; Saux = Saux->next;
  } 
}
  
void SaveBias(char *basepath, float *bias, int number_of_kernels)
{
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

void SaveKernelWeights(char *basepath, int *truelabel, int number_of_kernels)
{
  char filename[200];
  FILE *fp;
  float pw=1, nw=-1;
  
  sprintf(filename, "%s-weights.txt", basepath);
  fp = fopen(filename, "w");
  fprintf(fp, "%d\n", number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++) {
    if (truelabel[k]==2)
      fprintf(fp, "%f ", pw);
    else
      fprintf(fp, "%f ", nw);      
  }
  fclose(fp);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    /* Example: iftCreateMergedLayerModel bag arch.json 1 flim_models */

    if (argc != 5)
      iftError("Usage: iftCreateMergedLayerModel P1 P2 P3 P4\n"
	       "P1: input folder with feature points (-fpts.txt)\n"
	       "P2: input network architecture (.json)\n"
	       "P3: input layer for patch definition (1, 2, 3, etc)\n"
	       "P4: output folder with the model\n",
	       "main");
    
    tstart = iftTic();

    /* Read input parameters */
    
    iftFileSet  *fs    = iftLoadFileSetFromDirBySuffix(argv[1], "-fpts.txt", 1);
    iftFLIMArch *arch  = iftReadFLIMArch(argv[2]);
    int          layer = atoi(argv[3]);
	  
    char *filename     = iftAllocCharArray(512);
    char *model_dir    = argv[4];
    iftMakeDir(model_dir);

    /* Find layer parameters: number of samples, number of features,
       adjacency relation, scale, and image dimension. */

    int nsamples       = 0;
    char *basename     = iftFilename(fs->files[0]->path, "-fpts.txt");    
    sprintf(filename, "./layer%d/%s%s", layer-1, basename, ".mimg");
    iftMImage *mimg    = iftReadMImage(filename);
    int dim            = 0;
    if (iftIs3DMImage(mimg))
      dim = 3;
    else
      dim = 2;
    iftAdjRel *A       = GetPatchAdjacency(mimg, arch->layer[layer-1]);
    int nfeats         = A->n * mimg->m;
    float scale[3];
    sprintf(filename,"./layer0/%s.mimg",basename);
    iftMImage *input = iftReadMImage(filename);
    scale[0] = (float)input->xsize/(float)mimg->xsize;
    scale[1] = (float)input->ysize/(float)mimg->ysize;
    scale[2] = (float)input->zsize/(float)mimg->zsize;
    iftDestroyMImage(&input);
    iftDestroyMImage(&mimg);
    iftFree(basename);
    
    for (int i=0; i < fs->n; i++) { /* count the total number of
				       feature points */
      iftLabeledSet *S = iftReadLabeledSet(fs->files[i]->path,dim);
      nsamples        += iftLabeledSetSize(S);
      iftDestroyLabeledSet(&S);
    }

    /* Create a dataset with patches from all feature points */    
      
    iftDataSet *Z = iftCreateDataSet(nsamples, nfeats);
    int s = 0;
    for (int i=0; i < fs->n; i++) {
      basename           = iftFilename(fs->files[i]->path, "-fpts.txt");    
      sprintf(filename, "./layer%d/%s%s", layer-1, basename, ".mimg");
      mimg               = iftReadMImage(filename);
      iftLabeledSet *S   = iftReadLabeledSet(fs->files[i]->path,dim);
      AddImagePatches(Z,&s,scale,dim,mimg,S,A);
      iftFree(basename);
      iftDestroyMImage(&mimg);
      iftDestroyLabeledSet(&S);
    }
    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);
    iftNormalizeDataSetByZScoreInPlace(Z,NULL,arch->stdev_factor);
    iftDestroyAdjRel(&A);

    /* Create model with filters and biases from the dataset Z */
    
    iftMatrix *kernels = iftCreateMatrix(Z->nsamples, Z->nfeats);
    int *truelabel     = iftAllocIntArray(Z->nsamples);
      
    /* compute filters and their truelabels */
    for (int s = 0, col=0; s < Z->nsamples; s++) {
      iftUnitNorm(Z->sample[s].feat, Z->nfeats);
      truelabel[s] = Z->sample[s].truelabel;
      for (int row = 0; row < Z->nfeats; row++){
	iftMatrixElem(kernels, col, row) = Z->sample[s].feat[row];
      }
      col++;
    }

    /* compute biases and update filters */
  
    float *bias     = NULL;
    bias = iftAllocFloatArray(kernels->ncols);
    for (int col=0; col < kernels->ncols; col++){
      for (int row=0; row < Z->nfeats; row++){
	iftMatrixElem(kernels,col,row) =
	  iftMatrixElem(kernels,col,row) / Z->fsp.stdev[row];
	bias[col] -= (Z->fsp.mean[row]*iftMatrixElem(kernels,col,row));
      }
    }

    /* save kernels, biases, and truelabels of the kernels */
  
    sprintf(filename, "%s/conv%d-kernels.npy", model_dir, layer);
    iftWriteMatrix(kernels,filename);
    sprintf(filename, "%s/conv%d", model_dir, layer);
    SaveBias(filename, bias, kernels->ncols);
    SaveKernelWeights(filename, truelabel, kernels->ncols);

    iftFree(bias);
    iftFree(truelabel);
    iftDestroyMatrix(&kernels);
    iftDestroyDataSet(&Z);
    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    iftFree(filename);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}


