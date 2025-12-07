#include "ift.h"

/* Author: Alexandre Xavier Falcão (September, 10th 2023) 

   Description: Executes a convolutional block to encode the current
   layer using the consolidated model of all training images.

*/


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

iftAdjRel *GetPoolAdjacency(iftMImage *mimg, iftFLIMLayer layer)
{
  iftAdjRel *A;

  if (iftIs3DMImage(mimg)){
    A = iftCuboidWithDilationForConv(layer.pool_size[0],
				     layer.pool_size[1],
				     layer.pool_size[2],
				     1,
				     1,
				     1);
  }else{
    A = iftRectangularWithDilationForConv(layer.pool_size[0],
					  layer.pool_size[1],
					  1,
					  1);    
  }

  return(A);
}

void LoadMergedModel(char *basepath, iftMatrix **K, float **bias)
{
  char filename[200];
  FILE *fp;
  int nkernels;
  
  sprintf(filename, "%s-kernels.npy", basepath);
  *K = iftReadMatrix(filename);
  sprintf(filename, "%s-bias.txt", basepath);
  fp = fopen(filename, "r");
  fscanf(fp, "%d", &nkernels);
  *bias = iftAllocFloatArray(nkernels);
  for (int k = 0; k < nkernels; k++) {
    fscanf(fp, "%f ", &((*bias)[k]));
  }  
  fclose(fp);
}

int main(int argc, char *argv[]) {
    timer *tstart;

    /* Example: iftEncodeMergedLayer arch.json 1 flim_models */
    
    if (argc!=5)
      iftError("Usage: iftEncodeMergedLayer <P1> <P2> <P3> <P4>\n"
	       "[1] architecture of the network (.json) \n"
	       "[2] layer number (1, 2, 3) \n"
	       "[3] folder with the models \n"
	       "[4] batch size (1, 8, 16, 32, etc) \n",
	       "main");

    iftFLIMArch *arch   = iftReadFLIMArch(argv[1]);
    int          layer  = atoi(argv[2]);
    char    *model_dir  = argv[3];
    char    *filename   = iftAllocCharArray(512);
    char     input_dir[20], output_dir[20]; 
    int      batchsize  = atoi(argv[4]);
    
    tstart = iftTic();

    sprintf(input_dir,"layer%d",layer-1);
    sprintf(output_dir,"layer%d",layer);
    iftMakeDir(output_dir);

    iftFileSet *fs   = iftLoadFileSetFromDirBySuffix(input_dir, ".mimg", true);
    iftMImage *mimg  = iftReadMImage(fs->files[0]->path);
    iftAdjRel *A     = GetPatchAdjacency(mimg, arch->layer[layer-1]);
    iftAdjRel *B     = GetPoolAdjacency(mimg,arch->layer[layer-1]);
    int stride       = arch->layer[layer-1].pool_stride;
    char *pool_type  = arch->layer[layer-1].pool_type;
    iftDestroyMImage(&mimg);

    iftMatrix *Kmerged     = NULL;
    float     *bias_merged = NULL;

    sprintf(filename,"%s/conv%d",model_dir,layer);
    LoadMergedModel(filename,&Kmerged,&bias_merged);
    
    /* Encode layer with merged model */      

    for (int i=0; i < fs->n; i += batchsize) {
      int first    = i;
      int last     = iftMin(i+batchsize-1,fs->n-1);
      /* read images */
      iftMImageArray *imgArray   = iftReadMImageBatch(fs,first,last);
      /* convolution */
      iftMImageArray *convArray = iftMConvolutionArray(imgArray, A, Kmerged,
							1);
      iftDestroyMImageArray(&imgArray);
      /* activation */
      iftMImageArray *reluArray  = iftMReLUArray(convArray,
						 bias_merged,
						 arch->layer[layer-1].relu);
      if (reluArray != convArray)
	iftDestroyMImageArray(&convArray);
      
      /* pooling */
      iftMImageArray *poolArray = iftMPoolingArray(reluArray,
						   pool_type,
						   B,
						   stride);
      iftDestroyMImageArray(&reluArray);

      /* write activations */
      
      iftWriteMImageBatch(poolArray,fs,first,last,output_dir);
      iftDestroyMImageArray(&poolArray);
    }
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    iftDestroyMatrix(&Kmerged);
    iftFree(bias_merged);
    iftFree(filename);
    iftDestroyFileSet(&fs);
    iftDestroyFLIMArch(&arch);
    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    
    return (0);
}
