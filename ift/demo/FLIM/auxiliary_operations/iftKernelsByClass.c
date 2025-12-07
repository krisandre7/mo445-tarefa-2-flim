#include "ift.h"

void MyWriteBias(char *basepath, float *bias, int number_of_kernels) {
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

int main(int argc, char *argv[]) 
{
  iftDataSet      *Z=NULL;
  int              truelabel;
  size_t           mem_start, mem_end;
  timer           *t1, *t2;

  mem_start = iftMemoryUsed();

  if (argc != 5){
    printf("Usage: iftKernelsByClass <P1> <P2> <P3>\n");
    printf("P1: input/output folder with the model's parameters\n");
    printf("P2: input class to create a kernel bank (integer 1,2,...)\n");
    printf("P3: input/output single-layer network archtecture (.json)\n");
    printf("P4: input layer (integer 1,2,...)\n");
    exit(-1);
  }

  /* BIAS */
  
  char  *use_bias = getenv("USE_BIAS");
  if (use_bias == NULL){
    printf("Define: export USE_BIAS=1\n");
    exit(-1);
  }
  
  t1 = iftTic();

  // Read input parameters

  char *model_dir = argv[1];
  iftMakeDir(model_dir);
  char filename[300];
  
  truelabel         = atoi(argv[2]);
  char *arch_name   = argv[3];
  int layer         = atoi(argv[4]);

  iftFLIMArch *arch = iftReadFLIMArch(arch_name);

  if (arch->nlayers != 1)
    iftError("Architecture must have a single layer (nlayers=%d)",
	     "main",arch->nlayers);
  
  if (truelabel < 1)
    iftError("Class must be >= 1 (class %d)","main",truelabel);

  if (layer < 1)
    iftError("Layer must be >= 1 (layer %d)","main",layer);
  
  sprintf(filename,"%s/patches%d.zip",model_dir,layer);

  Z                 = iftReadDataSet(filename);
  
  // Compute kernel bank 

  int *nsamples_per_class = iftCountSamplesPerClassDataSet(Z);
  iftMatrix *kernels      = iftCreateMatrix(nsamples_per_class[truelabel],
					    Z->nfeats);
  iftFree(nsamples_per_class);

  for (int s = 0, col=0; s < Z->nsamples; s++) {
    if (Z->sample[s].truelabel == truelabel) {
      iftUnitNorm(Z->sample[s].feat, Z->nfeats);
      for (int row = 0; row < Z->nfeats; row++){
	iftMatrixElem(kernels, col, row) = Z->sample[s].feat[row];
      }
      col++;
    }
  }

  // Update network architecture

  arch->layer[0].noutput_channels = kernels->ncols;
  iftWriteFLIMArch(arch,arch_name);
  
  // Compute and save bias, incorporating it in the kernels
  
  float *bias     = NULL;
  bias = iftAllocFloatArray(kernels->ncols);
  
  for (int col=0; col < kernels->ncols; col++){ 
    for (int row = 0; row < kernels->nrows; row++) { 
      iftMatrixElem(kernels,col,row) = 
	iftMatrixElem(kernels,col,row) / Z->fsp.stdev[row]; 
      bias[col] -= (Z->fsp.mean[row]*iftMatrixElem(kernels,col,row)); 
    } 
  } 

  sprintf(filename, "%s/conv1-kernels.npy", model_dir);
  iftWriteMatrix(kernels,filename);
  printf("ncols (nkernels) %d nrows (nfeats) %d\n", kernels->ncols, kernels->nrows);
  
  sprintf(filename, "%s/conv1", model_dir);
  MyWriteBias(filename, bias, kernels->ncols);
  iftFree(bias);
  iftDestroyDataSet(&Z);
  iftDestroyMatrix(&kernels);
  iftDestroyFLIMArch(&arch);
  
  mem_end = iftMemoryUsed();
  iftVerifyMemory(mem_start,mem_end);
    
  t2 = iftToc();
  puts(iftFormattedTime(iftCompTime(t1,t2)));

  return(0);
}
