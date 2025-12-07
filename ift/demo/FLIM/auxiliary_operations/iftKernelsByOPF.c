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
  iftKnnGraph     *graph=NULL;
  int              kmax, layer;
  size_t           mem_start, mem_end;
  timer           *t1, *t2;
  char             filename[300];
  
  mem_start = iftMemoryUsed();

  if (argc != 5){
    printf("Usage: iftKernelsByOPF <P1> <P2> <P3> <P4>\n");
    printf("P1: input/output folder with the model's parameters\n");
    printf("P2: input kmax -- integer/percentage (fraction) of samples\n");
    printf("P3: input layer (integer 1,2,...)\n");
    printf("P4: input/output patch dataset (.zip) in P1 folder\n");
    exit(-1);
  }

  /* BIAS */
  
  char  *use_bias = getenv("USE_BIAS");
  if (use_bias == NULL){
    printf("Define: export USE_BIAS=1\n");
    exit(-1);
  }
  
  t1 = iftTic();

  /* Load input parameters */

  char *model_dir = argv[1];
  iftMakeDir(model_dir);
  
  layer = atoi(argv[3]);

  sprintf(filename,"%s/%s",model_dir,argv[4]);
  Z     = iftReadDataSet(filename);
  
  if (atoi(argv[2]) >= 1)
    kmax=iftMin(atoi(argv[2]),Z->nsamples/2);
  else
    kmax=iftMax(atof(argv[2])*Z->nsamples,1);   

  char archname[100];
  sprintf(archname,"%s/arch.json",model_dir);
  iftFLIMArch *arch = iftReadFLIMArch(archname);

  // Compute clusters by Knn
  
  graph = iftCreateKnnGraph(Z,kmax);

  iftUnsupTrain(graph,iftNormalizedCut);
  iftDestroyKnnGraph(&graph);
 
  iftWriteDataSet(Z,filename);

  printf("Number of groups: %d\n",Z->ngroups);

  // Update architecture 
  arch->layer[layer-1].noutput_channels = Z->ngroups;
  iftWriteFLIMArch(arch,archname);
  
  // Convert prototypes into kernels 

  iftMatrix *kernels = iftCreateMatrix(Z->ngroups, Z->nfeats);
  
  for (int s = 0, col=0; s < Z->nsamples; s++) {
    if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)) {
      iftUnitNorm(Z->sample[s].feat, Z->nfeats);
      for (int row = 0; row < Z->nfeats; row++){
	iftMatrixElem(kernels, col, row) = Z->sample[s].feat[row];
      }
      col++;
    }
  }

  // Estimate bias and update kernels
  
  float *bias     = NULL;
  bias = iftAllocFloatArray(kernels->ncols);
  for (int col=0; col < kernels->ncols; col++){
    for (int row=0; row < Z->nfeats; row++){
      iftMatrixElem(kernels,col,row) =
	iftMatrixElem(kernels,col,row) / Z->fsp.stdev[row];
      bias[col] -= (Z->fsp.mean[row]*iftMatrixElem(kernels,col,row));
    }
  }

  // Save output kernels and bias
  
  sprintf(filename, "%s/conv%d-kernels.npy", model_dir, layer);
  iftWriteMatrix(kernels,filename);
  printf("ncols (nkernels) %d nrows (nfeats) %d\n", kernels->ncols, kernels->nrows);

  sprintf(filename, "%s/conv%d", model_dir, layer);
  MyWriteBias(filename, bias, kernels->ncols);
  iftFree(bias);
  iftDestroyDataSet(&Z);
  iftDestroyMatrix(&kernels);
  
  mem_end = iftMemoryUsed();
  iftVerifyMemory(mem_start,mem_end);
    
  t2 = iftToc();
  puts(iftFormattedTime(iftCompTime(t1,t2)));

  return(0);
}
