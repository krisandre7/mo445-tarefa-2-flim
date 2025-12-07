#include "ift.h"


iftDataSet *iftSelectPatchesFromSeeds(iftMImage *img, iftConvNetwork *convnet, iftLabeledSet *Seed, int label)
{
  iftDataSet *Z;
  int dx, dy, dz, nsamples=0; 
  iftAdjRel *A; 
  iftLabeledSet *S=NULL;

  A = convnet->k_bank[0]->A;
  iftMaxAdjShifts(A, &dx, &dy, &dz);
  S = Seed;
  while(S != NULL) {
    if (S->label==label)
      nsamples++;
    S = S->next;
  }

  Z=iftCreateDataSet(nsamples,A->n*img->m);

  Z->ref_data = img;

  S = Seed; int s = 0;
  while(S != NULL) {
    int p = S->elem;
    if (S->label==label) {
      iftVoxel u  = iftMGetVoxelCoord(img,p);
      Z->sample[s].id = p;
      Z->sample[s].truelabel = S->label+1;
      int j=0;
      for (int b=0; b < img->m; b++) {
	for (int i=0; i < A->n; i++) {
	  iftVoxel v = iftGetAdjacentVoxel(A,u,i);
	  if (iftMValidVoxel(img,v)){
	    int    q = iftMGetVoxelIndex(img,v);
	    Z->sample[s].feat[j] = img->val[q][b];
	  }
	  j++;
	}
      }
      s++;
    }  
    S = S->next; 
  }
  printf("nsamples %d\n",s);

  iftSetStatus(Z, IFT_TRAIN);

  return(Z);
}

iftMMKernel *iftSupLearnKernels(iftDataSet* Z,iftAdjRel* A, char whitening)
{
  iftDataSet  *Zp;
  iftKnnGraph *graph=NULL;
  iftMMKernel *K;
  int          nkernels,nbands;

  if (whitening)
    Zp = iftWhiteningTransform(Z);
  else
    Zp = Z;

  /* iftNormalizeSamples(Zp); */
  /* Zp->iftArcWeight = iftDistance6; */


  /* Compute sup learning for over 2000 samples */
  if (Zp->nsamples <= 2000) {
    graph = iftCreateKnnGraph(Zp,(int)(Zp->nsamples*0.05));
    iftUnsupTrain(graph,iftNormalizedCut);
  } else {
    iftSetStatus(Zp,IFT_TEST);
    float train_perc = 2000.0/(float)Zp->nsamples;
    iftSelectUnsupTrainSamples(Zp,train_perc);
    graph = iftUnsupLearn(Zp,0.05,iftNormalizedCut,10);
  }

  /* Select kernels from prototypes */
  nkernels = 0;
  for (int u=0; u < graph->nnodes; u++)  {
    if (graph->node[u].root==u) {
      nkernels ++;
    }
  }
  nbands = Zp->nfeats/A->n;

  printf("Creating bank with %d kernels \n",nkernels);

  K = iftCreateMMKernel(A,nbands,nkernels); //img->m 
  int k=0;
  for (int u=0; u < graph->nnodes; u++)  {
    int s = graph->node[u].sample;
    if (graph->node[u].root==u) { 
      int j = 0;
      for (int b=0; b < nbands; b++) { // img->m  
	for (int i=0; i < A->n; i++) {
	  K->weight[k][b].val[i] = Zp->sample[s].feat[j];
	  j++;
	}
      }
      k++;
    }
  }

  if (Zp->fsp.W != NULL){
    K->W     = iftCopyMatrix(Zp->fsp.W);
    K->mean  = iftAllocFloatArray(Zp->nfeats);
    K->stdev = iftAllocFloatArray(Zp->nfeats);
    for (int i=0; i < Zp->nfeats; i++) {
      K->mean[i]  = Zp->fsp.mean[i];
      K->stdev[i] = Zp->fsp.stdev[i];
    }
  }

  if (graph != NULL)
    iftDestroyKnnGraph(&graph);
  if (Z != Zp)
    iftDestroyDataSet(&Zp);

  return(K);
}

void iftSupLearnKernelsFromSeeds(iftMImage *img, iftConvNetwork *convnet, iftLabeledSet *S, char whitening, int label)
{
  iftDataSet  *Z;
  iftMMKernel *K;

  if (convnet->nlayers != 1) 
    iftError("It requires a single-layer convolution network","iftSupLearnKernelsFromSeedClusters");

  Z  = iftSelectPatchesFromSeeds(img, convnet, S, label);
   
  K = iftSupLearnKernels(Z,convnet->k_bank[0]->A,whitening);

  iftDestroyMMKernel(&convnet->k_bank[0]);
  convnet->k_bank[0] = K;
  convnet->nkernels[0] = K->nkernels;
  iftDestroyDataSet(&Z);
}

int main(int argc, char **argv) 
{
  iftMImage      *input, *output;
  iftImage       *orig;
  iftLabeledSet  *S=NULL, *Saux=NULL;
  char            filename[200];
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *convnet;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=5)
    iftError("Usage: iftSupLearnKernelsFromSeeds <input.mig> <input_parameters.convnet> <markers.txt> <orig.ppm> ","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"mig")==0){
    input = iftReadMImage(argv[1]);    
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }
  
  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  iftRandomSeed(IFT_RANDOM_SEED);

  if (strcmp(ext,"convnet")==0){
    convnet = iftReadConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  pos = strrchr(argv[4],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"ppm")==0){
    orig = iftReadImageP6(argv[4]);    
  }else{
    if (strcmp(ext,"pgm")==0){
      orig = iftReadImageP5(argv[4]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  Saux = iftReadSeeds2D(argv[3],orig);
  S    = iftAdjustSeedCoordinates(Saux,orig,input);
  iftDestroyLabeledSet(&Saux);

  /* Compute kernels by supervised learning */

  t1     = iftTic(); 

  iftSupLearnKernelsFromSeeds(input,convnet,S,0,1); 
  output = iftApplyConvNetwork(input,convnet);   

  t2     = iftToc();
  fprintf(stdout,"Kernels learned in %f ms\n",iftCompTime(t1,t2));


  /* Write features */

  
  t1 = iftTic();
  for (int i=0; i < convnet->nkernels[0]; i++) {
    iftImage *img = iftMImageToImage(output,255,i);
    if (img != NULL){
      sprintf(filename,"feat%d.pgm",i);
      iftWriteImageP5(img,filename);
      iftDestroyImage(&img);
    }
  }
  t2     = iftToc();
  fprintf(stdout,"Output features written in %f ms\n",iftCompTime(t1,t2));
  

  /* Compute basins and watershed from gray marker */

  t1 = iftTic();

  iftAdjRel *A=iftCircular(sqrtf(2.0));
  iftImage *basins=iftMImageBasins(output,A);
  iftWriteImageP2(basins,"basinsByCNN.pgm");
  iftImage *marker = iftVolumeClose(basins,1000);
  iftImage *label  = iftWaterGray(basins,marker,A);
  iftDestroyImage(&marker);
  iftDestroyImage(&basins);
  iftColor RGB, YCbCr;
  RGB.val[0] = 255; RGB.val[1] = 0; RGB.val[2] = 0;
  YCbCr            = iftRGBtoYCbCr(RGB);
  iftAdjRel *B     = iftCircular(0.0);
  int         size = (orig->xsize - output->xsize)/2;
  iftImage *watergray    = iftRemFrame(orig,size);
  free(watergray->Cb); free(watergray->Cr); 
  watergray->Cb = watergray->Cr = NULL;
  iftDrawBorders(watergray,label,A,YCbCr,B);
  iftWriteImageP6(watergray,"WatergrayByCNN.ppm");
  iftDestroyImage(&label);
  iftDestroyImage(&watergray);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  t2     = iftToc();
  fprintf(stdout,"Watergray computed in %f ms\n",iftCompTime(t1,t2));

  iftDestroyLabeledSet(&S);
  iftDestroyImage(&orig);
  iftDestroyConvNetwork(&convnet);
  iftDestroyMImage(&input);
  iftDestroyMImage(&output);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
