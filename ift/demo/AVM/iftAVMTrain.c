#include "ift.h"

/* A code to train an OPF classifier for arteriovenous malformation
   (AVM) segmentation from a set of features provided as input */


int main(int argc, const char *argv[]) {
  iftImage    *gt = NULL, *brainmask = NULL;
  iftCplGraph *graph;
  iftDataSet  *Z = NULL, *Z1 = NULL;
  size_t mem_start, mem_end;
  timer *t1, *t2;

  mem_start = iftMemoryUsed();

  if (argc != 6){
    printf("usage: iftAVMTrain <features-basename (input)> <ground-truth.scn (input)> <brainmask.scn (input)> <classifier.zip (output)> <dataset.zip (output)>\n");
    exit(1);
  }

  t1 = iftTic();

  gt        = iftReadImageByExt(argv[2]);
  brainmask = iftReadImageByExt(argv[3]);

  int nsamples = 0;
  for (int p=0; p < brainmask->n; p++) { /* counting the number of samples */
    if (brainmask->val[p]!=0)
      nsamples++;
  }

  printf("nsamples %d\n",nsamples);

  /* Creating the dataset for training an OPF classifier */
  
  Z = iftCreateDataSet(nsamples, 9);  
  for (int f=0; f < 9; f++) {
    char filename[200];
    sprintf(filename,"%s_%02d.scn",argv[1],f+1);
    iftImage *feat = iftReadImageByExt(filename);
    for (int p=0, s=0; p < brainmask->n; p++) {
      if (brainmask->val[p]!=0){
	Z->sample[s].feat[f] = feat->val[p];
	Z->sample[s].id      = p; 
	if (gt->val[p]!=0)
	  Z->sample[s].truelabel = 2; // AVM
	else
	  Z->sample[s].truelabel = 1; // normal
	s++;
      }
    }
    iftDestroyImage(&feat);
  }
  
  Z->nclasses = 2;
  Z->ref_data_type = IFT_REF_DATA_IMAGE;
  Z->ref_data      = (void *)brainmask;

  iftWriteOPFDataSet(Z,argv[5]);
  
  /* Randomly select training samples */
  
  iftSampler* sampler = iftRandomSubsampling(Z->nsamples, 5, 500);
  iftDataSetSampling(Z, sampler, 3);
  iftDestroySampler(&sampler);
  Z1  = iftExtractSamples(Z,IFT_TRAIN);

  /* Project the training samples */
  
  iftDataSet *Z2d = iftDimReductionByTSNE(Z1, 2, 40, 1000);
  iftImage   *img = iftDraw2DFeatureSpace(Z2d,CLASS,0); 
  iftWriteImageByExt(img,"projection.png");
  iftDestroyImage(&img);
  iftDestroyDataSet(&Z2d);
    
  /* Training the OPF classifier and save it for iftClassifyImageByOPF */
  
  graph = iftCreateCplGraph(Z1);
  iftSupTrain(graph);
  iftWriteCplGraph(graph, argv[4]);

  t2 = iftToc();
  puts(iftFormattedTime(iftCompTime(t1,t2)));

  mem_end = iftMemoryUsed();
  iftVerifyMemory(mem_start,mem_end);

  iftDestroyDataSet(&Z1);
  iftDestroyDataSet(&Z);
  iftDestroyImage(&gt);
  iftDestroyImage(&brainmask);
  iftDestroyCplGraph(&graph);
  
  return 0;
}
