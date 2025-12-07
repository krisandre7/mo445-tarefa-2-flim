#include "ift.h"

/* A code to classify by OPF voxels as arteriovenous malformation
   (AVM) or normal voxels. */

int main(int argc, const char *argv[]) {
  iftImage    *segm = NULL;
  iftCplGraph *graph;
  iftDataSet  *Z = NULL;
  size_t mem_start, mem_end;
  timer *t1, *t2;

  mem_start = iftMemoryUsed();

  if (argc != 4){
    printf("usage: iftAVMClass <dataset.zip (input)>  <classifier.zip (input)> <segmented-mask.scn (output)>\n");
    exit(1);
  }

  t1 = iftTic();

  Z         = iftReadOPFDataSet(argv[1]);
  graph     = iftReadCplGraph(argv[2]);

  /* Classify samples */

  printf("errors %f\n", (float)iftClassify(graph, Z)/Z->nsamples);

  /* Write segmentation mask */

  segm = iftCreateImageFromImage((iftImage *)Z->ref_data);
  for (int s=0; s < Z->nsamples; s++)
    if (Z->sample[s].label==2)
      segm->val[Z->sample[s].id] = 255;

  iftWriteImageByExt(segm,argv[3]);

  t2 = iftToc();
  puts(iftFormattedTime(iftCompTime(t1,t2)));

  mem_end = iftMemoryUsed();
  iftVerifyMemory(mem_start,mem_end);

  iftDestroyDataSet(&Z);
  iftDestroyCplGraph(&graph);
  iftDestroyImage(&segm);
  
  return 0;
}
