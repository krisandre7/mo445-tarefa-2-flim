#include "ift.h"

int main(int argc, char* argv[])
{
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <potsdam.zip> <n_train> <result.zip>\n", argv[0]);
    return -1;
  }

  iftDataSet *Z = iftReadDataSet(argv[1]); 
  int nTrain = atol(argv[2]);

  // Fix missing id
  for (int s = 0; s < Z->nsamples; ++s)
    Z->sample[s].id = s;

  iftSetStatus(Z, IFT_TEST);

  // -- Sample random pixels for training
  srand(time(NULL));
  iftRandomSelector *rs = iftCreateRandomSelectorDefaultValues(Z->nsamples);
  for (int i = 0; i < nTrain; ++i) {
    int s = iftPickFromRandomSelector(rs, false);
    Z->sample[s].status = IFT_TRAIN;
    Z->ntrainsamples += 1;
  }

  // -- Train & classify
  iftCplGraph *graph = iftCreateCplGraph(Z);
  iftSupTrain(graph);
  iftClassify(graph, Z);
  printf("acc = %f\n", iftTruePositives(Z));

  // -- Save dataset
  iftWriteDataSet(Z, argv[3]);

  iftDestroyDataSet(&Z);
  iftDestroyCplGraph(&graph);
  iftDestroyRandomSelector(&rs);

  return 0;
}
