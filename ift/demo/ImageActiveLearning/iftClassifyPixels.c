#include "ift.h"

int main(int argc, char* argv[])
{
  if (argc < 5) {
    fprintf(stderr, "Usage: %s <origImg> <gtLabelMap> <n_train> <result.zip>\n", argv[0]);
    return -1;
  }

  iftImage *img = iftReadImageByExt(argv[1]);
  iftImage *gtLabelMap = iftReadImageByExt(argv[2]);
  int nTrain = atol(argv[3]);

  iftDataSet *Z = iftImageToDataSet(img);
  iftSetStatus(Z, IFT_TEST);

  // Set true label based on gt
  Z->nclasses = iftMaximumValue(gtLabelMap);
  for (int s = 0; s < Z->nsamples; ++s) {
    int p = Z->sample[s].id;
    Z->sample[s].truelabel = gtLabelMap->val[p];
  }

  // -- Sample random pixels for training
  srand(time(NULL));
  iftRandomSelector *rs = iftCreateRandomSelectorDefaultValues(Z->nsamples);
  assert(nTrain > 0);
  for (int i = 0; i < nTrain; ++i) {
    int s = iftPickFromRandomSelector(rs, false);
    Z->sample[s].status = IFT_TRAIN;
    Z->ntrainsamples += 1;
  }

  // -- Train & classify
  iftCplGraph *graph = iftCreateCplGraph(Z);
  iftSupTrain(graph);
  iftClassify(graph, Z);

  // -- Save dataset
  iftWriteDataSet(Z, argv[4]);

  iftDestroyImage(&img);
  iftDestroyImage(&gtLabelMap);
  iftDestroyDataSet(&Z);
  iftDestroyCplGraph(&graph);
  iftDestroyRandomSelector(&rs);

  return 0;
}
