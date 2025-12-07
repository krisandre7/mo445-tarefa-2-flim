#include <ift.h>

int main(int argc, char * argv[])
{
  if (argc < 4) {
    printf("Usage: %s <input_dataset.zip> <output_train.zip> <output_test.zip>\n", argv[0]);
    return 1;
  }

  iftRandomSeed(time(NULL));
  iftDataSet* Z = iftReadOPFDataSet(argv[1]);
  iftSelectUnsupTrainSamples(Z, 0.8);
  iftDataSet *Ztrain = iftExtractSamples(Z, IFT_TRAIN);
  iftDataSet *Ztest = iftExtractSamples(Z, IFT_TEST);
  iftWriteOPFDataSet(Ztrain, argv[2]);
  iftWriteOPFDataSet(Ztest, argv[3]);

  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&Ztrain);
  iftDestroyDataSet(&Ztest);
  return 0;
}
