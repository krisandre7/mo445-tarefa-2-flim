#include "ift.h"

int main(int argc, char *argv[])
{
  if (argc < 6) {
    printf("Usage: %s <full dataset.zip> <train.csv> <test.csv> <train dataset.zip> <test dataset.zip>\n", argv[0]);
    return -1;
  }

  iftDataSet *Z = iftReadDataSet(argv[1]);
  iftFileSet *fsTrain = iftLoadFileSetFromDirOrCSV(argv[2], 1, 1);
  iftFileSet *fsTest = iftLoadFileSetFromDirOrCSV(argv[3], 1, 1);
  char *outTrainPath = argv[4];
  char *outTestPath = argv[5];


  iftDataSet *ZTrain = iftCreateDataSet(fsTrain->n, Z->nfeats);
  ZTrain->ref_data_type = IFT_REF_DATA_FILESET; 
  ZTrain->ref_data = fsTrain;
  ZTrain->nclasses = Z->nclasses;
  ZTrain->ntrainsamples = ZTrain->nsamples;
  ZTrain->iftArcWeight = Z->iftArcWeight;
  ZTrain->function_number = Z->function_number;

  iftDataSet *ZTest = iftCreateDataSet(fsTest->n, Z->nfeats);
  ZTest->ref_data_type = IFT_REF_DATA_FILESET; 
  ZTest->ref_data = fsTest;
  ZTest->nclasses = Z->nclasses;
  ZTest->iftArcWeight = Z->iftArcWeight;
  ZTest->function_number = Z->function_number;

  iftFileSet *fsFull = Z->ref_data;
  int trainIdx = 0;
  int testIdx = 0;
  for (int i = 0; i < Z->nsamples; ++i) {
    int label = fsFull->files[i]->label;
    int id = fsFull->files[i]->sample;
    if (fsTrain->files[trainIdx]->label == label &&
        fsTrain->files[trainIdx]->sample == id) {
      iftCopySample(&(Z->sample[i]), &(ZTrain->sample[trainIdx]), Z->nfeats, true);
      trainIdx += 1;
    } else if (fsTest->files[testIdx]->label == label &&
        fsTest->files[testIdx]->sample == id) {
      iftCopySample(&(Z->sample[i]), &(ZTest->sample[testIdx]), Z->nfeats, true);
      testIdx += 1;
    } else {
      printf("full %d = (%d,%d)\ntrain %d = (%d,%d)\ntest %d = (%d,%d)\n", i, label, id,
          trainIdx, fsTrain->files[trainIdx]->label, fsTrain->files[trainIdx]->sample,
          testIdx, fsTest->files[testIdx]->label, fsTest->files[testIdx]->sample);
      iftError("Csvs do not match dataset.", "main");
    }
  }

  iftWriteDataSet(ZTrain, outTrainPath);
  iftWriteDataSet(ZTest, outTestPath);

  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&ZTrain);
  iftDestroyDataSet(&ZTest);

  return 0;
}

