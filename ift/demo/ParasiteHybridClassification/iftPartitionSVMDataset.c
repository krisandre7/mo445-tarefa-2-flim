#include "ift.h"

int main(int argc, char *argv[])
{
  if (argc < 4) {
    printf("Usage: %s <full dataset.zip> <partition.csv> <partition dataset.zip>\n", argv[0]);
    return -1;
  }

  iftDataSet *Z = iftReadDataSet(argv[1]);
  iftFileSet *fsPart = iftLoadFileSetFromDirOrCSV(argv[2], 1, 1);
  char *outPartPath = argv[3];


  iftDataSet *ZPart = iftCreateDataSet(fsPart->n, Z->nfeats);
  ZPart->ref_data_type = IFT_REF_DATA_FILESET; 
  ZPart->ref_data = fsPart;
  ZPart->ngroups = Z->ngroups;
  ZPart->nclasses = Z->nclasses;
  iftSetStatus(ZPart, IFT_TRAIN);
  ZPart->ntrainsamples = ZPart->nsamples;
  ZPart->iftArcWeight = Z->iftArcWeight;
  ZPart->function_number = Z->function_number;
  ZPart->fsp = iftCopyFeatSpaceParam(Z);
  for (int i = 0; i < ZPart->nfeats; ++i)
    ZPart->alpha[i] = Z->alpha[i];
  assert(Z->projection == NULL);

  iftFileSet *fsFull = Z->ref_data;
  for (int partIdx = 0; partIdx < fsPart->n; ++partIdx) {
    bool foundMatch = false;
    int partLabel = fsPart->files[partIdx]->label;
    int partId = fsPart->files[partIdx]->sample;

    for (int i = 0; i < Z->nsamples; ++i) {
      int fullLabel = fsFull->files[i]->label;
      int fullId = fsFull->files[i]->sample;
      if (partLabel == fullLabel && partId == fullId) {
        iftCopySample(&(Z->sample[i]), &(ZPart->sample[partIdx]), Z->nfeats, true);
        foundMatch = true;
        break;
      }
    }

    if (!foundMatch) {
      printf("Sample with label=%d and id=%d in csv is not present in full dataset.\n", partLabel, partId);
      return -1;
    }
  }

  iftWriteDataSet(ZPart, outPartPath);

  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&ZPart);

  return 0;
}

