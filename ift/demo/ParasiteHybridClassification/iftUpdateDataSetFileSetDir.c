#include "ift.h"
#include "iftHybridNetwork.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage: %s <dataset.zip> <new dir path>\n", argv[0]);
    return -1;
  }

  iftDataSet *Z = iftReadDataSet(argv[1]);
  if (Z->ref_data_type != IFT_REF_DATA_FILESET) {
    printf("DataSet does not contain fileset ref data!\n");
    return -1;
  }

  iftFileSet *fs = Z->ref_data;
  iftUpdateFileSetDir(fs, argv[2]);
  iftWriteDataSet(Z, argv[1]);

  iftDestroyDataSet(&Z);

  return 0;
}
