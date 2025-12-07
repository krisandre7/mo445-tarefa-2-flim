#include "ift.h"
#include "iftHybridNetwork.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage: %s <fileset.csv> <new dir path>\n", argv[0]);
    return -1;
  }

  iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1], 1, false);
  iftUpdateFileSetDir(fs, argv[2]);
  iftWriteFileSetAsCSV(fs, argv[1]);

  iftDestroyFileSet(&fs);

  return 0;
}
