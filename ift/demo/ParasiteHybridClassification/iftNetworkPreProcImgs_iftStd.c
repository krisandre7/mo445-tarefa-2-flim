#include "ift.h"
#include "iftHybridNetwork.h"

int main(int argc, char* argv[])
{
  if (argc < 5)
    iftError("%s <orig folder> <label folder> <impurity label start (eggs=9, larvae=2, protozoa=7)> <output folder>", "main", argv[0]);

  iftFileSet *fs1 = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);
  iftFileSet *fs2 = iftLoadFileSetFromDirOrCSV(argv[2], 1, 1);
  int impurityStart = atol(argv[3]);
  const char *outFolder = argv[4];

  iftMakeDir(outFolder);

  int nClasses = iftFileSetLabelsNumber(fs1);
  int nImgs = fs1->n;

  // --- Check if orig and label folders match
  if (nClasses != iftFileSetLabelsNumber(fs2))
    iftError("Directories do not have the same classes", "main");

  if (nImgs != fs2->n)
    iftError("Directories do not have the same number of images", "main");

  int *nSampPerClass1 = iftCountSamplesPerClassFileSet(fs1);
  int *nSampPerClass2 = iftCountSamplesPerClassFileSet(fs2);
  for (int c = 0; c < nClasses; ++c) 
    if (nSampPerClass1[c] != nSampPerClass2[c])
      iftError("Class %d does not have the same number of images and masks", "main", c);
  iftFree(nSampPerClass1);
  iftFree(nSampPerClass2);

  // --- Actually process images
  nClasses = iftMin(impurityStart, nClasses);
  int *classCount = iftAllocIntArray(nClasses);
  timer *t1 = iftTic();
  for (int i = 0; i < nImgs; ++i) {
    iftImage *origImg = iftReadImageByExt(fs1->files[i]->path);
    iftImage *labelImg = iftReadImageByExt(fs2->files[i]->path);

    printf("Processing image %d of %d\n", i + 1, nImgs);
    iftImage *resImg = iftHybridNetworkPreProcImg(origImg, labelImg);

    int label = iftMin(fs1->files[i]->label, nClasses);
    int id = ++classCount[label];

    char basename[200];
    sprintf(basename,"%06d_%08d", label, id);      
    char filename[200];
    sprintf(filename, "%s/%s.png", outFolder, basename);

    iftWriteImageByExt(resImg, filename);
    
    iftDestroyImage(&origImg);
    iftDestroyImage(&labelImg);    
    iftDestroyImage(&resImg);
  }
  timer *t2 = iftToc();
  float procTime = iftCompTime(t1, t2);
  printf("Finished in %fs (average %fms per sample)\n", procTime / 1000.0f, procTime / ((float) nImgs));

  iftDestroyFileSet(&fs1);
  iftDestroyFileSet(&fs2);

  return 0;
}
