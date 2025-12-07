#include "ift.h"
#include "iftHybridNetwork.h"

int main(int argc, char* argv[])
{
    static int FOLDER_COL = 0;
    static int LABEL_COL = 1;
    static int PATH_MAX_LEN = 512;

  if (argc < 4)
    iftError("%s <base_folder> <folder_map.csv> <output_folder>", "main", argv[0]);

  char *basePath = argv[1];
  iftCSV *csv = iftReadCSV(argv[2], ',');
  char *tgtPath = argv[3];

  int nClasses = 0;
  for (int row = 0; row < csv->nrows; ++row)
    nClasses = iftMax(nClasses, atol(csv->data[row][LABEL_COL]));
  if (nClasses <= 0)
    iftError("Could not read label information from column %d of the csv", "main", LABEL_COL);
  int *classCounter = iftAllocIntArray(nClasses);

  iftMakeDir(tgtPath);

  for (int row = 0; row < csv->nrows; ++row) {
    int label = atol(csv->data[row][LABEL_COL]);
    char folderFullPath[PATH_MAX_LEN];
    sprintf(folderFullPath, "%s/%s", basePath, csv->data[row][FOLDER_COL]);

    printf("Processing folder %d/%ld (%s)\n", row + 1, csv->nrows, folderFullPath);

      static char *objRegex = "^.*\\.obj\\.png";
    iftDir *objPaths = iftLoadFilesFromDirByRegex(folderFullPath, objRegex); 
    if (objPaths->nfiles <= 0)
      iftError("Folder %s has no files ending in '.obj.png'!", "main", folderFullPath);

    timer *t1 = iftTic();
    for (int file = 0; file < objPaths->nfiles; ++file) {
      printf("Processing image %d/%ld\n", file + 1, objPaths->nfiles);

      char *objPath = objPaths->files[file]->path;
      char mskPath[PATH_MAX_LEN];
      strcpy(mskPath, objPath);
      strcpy(&mskPath[strlen(mskPath) - 8], ".msk.pgm");
      if (!iftPathnameExists(mskPath))
        strcpy(&mskPath[strlen(mskPath) - 8], ".msk.png");

      printf("Reading image %d - %s\n", file, objPath);
      iftImage *origImg = iftReadImageByExt(objPath);
      if (iftImageDepth(origImg) != 16) {
        iftWarning("Skipping image \"%s\" which does not have 16-bit depth\n", "main", objPath);
        iftDestroyImage(&origImg);
        continue;
      }

      printf("Reading image %d - %s\n", file, mskPath);
      iftImage *labelImg = iftReadImageByExt(mskPath);
      iftImage *resImg = iftHybridNetworkPreProcImg(origImg, labelImg);

      int id = ++classCounter[label];

      char basename[PATH_MAX_LEN];
      sprintf(basename,"%06d_%08d", label, id);      
      char filename[PATH_MAX_LEN];
      sprintf(filename, "%s/%s.png", tgtPath, basename);

      iftWriteImageByExt(resImg, filename);

      iftDestroyImage(&origImg);
      iftDestroyImage(&labelImg);    
      iftDestroyImage(&resImg);
    }
    timer *t2 = iftToc();
    float procTime = iftCompTime(t1, t2);
    printf("Folder %s with %ld samples finished in %fs (average %fms per sample)\n",
        folderFullPath, objPaths->nfiles, procTime / 1000.0f, procTime / ((float) objPaths->nfiles));

    iftDestroyDir(&objPaths);
  }

  iftDestroyCSV(&csv);
  iftFree(classCounter);

  return 0;
}
