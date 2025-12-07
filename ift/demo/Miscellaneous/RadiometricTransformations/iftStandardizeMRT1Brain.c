#include "ift.h"

#define EXT3D ".nii.gz"
	
int main(int argc, char *argv[])
{
  timer *tstart;
  
  if (argc != 4)
    iftError("Usage: iftStandardizeMRT1Brain <...>\n"
	     "[1] Input dir with MRT1 images. \n"
	     "[2] input dir with label images. \n"
	     "[3] Output dir\n",	       
	     "main");
  
  tstart = iftTic();
  
  iftFileSet *fs_orig    = iftLoadFileSetFromDirBySuffix(argv[1],EXT3D, 1);
  iftFileSet *fs_label   = iftLoadFileSetFromDirBySuffix(argv[2],EXT3D, 1);
  if (fs_orig->n != fs_label->n)
    iftError("Original and label images are not the same in number","main");	
  int nimages            = fs_orig->n;   
  iftMakeDir(argv[3]);
  char filename[200];
  
  for (int i=0; i < nimages; i++){
    char *basename       = iftFilename(fs_orig->files[i]->path,EXT3D);
    
    printf("Processing file %s: %d of %d files\n", basename, i + 1, nimages);
    fflush(stdout);
    
    sprintf(filename,"%s/%s%s",argv[1],basename,EXT3D);
    iftImage *input       = iftReadImageByExt(filename);
    sprintf(filename,"%s/%s%s",argv[2],basename,EXT3D);
    iftImage *label       = iftReadImageByExt(filename);

    iftImage *aux = iftLinearStretch(input, iftMinimumValue(input), iftMaximumValue(input), 0, 4095);
    iftDestroyImage(&input);
    input = aux;
    
    iftHist  *hist = iftCalcGrayImageHist(input, label, 4095, 4095, false);
    int       mode = iftHistMode(hist, true);
    printf("mode %d\n",mode);
    iftImage *output = iftWindowAndLevel(input, 2000, mode, 4095);
    iftDestroyHist(&hist);
    sprintf(filename,"%s/%s%s",argv[3],basename,EXT3D);
    iftWriteImageByExt(output,filename);		
    iftDestroyImage(&input);
    iftDestroyImage(&label);
    iftDestroyImage(&output);
    iftFree(basename);
  }

  iftDestroyFileSet(&fs_orig);
  iftDestroyFileSet(&fs_label);

  printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
