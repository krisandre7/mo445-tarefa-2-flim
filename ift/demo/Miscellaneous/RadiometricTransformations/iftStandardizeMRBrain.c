#include "ift.h"

#define EXT3D ".nii.gz"
	
int main(int argc, char *argv[])
{
  timer *tstart;
  
  if (argc != 3)
    iftError("Usage: iftStandardizeMRBrain <...>\n"
	     "[1] Input folder with t1gd/flair images. \n"
	     "[2] Output folder \n",	       
	     "main");
  
  tstart = iftTic();
  
  iftFileSet *fs_orig    = iftLoadFileSetFromDirBySuffix(argv[1],EXT3D, 1);
  int nimages            = fs_orig->n;   
  char *out_dir          = argv[2];
  char filename[200];

  iftMakeDir(out_dir);
  
  for (int i=0; i < nimages; i++){
    char *basename       = iftFilename(fs_orig->files[i]->path,EXT3D);
    
    printf("Processing file %s: %d of %d files\n", basename, i + 1, nimages);
    fflush(stdout);
    
    sprintf(filename,"%s/%s%s",argv[1],basename,EXT3D);
    iftImage *aux   = iftReadImageByExt(filename);
    iftImage *input = iftNormalizeWithNoOutliersInRegion(aux,NULL,0,4095,99.0);
    iftDestroyImage(&aux);

    iftHist  *hist  = iftCalcGrayImageHist(input, NULL, 4096, 4095, false);

    int moda = 1;
    for (int i=moda+1; i < hist->nbins-1; i++)
      if (hist->val[i]>hist->val[moda])
	moda = i;
    
    iftDestroyHist(&hist);
    
    iftImage *output = iftWindowAndLevel(input,3000,moda,4095);
    sprintf(filename,"%s/%s%s",out_dir,basename,EXT3D);
    iftWriteImageByExt(output,filename);		
    iftDestroyImage(&input);
    iftDestroyImage(&output);
    iftFree(basename);
  }

  iftDestroyFileSet(&fs_orig);

  printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
