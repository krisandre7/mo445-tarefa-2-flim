#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 3)
      iftError("Usage: iftFLIM-BoVW P1 P2\n"
	       "P1: input  folder with the activations (.mimg)\n"
	       "P2: output dataset with BoVW features (.zip)\n",
	       "main");
    
    tstart = iftTic();

    iftFileSet  *fs = iftLoadFileSetFromDirBySuffix(argv[1], ".mimg", 1);
    int nimages     = fs->n;

    iftMImage *activ = iftReadMImage(fs->files[0]->path);
    int nwords       = activ->m; 
    iftDataSet *Z    = iftCreateDataSet(nimages,nwords);
    iftDestroyMImage(&activ);
    
    for (int i=0; i < nimages; i++){
      printf("Processing image %d of %d\r",i+1,nimages);
      
      activ  = iftReadMImage(fs->files[i]->path);
      Z->sample[i].truelabel = fs->files[i]->label;
      for (int p=0; p < activ->n; p++) {
	int bmax=0;
	for (int b=1; b < activ->m; b++) {
	  if (activ->val[p][b] > activ->val[p][bmax])
	    bmax = b;
	}
	Z->sample[i].feat[bmax] += activ->val[p][bmax];
      }
      iftDestroyMImage(&activ);
    }
    Z->nclasses =  iftCountNumberOfClassesDataSet(Z);
    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);
    iftWriteDataSet(Z,argv[2]);

    iftDestroyDataSet(&Z);
    iftDestroyFileSet(&fs);
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
