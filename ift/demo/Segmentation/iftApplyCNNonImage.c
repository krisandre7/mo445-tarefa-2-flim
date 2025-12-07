#include "ift.h"

int main(int argc, char **argv) {
  iftImage        *orig;
  iftMImage       *mimg, *aux;
  char             ext[10],*pos;
  iftConvNetwork  *convnet;
  timer           *t1=NULL,*t2=NULL;
  
  if(argc != 4)
    iftError("Usage: iftApplyCNNonImage input.[ppm,pgm,scn] output.mig input.convnet","main");
  
  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);

  iftRandomSeed(IFT_RANDOM_SEED);

  convnet = iftReadConvNetwork(argv[3]);

 t1 = iftTic();

  if (strcmp(ext,"scn")==0){
    orig    = iftReadImage(argv[1]);
    aux     = iftImageToMImage(orig,GRAY_CSPACE);
    mimg    = iftApplyConvNetwork(aux,convnet);
    iftDestroyMImage(&aux);
    iftDestroyImage(&orig);
    iftDestroyConvNetwork(&convnet);
  }else{
    if (strcmp(ext,"ppm")==0){
      orig = iftReadImageP6(argv[1]);    
      aux     = iftImageToMImage(orig,WEIGHTED_YCbCr_CSPACE);
      iftUnsupLearnKernels(aux,convnet,2000,0.01,1);
      mimg = iftApplyConvNetwork(aux,convnet);
      iftDestroyConvNetwork(&convnet);
      iftDestroyMImage(&aux);
      iftDestroyImage(&orig);
    }else{
      if (strcmp(ext,"pgm")==0){
	orig = iftReadImageP5(argv[1]);
	aux  = iftImageToMImage(orig,GRAY_CSPACE);
	mimg = iftApplyConvNetwork(aux,convnet);
	iftDestroyMImage(&aux);
	iftDestroyImage(&orig);
	iftDestroyConvNetwork(&convnet);
      }else{
	printf("Invalid image format: %s\n",ext);
	exit(-1);
      }
    }
  }

  iftWriteMImage(mimg,argv[2]);
  iftDestroyMImage(&mimg);

  t2     = iftToc(); 
  fprintf(stdout,"CNN applied in %f ms\n",iftCompTime(t1,t2));

  return(0);
}
