#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL,*basins;
  iftImage        *marker=NULL;
  iftImageForest  *fst = NULL;
  iftAdjRel       *A=NULL;
  char             ext[10],*pos;
  timer           *tstart=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc<4)
    iftError("Usage: iftWaterGray <image.scn> <spatial_radius> <volume_thres> [basins.scn]","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  img   = iftReadImageByExt(argv[1]);    
  
  tstart = iftTic();

  /* the operation is connected for the topology defined by A: A must
     be the same in all operators (including
     iftVolumeClose?). Otherwise, this is not a connected operation in
     the desired topology. */

  A      = iftSpheric(atof(argv[2]));
  if(argc >= 5) {
    basins = iftReadImage(argv[4]);
  } else {
    basins = iftImageBasins(img, A);
  }
  iftWriteImage(basins,"basins.scn");

  fst = iftCreateImageForest(basins, A);

  marker = iftVolumeClose(basins,atof(argv[3]),NULL);
  iftWaterGrayForest(fst,marker);

  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));

  iftWriteImageByExt(fst->label,"result.scn");
  iftWriteImageByExt(fst->pred,"pred.scn");

  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);  
  iftDestroyImage(&marker);  
  iftDestroyImage(&basins);  
  iftDestroyImageForest(&fst);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

