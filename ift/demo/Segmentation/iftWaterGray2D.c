#include "ift.h"


int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL,*basins;
  iftImage        *marker=NULL;
  iftImageForest  *fst=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  char             ext[10],*pos;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc<4)
    iftError("Usage: iftWaterGray2D <image.[pgm,ppm]> <spatial_radius (e.g. 3.0)> <volume thresh> [basins.[pgm,png]]","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  img   = iftReadImageByExt(argv[1]);    

  
  t1 = iftTic();

  A      = iftCircular(atof(argv[2]));
  B      = iftCircular(1.5);

  if(argc <= 4) {
    iftImage *filtered = iftMedianFilter(img, B);
    basins = iftImageBasins(filtered, A);
    iftWriteImageP2(basins, "basins.pgm");
    iftDestroyImage(&filtered);
  } else {
    basins = iftReadImageByExt(argv[4]);
  }

  fst = iftCreateImageForest(basins, B);
//  marker = iftAddValue(basins,atof(argv[3])*iftMaximumValue(basins));
  marker = iftVolumeClose(basins,atof(argv[3]));
  iftWaterGrayForest(fst,marker);

  t2     = iftToc(); 

  fprintf(stdout,"watergray in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(fst->label));

  iftDestroyAdjRel(&A);
  iftWriteImageByExt(fst->label,"labels.pgm");
  iftWriteImageP2(fst->pred, "pred.pgm");

  RGB.val[0] = 255;
  RGB.val[1] = 255;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  A          = iftCircular(sqrtf(2.0));
  B          = iftCircular(0.0);
  iftDrawBorders(img,fst->label,A,YCbCr,B);
  iftWriteImageByExt(img,"result.ppm");

  RGB.val[0] = 255;
  RGB.val[1] = 255;
  RGB.val[2] = 255;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  iftSetImage(img, 0);
  iftDestroyAdjRel(&A);
  A      = iftCircular(atof(argv[2]));
  iftDrawBorders(img,fst->label,A,YCbCr,B);

  iftWriteImageByExt(img,"borders.pgm");

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
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

