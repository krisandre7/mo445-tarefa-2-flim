#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL,*label=NULL,*domes=NULL;
  iftImage        *marker=NULL;
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


  if (argc!=4)
    iftError("Usage: iftDualWaterGray2D <image.ppm> <spatial_radius> <volume_thres>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"ppm")==0){
    img   = iftReadImageP6(argv[1]);    
  }else{
    if (strcmp(ext,"pgm")==0){
      img   = iftReadImageP5(argv[1]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  t1 = iftTic();

  /* the operation is connected for the topology defined by A: A must
     be the same in all operators (including iftVolumeOpen?). */
  A      = iftCircular(atof(argv[2]));
  domes  = iftImageDomes(img,A);
  marker = iftVolumeOpen(domes,atof(argv[3]));
  label  = iftDualWaterGray(domes,marker,A);

  t2     = iftToc(); 

  fprintf(stdout,"dual-watergray in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(label));

  iftDestroyAdjRel(&A);

  iftWriteImageP2(label,"labels.pgm");
  iftWriteImageP2(domes,"domes.pgm");

  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB);
  A          = iftCircular(sqrtf(2.0));
  B          = iftCircular(0.0);
  iftDrawBorders(img,label,A,YCbCr,B);
  iftWriteImageP6(img,"result.ppm");

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  iftDestroyImage(&img);  
  iftDestroyImage(&marker);  
  iftDestroyImage(&domes);  
  iftDestroyImage(&label);  

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

