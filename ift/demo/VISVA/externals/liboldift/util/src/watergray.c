#include <ift.h>


int main(int argc, char *argv[]) 
{
  timer    *t1=NULL,*t2=NULL;
  CImage   *cimg[2];
  Image    *grad=NULL,*handicap=NULL;
  Image    *label=NULL;  
  char      ext[10],*pos;
  AdjRel   *A=NULL;

  /*--------------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/

  if (argc!=3)
    Error("Usage must be: watergray <image> <filter-parameter>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);


  if (strcmp(ext,"ppm")==0){
    cimg[0] = ReadCImage(argv[1]);
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }
  
  A = Circular(1.5);

  t1 = Tic();
  
  grad     = ColorGradient(cimg[0]);
  handicap = CTVolumeClose(grad,atoi(argv[2]));
  label    = WaterGray(grad,handicap,A);

  t2 = Toc();      
  fprintf(stdout,"watergray in %f ms\n",CTime(t1,t2));
  
  DestroyAdjRel(&A);
  cimg[1] = DrawLabeledRegions(cimg[0]->C[1],label);
  WriteCImage(cimg[1],"watergray-result.ppm");    
  WriteImage(grad,"watergray-grad.pgm");
  WriteImage(handicap,"watergray-handicap.pgm");
  WriteImage(label,"watergray-label.pgm");

  DestroyCImage(&cimg[0]);
  DestroyCImage(&cimg[1]);
  DestroyImage(&grad);
  DestroyImage(&handicap);
  DestroyImage(&label);


  /* ---------------------------------------------------------- */
#ifndef _WIN32
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}




