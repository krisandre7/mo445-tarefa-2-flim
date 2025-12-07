#include "ift.h"


int main(int argc, char *argv[]) 
{
  iftImage            *img=NULL;
  iftImage            *mip=NULL;
  iftPlane            *cutplane=NULL;
  char                 ext[10],*pos;
  timer               *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=7)
    iftError("Usage: iftRootView <image.scn> <theta_x> <theta_y> <xviewsize> <yviewsize> <zviewsize>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"scn")==0){
    img   = iftReadImage(argv[1]);    
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }
  
  t1     = iftTic();
  
  /* Compute Maximum Intensity Projection */

  cutplane = iftCreatePlane();
  iftSetPlanePos(cutplane,img->xsize/2.0,img->ysize/2.0,img->zsize/2.0);
  iftRotatePlane(cutplane,AXIS_X,atof(argv[2]));
  iftRotatePlane(cutplane,AXIS_Y,atof(argv[3]));
  mip = iftProjectMaxValue(img,cutplane,atoi(argv[4]),atoi(argv[5]),atoi(argv[6]));

  t2     = iftToc();

  fprintf(stdout,"Maximum intensity projection in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageP2(mip,"mip.pgm");

  iftDestroyImage(&mip);
  iftDestroyPlane(&cutplane);
  iftDestroyImage(&img);  


  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




