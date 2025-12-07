#include "common.h"
#include "scene.h"

int main(int argc, char **argv)
{
  int x,y,z,xsize,ysize,zsize;
  Scene *scn;

  /*------- -------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/
  
  if (argc != 4) {
    fprintf(stderr,"usage: cube <xsize> <ysize> <zsize>\n");
    exit(-1);
  }
  xsize = atoi(argv[1]);
  ysize = atoi(argv[2]);
  zsize = atoi(argv[3]);

  scn   = CreateScene(xsize,ysize,zsize);
  for (z=zsize/4; z < (3*zsize/4); z++) 
    for (y=ysize/4; y < (3*ysize/4); y++) 
      for (x=xsize/4; x < (3*xsize/4); x++) 
	scn->data[x + scn->tby[y] + scn->tbz[z]] = 255;


  WriteScene(scn,"cube.scn");
  DestroyScene(&scn);

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
