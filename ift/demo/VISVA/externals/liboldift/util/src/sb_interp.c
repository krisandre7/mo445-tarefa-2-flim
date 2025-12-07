#include "common.h"
#include "scene.h"

int main(int argc, char **argv)
{
  char filename[200];
  Scene *scn,*inter;

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

  if (argc != 5) {
    fprintf(stderr,"usage: interp <basename> <dx> <dy> <dz>\n");
    exit(-1);
  }
  sprintf(filename,"%s.scn",argv[1]);
  scn    = ReadScene(filename);
  inter    = ShapeBasedInterp(scn,atof(argv[2]),atof(argv[3]),atof(argv[4]));
  sprintf(filename,"%s_int.scn",argv[1]);
  WriteScene(inter,filename);
  DestroyScene(&scn);
  DestroyScene(&inter);

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
