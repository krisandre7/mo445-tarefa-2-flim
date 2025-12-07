#include "common.h"
#include "scene.h"
#include "segmentation3.h"

int main(int argc, char **argv)
{
  char filename[200];
  Scene *scn,*bin;

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
    fprintf(stderr,"usage: %s <scn_input> <scn_output> <min> <max>\n",argv[0]);
    exit(-1);
  }
  sprintf(filename,"%s",argv[1]);
  scn    = ReadScene(filename);
  bin = Threshold3(scn,atoi(argv[3]),atoi(argv[4])); 
  sprintf(filename,"%s",argv[2]);
  WriteScene(bin,filename);
  DestroyScene(&scn);
  DestroyScene(&bin);

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

