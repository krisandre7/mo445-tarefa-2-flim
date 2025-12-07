#include "common.h"
#include "scene.h"

int main(int argc, char **argv)
{
  int x,y,z;
  Scene *scn,*oscn;

  /*------- -------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 3) {
    fprintf(stderr,"usage: resliceZY <input scene> <output scene>\n");
    exit(-1);
  }

  /* Supõe entrada coronal, então inverte z e forma o sistema zyx */
  
  scn   = ReadScene(argv[1]);
  oscn  = CreateScene(scn->zsize,scn->ysize,scn->xsize);
  oscn->dx = scn->dz;
  oscn->dy = scn->dy;
  oscn->dz = scn->dx;

  for (z=0; z < scn->zsize; z++) 
    for (y=0; y < scn->ysize; y++) 
      for (x=0; x < scn->xsize; x++) 
	oscn->data[scn->zsize-1-z+oscn->tby[y]+oscn->tbz[x]] = 
	  scn->data[x+scn->tby[y]+scn->tbz[z]];

  WriteScene(oscn,argv[2]);
  DestroyScene(&scn);
  DestroyScene(&oscn);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
