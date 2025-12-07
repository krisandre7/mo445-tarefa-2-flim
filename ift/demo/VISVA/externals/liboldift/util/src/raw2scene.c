/* ESTOU ASSUMINDO 16 BITS */

#include "common.h"
#include "scene.h"

int main(int argc, char **argv)
{
  FILE *fp=NULL; 
  char filename[200];
  int  xsize,ysize,zsize,i,n,byte1,byte2;
  ushort *data16;
  Scene *scn;

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

  if (argc != 9) {
    fprintf(stderr,"usage: raw2scene <basename> <width> <height> <nslices> <dx> <dy> <dz> <originated in: type 0 for msdos or 1 for unix> \n");
    exit(-1);
  }
  xsize = atoi(argv[2]);
  ysize = atoi(argv[3]);
  zsize = atoi(argv[4]);
  sprintf(filename,"%s.raw",argv[1]);
  fp = fopen(filename,"rb");
  if (fp == NULL) {
    Error(MSG2,"raw2scene");
  }
  scn   = CreateScene(xsize,ysize,zsize);
  scn->dx = atof(argv[5]);
  scn->dy = atof(argv[6]);
  scn->dz = atof(argv[7]);
  n = xsize*ysize*zsize;
  if (atoi(argv[8])==0)
    for (i=0; i < n; i++) {
      byte1 = fgetc(fp);
      byte2 = fgetc(fp);
      scn->data[i] = (int)((byte1<<8)+byte2);
    }
  else {
    data16 = AllocUShortArray(n);
    fread(data16,n,sizeof(unsigned short),fp);
    for (i=0; i < n; i++)
      scn->data[i] = (int) data16[i];
  }
  sprintf(filename,"%s.scn",argv[1]);
  WriteScene(scn,filename);
  DestroyScene(&scn);
  fclose(fp);
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
