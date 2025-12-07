#include "common.h"
#include "scene.h"

int main(int argc, char **argv)
{
  FILE *fp=NULL; 
  char filename[200];
  int i,j,n,first,last,xsize,ysize,zsize;
  Scene *scn;
  ushort *data16;
  int *data32;

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

  if (argc != 7) {
    fprintf(stderr,"usage: rawslice2scene <basename> <first> <last> <dx> <dy> <dz>\n");
    exit(-1);
  }
  first = atoi(argv[2]);
  last  = atoi(argv[3]);
  sprintf(filename,"%s.%3d",argv[1],first);
  fp = fopen(filename,"r");
  if (fp == NULL) {
    Error(MSG2,"rawslice2scene");
  }
  fscanf(fp,"%d %d",&xsize,&ysize);
  n     = xsize*ysize;
  zsize = last - first + 1;
  scn   = CreateScene(xsize,ysize,zsize);
  scn->dx = atof(argv[4]);
  scn->dy = atof(argv[5]);
  scn->dz = atof(argv[6]);
  fclose(fp);

  data32 = scn->data;
  data16 = AllocUShortArray(n);
  for (i=first; i <= last; i++) {
    sprintf(filename,"%s.%3d",argv[1],i);
    fp = fopen(filename,"r");
    if (fp == NULL) {
      Error(MSG2,"rawslice2scene");
    }
    fscanf(fp,"%d %d",&xsize,&ysize);    
    fread(data16,n,sizeof(ushort),fp);
    for (j = 0; j < n; j++)
      data32[j] = data16[j];
    data32 = data32 + n;
    fclose(fp);
  }
  free(data16);
  data16 = NULL;

  sprintf(filename,"%s.scn",argv[1]);
  WriteScene(scn,filename);
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
