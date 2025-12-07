#include "common.h"
#include "scene.h"

int main(int argc, char **argv)
{
  FILE *fp=NULL; 
  char filename[200];
  unsigned char *data8;
  int i,n,z;
  int Imax;
  Scene *scn;
  int *data32;
  Image *tmp;

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

  if (argc != 2) {
    fprintf(stderr,"usage: scene2pgm <basename>\n");
    exit(-1);
  }
  sprintf(filename,"%s.scn",argv[1]);
  scn  = ReadScene(filename);
  Imax = MaximumValue3(scn);
  n    = scn->xsize*scn->ysize;

  if (Imax < 256) { /* 8 bits */
    data8 = AllocUCharArray(n);
    data32  = scn->data;
    for (z=0; z < scn->zsize; z++) {
      sprintf(filename,"%s.%d",argv[1],z+1);
      fp = fopen(filename,"w");
      if (fp == NULL) {
	Error(MSG2,"scene2pgm");
      }
      for (i=0; i < n; i++) 
	data8[i] = (uchar) data32[i];
      fprintf(fp,"P5\n");
      fprintf(fp,"%d %d\n",scn->xsize,scn->ysize);
      fprintf(fp,"%d\n",Imax);
      fwrite(data8,sizeof(uchar),n,fp);
      fclose(fp);
      data32 = data32 + n;
    }
    free(data8);
  } else { /* 32 bits */
    data32 = scn->data;
    for (z=0; z < scn->zsize; z++) {
      sprintf(filename,"%s.%d",argv[1],z+100);
      fp = fopen(filename,"w");
      if (fp == NULL) {
	Error(MSG2,"scene2pgm");
      }
      tmp    = GetSlice(scn,z);
      Imax   = MaximumValue(tmp);
      DestroyImage(&tmp);
      fprintf(fp,"P2\n");
      fprintf(fp,"%d %d\n",scn->xsize,scn->ysize);
      fprintf(fp,"%d\n",Imax);
      for (i=0; i < n; i++) 
	fprintf(fp,"%d ",(int)data32[i]);
      fclose(fp);
      data32 = data32 + n;
    }    
  }

  sprintf(filename,"%s.info",argv[1]);
  fp = fopen(filename,"w");
  if (fp == NULL) {
    Error(MSG2,"scene2pgm");
  }
  fprintf(fp,"%s\n",argv[1]);
  fprintf(fp,"Voxel size %f %f %f\n",scn->dx,scn->dy,scn->dz);
  fclose(fp);
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
