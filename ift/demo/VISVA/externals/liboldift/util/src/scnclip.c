#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "scene.h"

/* faz um clip ortogonal da cena e grava em clip.scn */

int main(int argc, char **argv) {
  Scene *src, *dest;
  int i,j,k, a,b,c, lim[6], xs,ys,zs;

  if (argc!=8) {
    fprintf(stderr,"scnclip arquivo x0 x1 y0 y1 z0 z1\n\n");
    return 1;
  }

  src = ReadScene(argv[1]);
  if (!src) {
    fprintf(stderr,"unable to read %s\n",argv[1]);
    return 2;
  }

  for(i=2;i<8;i++)
    lim[i-2] = atoi(argv[i]);

  if (lim[0] < 0) lim[0] = 0;
  if (lim[2] < 0) lim[2] = 0;
  if (lim[4] < 0) lim[4] = 0;

  if (lim[1] >= src->xsize) lim[1] = src->xsize-1;
  if (lim[3] >= src->ysize) lim[3] = src->ysize-1;
  if (lim[5] >= src->zsize) lim[5] = src->zsize-1;

  xs = lim[1] - lim[0] + 1;
  ys = lim[3] - lim[2] + 1;
  zs = lim[5] - lim[4] + 1;

  dest=CreateScene(xs,ys,zs);
  if (!dest) {
    fprintf(stderr,"unable to create scene.\n");
    return 2;
  }

  dest->dx = src->dx;
  dest->dy = src->dy;
  dest->dz = src->dz;

  for(i=0;i<zs;i++)
    for(j=0;j<ys;j++)
      for(k=0;k<xs;k++) {
	a = lim[0]+k;
	b = lim[2]+j;
	c = lim[4]+i;
	dest->data[k+dest->tby[j]+dest->tbz[i]] = 
	  src->data[a+src->tby[b]+src->tbz[c]];
      }

  WriteScene(dest,"clip.scn");
  DestroyScene(&dest);
  DestroyScene(&src);
  return 0;
}
