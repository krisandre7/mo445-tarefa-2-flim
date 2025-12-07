#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "scene.h"

int main(int argc, char **argv) {
  Scene *scn;  
  FILE *f;
  int i,ecount=0, scount=0,bpv;
  float a,b,c;
  int x,y,z;

  if (argc<2) {
    fprintf(stderr,"usage: scninfo arq1 [arq2 ...]\n");
    return 1;
  }

  for(i=1;i<argc;i++) {    
    scn=ReadScene(argv[i]);
    if (!scn) {
      fprintf(stderr,"cannot read %s\n",argv[i]);
      ++ecount;
      continue;
    } else {
      ++scount;
    }

    printf("%s:\n",argv[i]);
    printf("volume size     : W: %d H: %d D: %d\n",
	   scn->xsize,scn->ysize,scn->zsize);
    printf("number of voxels: %d\n",scn->xsize*scn->ysize*scn->zsize);
    printf("voxel size      : X: %.4f Y: %.4f Z: %.4f\n",
	   scn->dx, scn->dy, scn->dz);
    printf("minimum         : %d\n",MinimumValue3(scn));
    printf("maximum         : %d\n",MaximumValue3(scn));

    f=fopen(argv[i],"r");
    if (fscanf(f,"SCN \n %d %d %d \n %f %f %f \n %d \n",
	       &x,&y,&z,&a,&b,&c,&bpv)!=7) {
      bpv = -1;
    }

    printf("bits per voxel : %d\n\n",bpv);
    DestroyScene(&scn);
  }

  if (scount) return 0; else return 1;
}
