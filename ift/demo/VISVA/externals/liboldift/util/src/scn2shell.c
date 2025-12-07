#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ift.h"
#include "comptime.h"

int main(int argc, char **argv) {

  Scene *bin;
  Context *cxt;
  Shell *shell;
  char *p;
  char file[256];
  char basename[256];
  Image *img=NULL;
  float ta,tr;
  timer *tic,*toc;
  int i,n=0;
  float thx,thy;


  if (argc < 2) {
    printf("Usage: %s <scn basename> ?nframes?\n",argv[0]);
    exit(1);
  } else {
     n = atoi(argv[2]);
  }

  p = strrchr(argv[1],'/');

  if (p !=NULL) {
    p++;
  }  else {
    p = argv[1];
  }

  sprintf(basename,"%s",p);

  sprintf(file,"%s.scn",argv[1]);

  
  // Initial settings

  img = NULL;
  cxt = NULL;
  shell =NULL;
  tr = 0.;
  thx = thy = 0.0;

  srand(1);

  // Read scene
  fprintf(stderr,"Opening scene...\n");

  bin = ReadScene(file);

  // Create shell
  fprintf(stderr,"Creating shell...\n");
  tic = Tic();

  shell = Object2Shell (bin,5);

  toc = Toc();
  ta = CTime(tic,toc); 
  printf("%d voxels. Elapsed time: %f ms\n",shell->nvoxels,ta);

  // Create context
  cxt = CreateContext(bin);

  // Free unused data
  DestroyScene(&bin);

  // Render n times
  if (n != 0) {
    for (i=0;i<n;i++) {
      tic = Tic();
      thx += 360.0*rand()/(RAND_MAX+1.0);
      thy += 360.0*rand()/(RAND_MAX+1.0);
      SetAngles(cxt,thx,thy);
      img = SWShellRendering(shell,cxt); 
      toc = Toc();
      tr += CTime(tic,toc);

      sprintf(file,"%s.%d",basename,i+100);
      fprintf(stderr,"\n");
      fprintf(stderr,"Writing image %s...\n",file);
      fprintf(stderr,"Paxis = %c\n",cxt->PAxis);
      fprintf(stderr,"Angle = (%3.3f,%3.3f)\n",cxt->thx,cxt->thy);
      fprintf(stderr,"Viewer = (%3.3f,%3.3f,%3.3f)\n", -cxt->IR[0][2], -cxt->IR[1][2], -cxt->IR[2][2]);

      WriteImage(img,file);

    }
    printf("Average rendering elapsed time: %f ms\n",tr/(float)n); 
  }

  DestroyImage(&img);
  sprintf(file,"%s.sh",argv[1]); 
  WriteShell(shell,file);
  DestroyShell(&shell);
  DestroyContext(&cxt);
  return 0;
}

