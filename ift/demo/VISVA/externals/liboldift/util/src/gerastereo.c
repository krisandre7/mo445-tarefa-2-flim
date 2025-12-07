#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ift.h"
#include "comptime.h"

int main(int argc, char **argv) {

  Scene *scn,*grad;
  Context *cxt;
  Shell *shell;
  Image *img;
  CImage *cimg;
  int i, num;
  float thx, thy;
  char filename[200];

  if (argc < 2 || argc > 3) {
    printf("gerastereo <labels.scn> <gradient.scn (opcional)>\n");
    exit(1);
  }

  
  // Initial settings
  img = NULL;
  cxt = NULL;
  scn = NULL;
  cxt = NULL;
  shell =NULL;
  thy = thx = 0.0;

  // Read scene

  scn = ReadScene(argv[1]);

  // Create context
  cxt = CreateContext(scn);

  // Create shell
  fprintf(stderr,"Creating shell...");
  shell = Object2Shell(scn,1); // 1 -> shell thickness
  Curve *c = ShellObjHistogram(shell);
  fprintf(stderr," done. %d objects in shell.\n", shell->nobjs);
  if (argc == 3) {
    grad = ReadScene(argv[2]);
    for (i = 0; i <  c->n; i++) {
      if (c->Y[i] > 0) {
        fprintf(stderr,"Setting gradient for object %d...",i);
        SetObjectNormal(shell,grad,Gradient3,i);
        fprintf(stderr," done.\r");
      }
    }
    printf("\n");
    DestroyScene(&grad);
  }

  fprintf(stderr,"shell: %d voxels\n",shell->nvoxels);
  fprintf(stderr,"scn: %d voxels\n",scn->xsize*scn->ysize*scn->zsize);

  // Free unused data
  DestroyScene(&scn);
 
  // Render n times
  for (i = 0; i <  c->n; i++) {
    if (c->Y[i] > 0) {
      cxt->obj[i].opac = 1.0;
      SetObjectColor(cxt,(uchar)i,1.0,1.0,1.0);
    } 
  }
  
  num=0;
  thx = 90.0;
  for (thy=-90; thy <= 90; thy+=5.0){
      SetAngles(cxt, thx, thy);
      cimg = CShellRenderingStereo(shell,cxt);
      sprintf(filename,"stereoframe%03d.ppm",num);
      WriteCImage(cimg,filename);
      DestroyCImage(&cimg);

      sprintf(filename,"convert stereoframe%03d.ppm stereoframe%03d.gif",num,num);
      system(filename);
      sprintf(filename,"rm stereoframe%03d.ppm",num);
      system(filename);

      SetAngles(cxt, thx, thy-5.0);
      cimg = CShellRendering(shell,cxt);
      sprintf(filename,"leftview%03d.ppm",num);
      WriteCImage(cimg,filename);
      DestroyCImage(&cimg);

      sprintf(filename,"convert leftview%03d.ppm leftview%03d.gif",num,num);
      system(filename);
      sprintf(filename,"rm leftview%03d.ppm",num);
      system(filename);

      SetAngles(cxt, thx, thy+5.0);
      cimg = CShellRendering(shell,cxt);
      sprintf(filename,"rightview%03d.ppm",num);
      WriteCImage(cimg,filename);
      DestroyCImage(&cimg);

      sprintf(filename,"convert rightview%03d.ppm rightview%03d.gif",num,num);
      system(filename);
      sprintf(filename,"rm rightview%03d.ppm",num);
      system(filename);


      num++;
    }
  // Use animate to visualize the movies
  sprintf(filename,"convert -delay 20 -loop 0 stereoframe*.gif stereomovie.gif");
  system(filename);
  sprintf(filename,"convert -delay 20 -loop 0 rightview*.gif rightmovie.gif");
  system(filename);
  sprintf(filename,"convert -delay 20 -loop 0 leftview*.gif leftmovie.gif");
  system(filename);


  // Destroy remaining data

  DestroyShell(&shell);
  DestroyContext(&cxt);
  DestroyCurve(&c);
 
  return 0;
}

// Copy and paste


    /*    if (i%10 == 0) {
      sprintf(file,"test.%d",i+100);
      WriteImage(img,file);
      }*/


  /*
  DestroyScene(&scn);  
  sprintf(file,"../../data/3D/%s.lab",argv[1]);
  scn = ReadScene(file);
  */

    /*SetAngles(cxt,360.0*rand()/(RAND_MAX+1.0), 360.0*rand()/(RAND_MAX+1.0),\
      360.0*rand()/(RAND_MAX+1.0));	      */




