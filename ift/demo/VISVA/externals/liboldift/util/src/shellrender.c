#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ift.h"
#include "comptime.h"

int main(int argc, char **argv) {

  Scene *scn,*grad;
  Context *cxt;
  Shell *shell;
  char line[256], *tok;
  Image *img;
  CImage *cimg;
  float ta,tr;
  timer *tic,*toc;
  int i,j, loop;
  float thx, thy, opac, color[3] = {1.0,1.0,1.0};

  if (argc < 2 || argc > 3) {
    printf("shellrender <labels.scn> <gradient.scn (opcional)>\n");
    exit(1);
  }

  
  // Initial settings
  img = NULL;
  cxt = NULL;
  scn = NULL;
  cxt = NULL;
  shell =NULL;
  thy = thx = 0.0;
  loop = 1;

  tr = ta = 0.0;
  srand(1); // for random number generator

  // Read scene

  scn = ReadScene(argv[1]);

  // Create context
  cxt = CreateContext(scn);

  // Create shell
  tic = Tic();
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
  toc = Toc();
  ta = CTime(tic,toc); 

  fprintf(stderr,"shell: %d voxels\n",shell->nvoxels);
  fprintf(stderr,"scn: %d voxels\n",scn->xsize*scn->ysize*scn->zsize);

  // Free unused data
  DestroyScene(&scn);
 
  // Render n times
  do {
    printf("Select pitch angle: ");
    scanf("%f", &thx);
    printf("Select tilt angle: ");
    scanf("%f", &thy);
    SetAngles(cxt, thx, thy);
    for (i = 0; i <  c->n; i++) {
      if (c->Y[i] > 0) {
        printf("Label %d: ",i);
        printf(">Select opacity ([0,1]: ");
        scanf("%f", &opac);
        cxt->obj[i].opac = MIN(1.0,MAX(0.0,opac));
        printf(">Select RGB color ([0,1],[0,1],[0,1]): ");
        scanf("%s",line);
        tok = strtok(line," ,");
        j = 0;
        while (tok != NULL && j < 3) {
          color[j++] = MIN(1.0,MAX(0.0,atof(tok)));
          tok = strtok(NULL," ,");
        }
        //fprintf (stderr,"color: [%f, %f, %f]\n", color[0], color[1], color[2]);
        SetObjectColor(cxt,(uchar)i,color[0],color[1],color[2]);
      } 
    }
    tic = Tic();
    cimg = CShellRendering(shell,cxt);
    toc = Toc();
    tr += CTime(tic,toc);   
    WriteCImage(cimg,"result.ppm"); 
    DestroyCImage(&cimg);
    
    printf("Select:\n(0) Exit\n(1) Change options\nCommand: ");
    scanf("%d", &loop);
  } while (loop);
  

  DestroyCurve(&c);
 

 
  // Colorize and write color image
 


  // Print information
  printf("createshell: %f ms\n",ta);
  printf("render: %f ms\n",tr);


  // Destroy remaining data

  DestroyShell(&shell);
  DestroyContext(&cxt);
 
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




