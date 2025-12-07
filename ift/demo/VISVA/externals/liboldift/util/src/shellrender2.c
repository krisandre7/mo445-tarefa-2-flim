#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ift.h"
#include "comptime.h"

int main(int argc, char **argv) {

  Context *cxt;
  Shell *shell;
  Image *img;
  float ta,tr;
  timer *tic,*toc;
  int i, loop;
  float thx, thy, opac;

  if (argc < 2 || argc > 3) {
    printf("shellrender <labels.scn> <gradient.scn (opcional)>\n");
    exit(1);
  }

  
  // Initial settings
  img = NULL;
  cxt = NULL;
  shell =NULL;
  thy = thx = 0.0;
  loop = 1;

  tr = ta = 0.0;
  srand(1); // for random number generator

  // Read scene

 

  // Create shell
  tic = Tic();
  fprintf(stderr,"Creating shell...");
  shell = ReadShell(argv[1]);
  toc = Toc();
  ta = CTime(tic,toc); 

  fprintf(stderr,"shell: %d voxels\n",shell->nvoxels);

 // Create context
  cxt = NewContext(shell->xsize, shell->ysize, shell->zsize);

  Curve *c = ShellObjHistogram(shell);
 
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
      }
    }
    tic = Tic();
    img = ShellRendering(shell,cxt);
    toc = Toc();
    tr += CTime(tic,toc);   
    WriteImage(img,"result.ppm"); 
    DestroyImage(&img);
    
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




