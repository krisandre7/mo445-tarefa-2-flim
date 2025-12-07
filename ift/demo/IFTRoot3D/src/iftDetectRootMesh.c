#include "ift.h"

/* 
   Author: Alexandre Falcao

   Description: This program reads a sequence of 2D tif images
   containing a root system with a mesh, removes the mesh and outputs
   a sequence of 2D binary tif images with the root pixels for 3D
   reconstruction.

   All images of the input sequence must be named by following the
   template %s_%03d.tif (%s is a basename and %03d is a number with 3
   digits from 001 to 999). The output sequence will be named
   %s_bin_%03d.tif. It also creates another sequence %s_root_%03d.tif
   that illustrates the segmented pixels on the original images.

*/

iftImage *iftDetectRootPixels(iftImage *img, iftImage *mask, iftAdjRel *A, float perc, int niters, int value)
{
  iftVoxel u,v;
  int i,I,p,q,n;
  float mean;
  iftImage  *prev  = iftCreateImage(img->xsize,img->ysize,img->zsize);
  iftImage  *next  = iftCreateImage(img->xsize,img->ysize,img->zsize);

  /* Repeat the following process during a certain number of
     iterations (niters), excluding previously selected pixels: For
     each pixel p, compute the mean value among its adjacent
     pixels. If the intensity of p is above a percentile of the mean
     value, then p is selected (i.e., assign value to the output
     intensity of p). */
  
    for (I=1; I <= niters; I++) {    
      for (p=0; p < img->n; p++) {
	if (mask->val[p]!=0) {
	  u = iftGetVoxelCoord(img,p);
	  if (prev->val[p] == 0) {
	    mean = 0.0; n=0;
	    for (i=1; i < A->n; i++) {
	      v = iftGetAdjacentVoxel(A,u,i);
	      if (iftValidVoxel(img,v)){
		q     = iftGetVoxelIndex(img,v);
		if ((prev->val[q]==0)&&(mask->val[q]!=0)){
		  mean += img->val[q];
		  n++;
		}
	      }
	    }
	    if (n > 0) mean /= n;	
	    if (img->val[p] > perc*mean){
	      next->val[p] = value;
	    }
	  }
	}
      }

      for (p=0; p < img->n; p++) {
	prev->val[p]  = iftMax(prev->val[p],next->val[p]);
	next->val[p]  = 0;
      }
    }

  iftDestroyImage(&next);

  return(prev);
}


int main(int argc, char *argv[]) 
{
  iftImage       *orig=NULL, *bin=NULL;
  iftImage       *aux[2];
  int             T, i, n;
  char            filename[150],command[400];
  iftAdjRel      *A=NULL;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=3)
    iftError("Usage: iftDetectRootMesh <basename> <number of images>","main");

  n = atoi(argv[2]);

  t1     = iftTic();

  for (i=1; i <= n; i++) {

    /* From ppm images, the pipeline requires area filtering at the
       end of the process to eliminate background components from the
       mesh */
    /* sprintf(filename,"%s_%03d.tif",argv[1],i); */
    /* sprintf(command,"convert %s temp.ppm",filename); */
    /* system(command); */
    /* orig  = iftReadImageP6("temp.ppm"); */
    /* sprintf(command,"rm -f temp.ppm"); */
    /* system(command); */

    sprintf(filename,"%s_%03d.tif",argv[1],i);
    sprintf(command,"convert %s temp.pgm",filename);
    system(command);
    orig  = iftReadImageP5("temp.pgm");
    sprintf(command,"rm -f temp.pgm");
    system(command);

    T       = iftOtsu(orig);
    bin     = iftThreshold(orig, T, 255, 255);

    A      = iftCircular(5.0);
    aux[0] = iftDilate(bin,A);
    iftDestroyAdjRel(&A);

    sprintf(filename,"%s_otsu_%03d.tif",argv[1],i);
    iftWriteImageP5(aux[0],"temp.pgm");
    sprintf(command,"convert temp.pgm %s",filename);
    system(command);
    sprintf(command,"rm -f temp.pgm");
    system(command);

    aux[1]  = iftMask(orig, aux[0]);
    iftDestroyImage(&aux[0]);
   
    A      = iftCircular(2.5);   
    aux[0] = iftDilate(bin,A);
    iftDestroyImage(&bin);
    bin    = iftDetectRootPixels(aux[1], aux[0], A, 0.9, 2, 255);
    iftDestroyImage(&aux[1]);
    iftDestroyImage(&aux[0]);
    iftDestroyAdjRel(&A);

    aux[0]  = iftMask(orig, bin);

    sprintf(filename,"%s_root_%03d.tif",argv[1],i);
    iftWriteImageP5(aux[0],"temp.pgm");
    sprintf(command,"convert temp.pgm %s",filename);
    system(command);
    sprintf(command,"rm -f temp.pgm");
    system(command);

    iftDestroyImage(&aux[0]);

    sprintf(filename,"%s_bin_%03d.tif",argv[1],i);
    iftWriteImageP5(bin,"temp.pgm");
    sprintf(command,"convert temp.pgm %s",filename);
    system(command);
    sprintf(command,"rm -f temp.pgm");
    system(command);
     
    iftDestroyImage(&orig);
    iftDestroyImage(&bin);

  }

  t2     = iftToc();
  fprintf(stdout,"Root detection with mesh removal in %f ms\n",iftCompTime(t1,t2));

 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



