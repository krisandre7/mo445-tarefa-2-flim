#include "ift.h"


iftImage *iftDetectRootPixels(iftImage *img, iftAdjRel *A, float perc, int iter)
{
  iftVoxel u,v;
  int i,I,p,q,n;
  float mean;
  iftImage  *prev  = iftCreateImage(img->xsize,img->ysize,img->zsize);
  iftImage  *next  = iftCreateImage(img->xsize,img->ysize,img->zsize);

  /* For each pixel p, compute the mean value among its adjacent
     pixels. If the intensity of p is below a percentile of the mean
     value, then p is marked as root pixel. */

  for (I=1; I <= iter; I++) {
    
    for (p=0; p < img->n; p++) {

      u.x = iftGetXCoord(img,p);
      u.y = iftGetYCoord(img,p);
      u.z = iftGetZCoord(img,p);

      if (prev->val[p] == 0) {
	mean = 0.0; n=0;
	for (i=1; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (iftValidVoxel(img,v)){
	    q     = iftGetVoxelIndex(img,v);
	    if (prev->val[q]==0){
	      mean += img->val[q];
	      n++;
	    }
	  }
	}
	mean /= n;

	if (img->val[p] < perc*mean){
	  next->val[p] = 255;
	}
      }
    }

    for (p=0; p < img->n; p++) {
      prev->val[p]  = MAX(prev->val[p],next->val[p]);
      next->val[p]  = 0;
    }
  }

  iftDestroyImage(&next);

  return(prev);
}


int main(int argc, char *argv[]) 
{
  iftImage       *orig=NULL, *bin[3];
  iftAdjRel      *A=NULL,*B=NULL;
  char            filename[100],command[150];
  timer          *t1=NULL,*t2=NULL;
  int i,n;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=5)
    iftError("Usage must be: iftDetectRoot <basename> <number of images> <perc [e.g., 0.99]> <iter [e.g., 5]>","main");


  A = iftCircular(3.0);
  B = iftCircular(sqrtf(2.0));
  n = atoi(argv[2]);

  t1     = iftTic();

  for (i=1; i <= n; i++) {
    sprintf(filename,"%s_%02d.tif",argv[1],i);
    sprintf(command,"convert %s temp.pgm",filename);
    system(command);
    orig  = iftReadImageP5("temp.pgm");
    sprintf(command,"rm -f temp.pgm");
    system(command);
    
    bin[0] = iftDetectRootPixels(orig,A,atof(argv[3]),atoi(argv[4]));
    bin[1] = iftDetectRootPixels(orig,A,atof(argv[3])-0.2,1);
    bin[2] = iftInfRec(bin[0],bin[1]);
    iftDestroyImage(&bin[1]);
    bin[1] = iftSelectLargestComp(bin[2],B);
    iftDestroyImage(&bin[0]);
    iftDestroyImage(&bin[2]);
    bin[0] = iftComplement(bin[1]);

    sprintf(filename,"%s_root_%02d.tif",argv[1],i);
    iftWriteImageP5(bin[0],"temp.pgm");
    sprintf(command,"convert temp.pgm %s",filename);
    system(command);
    sprintf(command,"rm -f temp.pgm");
    system(command);
    iftDestroyImage(&orig);  
    iftDestroyImage(&bin[1]);
    iftDestroyImage(&bin[0]);
  }

  t2     = iftToc();
  fprintf(stdout,"Root detection in %f ms\n",iftCompTime(t1,t2));

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



