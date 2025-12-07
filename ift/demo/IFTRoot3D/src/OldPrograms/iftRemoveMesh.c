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
  iftImage       *orig=NULL, *green=NULL, *red=NULL;
  iftImage       *aux=NULL,  *enha=NULL;
  iftImage       *mask=NULL;
  int             p, T;
  iftAdjRel      *A=NULL, *B=NULL;
  iftColor        RGB, YCbCr;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage: iftRemoveMesh <filename.ppm>","main");

  orig = iftReadImageP6(argv[1]);

  t1     = iftTic();

  green  = iftImageGreen(orig);
  red    = iftImageRed(orig);

  enha  = iftCreateImage(orig->xsize,orig->ysize,orig->zsize);
  for (p=0; p < enha->n; p++){
    if ((green->val[p]!=0)&&(red->val[p]!=0))
      enha->val[p] = (int)(255.0*fabs((float)red->val[p]-(float)green->val[p])/((float)green->val[p]+(float)red->val[p]));
  } 

  T    = iftOtsu(enha);
  aux  = iftThreshold(enha, 0, T, 1);

  A    = iftCircular(sqrtf(2.0));
  mask = iftDilate(aux,A);  
  iftDestroyAdjRel(&A);
  iftDestroyImage(&aux);
  

  //aux  = iftMult(orig, mask);
  aux  = iftMask(green, mask);
  
  iftDestroyImage(&enha);
  iftDestroyImage(&mask);

  enha = iftComplement(aux);

  A     = iftCircular(5.0);

  mask = iftDetectRootPixels(enha,A,0.92,2);
  iftDestroyImage(&aux);
  iftDestroyImage(&enha);   
  aux = iftCopyImage(mask);//iftAreaOpen(mask, 15);
  
  iftDestroyAdjRel(&A);
  RGB.val[0] = 255;
  RGB.val[1] = 255;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB);  
  A          = iftCircular(sqrtf(2.0));
  B          = iftCircular(0.0);
  iftDrawBorders(orig,aux,A,YCbCr,B);

  iftWriteImageP6(orig,"result.ppm");

  t2     = iftToc();
  fprintf(stdout,"Remove mesh in %f ms\n",iftCompTime(t1,t2));

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  iftDestroyImage(&orig);
  iftDestroyImage(&aux);
  iftDestroyImage(&green);
  iftDestroyImage(&red);
  iftDestroyImage(&mask);
  iftDestroyImage(&enha);

 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



