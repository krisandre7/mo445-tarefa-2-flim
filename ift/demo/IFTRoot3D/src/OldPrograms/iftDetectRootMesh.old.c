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
  iftImage       *mask=NULL, *marker=NULL;
  int             i, n, p, T;
  char           filename[150],command[400];
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

  if (argc!=3)
    iftError("Usage: iftDetectRootMesh <basename> <number of images>","main");

  n = atoi(argv[2]);

  t1     = iftTic();

  for (i=1; i <= n; i++) {

    sprintf(filename,"%s_%02d.tif",argv[1],i);
    sprintf(command,"convert %s temp.ppm",filename);
    system(command);
    orig  = iftReadImageP6("temp.ppm");
    sprintf(command,"rm -f temp.ppm");
    system(command);


    green  = iftImageGreen(orig);
    red    = iftImageRed(orig);
    enha  = iftCreateImage(orig->xsize,orig->ysize,orig->zsize);
    for (p=0; p < enha->n; p++){
      if ((green->val[p]!=0)&&(red->val[p]!=0))
	enha->val[p] = (int)(255.0*fabs((float)green->val[p]-(float)red->val[p])/((float)green->val[p]+(float)red->val[p]));
    } 
    iftDestroyImage(&green);
    iftDestroyImage(&red);

    T    = iftOtsu(enha);
    mask = iftThreshold(enha, 0, T, 1);
    aux  = iftMult(orig, mask);

    iftDestroyImage(&mask);
    iftDestroyImage(&enha);
    enha = iftComplement(aux);


    A     = iftCircular(3.0);
    
    mask   = iftDetectRootPixels(enha,A,0.98,1);
    iftDestroyImage(&aux);
    aux    = iftDetectRootPixels(enha,A,0.90,1);
    iftDestroyImage(&enha);
    marker = iftFastAreaOpen(aux, 15);
    iftDestroyImage(&aux);
    aux    = iftInfRec(mask,marker);
    iftDestroyImage(&mask);
    iftDestroyImage(&marker);
    
    iftDestroyAdjRel(&A);
    RGB.val[0] = 255;
    RGB.val[1] = 255;
    RGB.val[2] = 0;
    YCbCr      = iftRGBtoYCbCr(RGB);
    A          = iftCircular(sqrtf(2.0));
    B          = iftCircular(0.0);
    iftDrawBorders(orig,aux,A,YCbCr,B);
    
    enha       = iftComplement(aux);
    iftDestroyImage(&aux);

    sprintf(filename,"%s_root_%02d.tif",argv[1],i);
    iftWriteImageP6(orig,"temp.ppm");
    sprintf(command,"convert temp.ppm %s",filename);
    system(command);
    sprintf(command,"rm -f temp.ppm");
    system(command);

    sprintf(filename,"%s_bin_%02d.tif",argv[1],i);
    iftWriteImageP5(enha,"temp.pgm");
    sprintf(command,"convert temp.pgm %s",filename);
    system(command);
    sprintf(command,"rm -f temp.pgm");
    system(command);

    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    iftDestroyImage(&orig);
    iftDestroyImage(&enha);
  }

  t2     = iftToc();
  fprintf(stdout,"Root detection with mesh removal in %f ms\n",iftCompTime(t1,t2));

 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



