#include "ift.h"

iftMImage *KernelToMImage(iftMImage *img, iftKernel *K)
{
  int i, p, dmin[3];
  float wsum;
  iftMImage *J=iftCreateMImage(img->xsize,img->ysize,img->zsize,2);
  iftVoxel u;
  
  wsum=0.0;
  dmin[0]=IFT_INFINITY_INT;
  dmin[1]=IFT_INFINITY_INT;
  dmin[2]=IFT_INFINITY_INT;
  
  for (i=0; i < K->A->n; i++) {
    wsum += fabs(K->weight[i]);
    if (K->A->dx[i]<dmin[0])
      dmin[0] = K->A->dx[i];
    if (K->A->dy[i]<dmin[1])
      dmin[1] = K->A->dy[i];
    if (K->A->dz[i]<dmin[2])
      dmin[2] = K->A->dz[i];
  }

  for (i=0; i < K->A->n; i++) {
    K->weight[i] = 255.0*(K->weight[i]/wsum);
  }

  for (i=0; i < K->A->n; i++) {
    u.x = K->A->dx[i]-dmin[0];
    u.y = K->A->dy[i]-dmin[1];
    u.z = K->A->dz[i]-dmin[2];
    p   = iftGetVoxelIndex(img,u);
    J->val[p][0] = iftRound(K->weight[i]);
    J->val[p][1] = 0.0;
  }

  return(J);
}

char *Basename(char *path)
{
  char *basename     = iftBasename(path);
  iftSList *slist    = iftSplitString(basename,"/");
  strcpy(basename,slist->tail->elem);
  iftDestroySList(&slist);
  return(basename);
}

int main(int argc, char *argv[])
{
  timer           *tstart=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  
  if (argc != 2) {
    iftError("iftFFTDemo <image.*>", "main");
  }

  tstart = iftTic();
  
  iftImage  *orig  = iftReadImageByExt(argv[1]);
  iftMImage *mimg  = iftFFTImageToMImage(orig);
  char *basename   = Basename(argv[1]);
  char filename[200];


  /* iftMImage *filt = iftButterworthHighPass(mimg,2,0.8); */
  /* iftImage  *view = iftViewLogMagnitude(filt); */
  /* sprintf(filename,"%s-highpass.png",basename); */
  /* iftWriteImageByExt(view,filename); */
  /* exit(0); */
  
  /* Compute FFT */
  
  iftMImage *spec;
  
  if (iftIs3DMImage(mimg))
    spec = iftFFT3D(mimg);
  else
    spec = iftFFT2D(mimg);

  /* Test viewing magnitude and phase */
  
  iftImage *magnitude = iftViewLogMagnitude(spec);
  iftImage *phase     = iftViewPhase(spec);
  
  if (iftIs3DMImage(mimg)){
    sprintf(filename,"%s-mag.nii.gz",basename);
    iftWriteImageByExt(magnitude,filename);
    sprintf(filename,"%s-phase.nii.gz",basename);
    iftWriteImageByExt(phase,filename);
  } else {
    sprintf(filename,"%s-mag.png",basename);
    iftWriteImageByExt(magnitude,filename);
    sprintf(filename,"%s-phase.png",basename);
    iftWriteImageByExt(phase,filename);
  }
  
  /* Test the inverse FFT */

  iftDestroyMImage(&mimg);
  
  if (iftIs3DMImage(spec)){
    mimg = iftInvFFT3D(spec);
    iftImage *img = iftFFTMImageToImage(mimg,orig);
    sprintf(filename,"%s-inv.nii.gz",basename);
    iftWriteImageByExt(img,filename);
    iftDestroyImage(&img);
  } else {
    mimg = iftInvFFT2D(spec);
    iftImage *img = iftFFTMImageToImage(mimg,orig);
    sprintf(filename,"%s-inv.png",basename);
    iftWriteImageByExt(img,filename);
    iftDestroyImage(&img);
  }

  /* Test filtering by a Sobel kernel */

  if (iftIs3DMImage(mimg)){
    iftKernel *K      = iftSobelXKernel();
    iftMImage  *kernel = KernelToMImage(mimg,K);
    iftDestroyKernel(&K);
    iftMImage *filter = iftFFT3D(kernel);
    iftDestroyMImage(&kernel);
    iftDestroyImage(&magnitude);
    magnitude         = iftViewLogMagnitude(filter);
    sprintf(filename,"%s-xsobel.nii.gz",basename);
    iftWriteImageByExt(magnitude,filename);
    iftMImage *fspec  = iftMultSpectra(spec,filter);
    iftDestroyMImage(&mimg);
    mimg              = iftInvFFT3D(fspec);
    iftImage *img     = iftFFTMImageToImage(mimg, orig);
    sprintf(filename,"%s-filt.nii.gz",basename);
    iftWriteImageByExt(img,filename);
    iftDestroyImage(&img);
    iftDestroyMImage(&filter);
  } else {
    iftKernel *K = iftSobelXKernel2D();
    iftMImage *kernel = KernelToMImage(mimg,K);
    iftDestroyKernel(&K);
    iftMImage *filter = iftFFT2D(kernel);
    iftDestroyMImage(&kernel);
    iftDestroyImage(&magnitude);
    magnitude         = iftViewLogMagnitude(filter);
    sprintf(filename,"%s-xsobel.png",basename);
    iftWriteImageByExt(magnitude,filename);
    iftMImage *fspec  = iftMultSpectra(spec,filter);
    iftDestroyMImage(&mimg);
    mimg              = iftInvFFT2D(fspec);
    iftImage *img     = iftFFTMImageToImage(mimg, orig);
    sprintf(filename,"%s-filt.png",basename);
    iftWriteImageByExt(img,filename);
    iftDestroyImage(&img);
    iftDestroyMImage(&filter);
    iftDestroyMImage(&fspec);
  }
  
  iftFree(basename);
  iftDestroyMImage(&mimg);
  iftDestroyImage(&orig);
  iftDestroyMImage(&spec);
  iftDestroyImage(&magnitude);
  iftDestroyImage(&phase);

  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return 0;
}








