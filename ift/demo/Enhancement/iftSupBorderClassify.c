#include "ift.h"

#include "iftTrainPixels.h"
#include "common.h"

#define _DEBUG

iftImage *iftBorderProbImageGT(iftMImage *img,iftImage *GT,int* pnum)
{
  iftAdjRel *A;
  iftImage  *prob;

  if (iftIs3DMImage(img))
    A = iftSpheric(sqrtf(3.0));
  else
    A = iftCircular(sqrtf(2.0));

  prob = iftMImageBasins(img,A);
  iftImage* probc= iftCopyImage(prob);

  int maxval=0,minval=255;
  (*pnum) = 0;
  for (int p=0; p < prob->n; p++)
    if (GT->val[p] == 0 )
      prob->val[p] = 0;
    else {
      (*pnum)++;
      if (prob->val[p] > maxval)
	maxval = prob->val[p];
      else if  (prob->val[p] < minval)
	minval = prob->val[p];
    }

  int range;
  if (maxval==minval) range = 1;
  else                range = maxval-minval;
  for (int p=0; p < prob->n; p++) {
    if (GT->val[p] != 0 ) {
      prob->val[p] = (int)(((float)(prob->val[p]-minval)/range) * 100.0);
      if (prob->val[p] > 50) {
	(*pnum)--;
	prob->val[p] = 0; 
      } else {
	prob->val[p] = (int)((float)(50.-prob->val[p])/50. * 100.0);
	if (prob->val[p] == 0)
	  (*pnum)--;
      }
    }
  }

  // P2 format normalizes to 255
  // in this case, 100 -> 255, for visualization purposes
  iftWriteImageP2(prob,"prob.pgm"); 

  iftDestroyImage(&probc);
  iftDestroyAdjRel(&A);

  return(prob);
}


int main(int argc, char **argv) 
{
  iftImage       *img,   *orig,*prob,*GT=NULL;
  iftImage       *inputW;
  iftMImage      *input, *output;
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *convnet;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if ((argc!=3)&&(argc!=4))
    iftError("Usage: iftTestConvNetwork <image.[ppm,pgm]> <parameters.convnet> <imageGT.pgm> (optional)>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"ppm")==0){
    orig   = iftReadImageP6(argv[1]);    
  }else{
    if (strcmp(ext,"pgm")==0){
      orig   = iftReadImageP5(argv[1]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  if (argc==4) {
    pos = strrchr(argv[3],'.') + 1;
    sscanf(pos,"%s",ext);
    if (strcmp(ext,"pgm")==0) {
      GT = iftReadImageP5(argv[3]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  iftRandomSeed(IFT_RANDOM_SEED);

  if (strcmp(ext,"convnet")==0){
    convnet = iftReadConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

#ifdef _DEBUG
  int ncolors=12;
  iftColor ColorRGB[ncolors];
  iftColor ColorYCbCr[ncolors];
  ColorRGB[ 0].val[0] = 255; ColorRGB[ 0].val[1] =   0; ColorRGB[ 0].val[2] =   0;
  ColorRGB[ 1].val[0] =   0; ColorRGB[ 1].val[1] = 255; ColorRGB[ 1].val[2] =   0;
  ColorRGB[ 2].val[0] =   0; ColorRGB[ 2].val[1] =   0; ColorRGB[ 2].val[2] = 255;
  ColorRGB[ 3].val[0] =   0; ColorRGB[ 3].val[1] = 255; ColorRGB[ 3].val[2] = 255;
  ColorRGB[ 4].val[0] = 255; ColorRGB[ 4].val[1] =   0; ColorRGB[ 4].val[2] = 255;
  ColorRGB[ 5].val[0] = 255; ColorRGB[ 5].val[1] = 255; ColorRGB[ 5].val[2] =   0;
  ColorRGB[ 6].val[0] =   0; ColorRGB[ 6].val[1] = 128; ColorRGB[ 6].val[2] = 128;
  ColorRGB[ 7].val[0] = 255; ColorRGB[ 7].val[1] = 128; ColorRGB[ 7].val[2] = 128;
  ColorRGB[ 8].val[0] = 128; ColorRGB[ 8].val[1] =   0; ColorRGB[ 8].val[2] = 128;
  ColorRGB[ 9].val[0] = 128; ColorRGB[ 9].val[1] = 255; ColorRGB[ 9].val[2] = 128;
  ColorRGB[10].val[0] = 128; ColorRGB[10].val[1] = 128; ColorRGB[10].val[2] =   0;
  ColorRGB[11].val[0] = 128; ColorRGB[11].val[1] = 128; ColorRGB[11].val[2] = 255;
  for(int c=0;c<ncolors;c++) {
    ColorYCbCr[c] = iftRGBtoYCbCr(ColorRGB[c]);
  }

  iftAdjRel *adjA          = iftCircular(sqrtf(2.0));
  iftAdjRel *adjB          = iftCircular(0.0);
  iftAdjRel *adjC          = iftCircular(3.0);
#endif


  /* New code for John */


  iftWriteImageP6(orig,"origRGB.ppm");  
  // 
  input  = iftImageToMImage(orig,RGB_CSPACE);
  inputW = iftMImageToImage(input,255,0);iftWriteImageP5(inputW,"origRGB-R.pgm");iftDestroyImage(&inputW);
  inputW = iftMImageToImage(input,255,1);iftWriteImageP5(inputW,"origRGB-G.pgm");iftDestroyImage(&inputW);
  inputW = iftMImageToImage(input,255,2);iftWriteImageP5(inputW,"origRGB-B.pgm");iftDestroyImage(&inputW);
  iftDestroyMImage(&input);
  // 
  input  = iftImageToMImage(orig,YCbCr_CSPACE);
  inputW = iftMImageToImage(input,255,0);iftWriteImageP5(inputW,"origYCbCr-Y.pgm" );iftDestroyImage(&inputW);
  inputW = iftMImageToImage(input,255,1);iftWriteImageP5(inputW,"origYCbCr-Cb.pgm");iftDestroyImage(&inputW);
  inputW = iftMImageToImage(input,255,2);iftWriteImageP5(inputW,"origYCbCr-Cr.pgm");iftDestroyImage(&inputW);
  iftDestroyMImage(&input);
  // 
  input  = iftImageToMImage(orig,WEIGHTED_YCbCr_CSPACE);
  inputW = iftMImageToImage(input,255,0); iftWriteImageP5(inputW,"origWYCbCr-Y.pgm" );iftDestroyImage(&inputW);
  inputW = iftMImageToImage(input,255,1); iftWriteImageP5(inputW,"origWYCbCr-Cb.pgm");iftDestroyImage(&inputW);
  inputW = iftMImageToImage(input,255,2); iftWriteImageP5(inputW,"origWYCbCr-Cr.pgm");iftDestroyImage(&inputW);

  int num = 128;
  if (GT == NULL)
    prob   = iftBorderProbImage(input);
  else {
    prob   = iftBorderProbImageGT(input,GT,&num); /* GT - 100% */
    //  if (num <  128) num = 128; // num = min(num,2000)
  }

  fprintf(stderr,"nsamples: %d\n",num);
  //  iftUnsupLearnKernels(input,convnet,num/4,0.4,1);
  iftUnsupLearnKernelsByKmedoids(input,convnet,num/4,4,1);
#ifdef _DEBUG
  iftImage* imgout = iftCopyImage(orig);

  //  iftDrawBorders(imgout,GT,adjA,ColorYCbCr[0],adjB);
  iftVoxel u;
  for(int p=0;p<prob->n;p++) {
    if (prob->val[p] != 0) {
       u.x = iftGetXCoord(imgout,p);
       u.y = iftGetYCoord(imgout,p);
       u.z = iftGetZCoord(imgout,p);
       iftDrawPoint(imgout,u,ColorYCbCr[((prob->val[p]-1)%(ncolors-1))],adjC);
    }
  }

  iftWriteImageP6(imgout,"clustering.ppm");
  iftDestroyImage(&imgout);
#endif // _DEBUG
  iftDestroyImage(&prob);

  /*
  for (int k=0; k < convnet->k_bank[0]->nkernels; k++) {
    printf("kernel %d\n",k);
    float sum=0.0;
    for (int b=0; b < convnet->k_bank[0]->nbands; b++) {
      for (int i=0; i < convnet->k_bank[0]->A->n; i++){
  	printf("%f ",convnet->k_bank[0]->weight[k][b].val[i]);
       sum += convnet->k_bank[0]->weight[k][b].val[i];
      }
    }
    printf("sum = %f\n",sum);

  }
  */
  t1     = iftTic();
  output = iftApplyConvNetwork(input,convnet);  
  t2     = iftToc();
  fprintf(stdout,"Convolution network executed in %f ms\n",iftCompTime(t1,t2));

  // extract X% of samples for training being GT-Border and Non-GTxWatergray-NonBorder.
  iftAdjRel* A = iftCircular(sqrtf(1.5));
  iftLabelPixelsData* pLabelPixel = iftExtractLabelPixelsGTBorderWGNonBorder(output,GT,A,100,0.5,2.0);

#ifdef _DEBUG
  imgout = iftCopyImage(orig);

  iftDrawBorders(imgout,GT,adjA,ColorYCbCr[1],adjB);
  for(int s=0;s<pLabelPixel->n;s++) {
    int p = pLabelPixel->labelPixel[s].p;
    u.x = iftGetXCoord(imgout,p);
    u.y = iftGetYCoord(imgout,p);
    u.z = iftGetZCoord(imgout,p);
    if (pLabelPixel->labelPixel[s].truelabel == 1) // Non-Border
      iftDrawPoint(imgout,u,ColorYCbCr[2],adjC);
    else if (pLabelPixel->labelPixel[s].truelabel == 2) // Border
      iftDrawPoint(imgout,u,ColorYCbCr[0],adjC);
  }

  iftWriteImageP6(imgout,"selected.ppm");
  iftDestroyImage(&imgout);
#endif // _DEBUG
  // Addin pixels to DataSet
  iftDestroyLabelPixelsData(pLabelPixel);


  // Testing the trained classifier in entire image



  //  convnet->with_weights = 1;
  //  iftWriteConvNetwork(convnet,"teste1.convnet");  

  char filename[200];

  for (int i=0; i < convnet->nkernels[convnet->nlayers-1]; i++) {
    img = iftMImageToImage(output,255,i);
    if (img != NULL){
      sprintf(filename,"feat%d.pgm",i);
      iftWriteImageP5(img,filename);
      iftDestroyImage(&img);    
    }
  }

  iftAdjRel *AdjBasins = iftCircular(sqrtf(2.0));
  t1     = iftTic();
  img = iftMImageBasins(output, AdjBasins);
  t2     = iftToc();

  if ( (input->xsize-img->xsize) != (input->ysize-img->ysize) ) {
    char msg[200];
    sprintf(msg,"The difference of height and width between input and processed images are incompatible!\n img(%d,%d):input(%d,%d)\n",img->xsize,img->ysize,input->xsize,input->ysize);
    iftError(msg,"iftTestConvNetwork");
  }

  iftImage* fimg = iftAddFrame(img,(int)((input->xsize-img->xsize)/2),0);
  iftDestroyImage(&img);
  img = fimg;

  if ( (input->xsize != img->xsize) || (input->ysize != img->ysize) ) {
    char msg[200];
    sprintf(msg,"Impossible to correct image dimensions due to rescale (odd numbers are being used)\n img(%d,%d):input(%d,%d)\n",img->xsize,img->ysize,input->xsize,input->ysize);
    iftError(msg,"iftTestConvNetwork");
  }

  fprintf(stdout,"Basins computed in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageP2(img,"basins.grad.pgm");

  iftImage* marker = iftVolumeClose(img,100);
  iftImage* label  = iftWaterGray(img,marker,A);
  img = iftWaterGrayIBorder(img,label,A);

  iftWriteImageP2(img,"basins.thin.pgm");

  iftDestroyImage(&marker);
  iftDestroyImage(&label);

  iftDestroyImage(&orig);
  iftDestroyImage(&img);
  iftDestroyAdjRel(&AdjBasins);

  /* end of John's code */

  if (GT != NULL)
    iftDestroyImage(&GT);
  iftDestroyMImage(&input);
  iftDestroyMImage(&output);
  iftDestroyConvNetwork(&convnet);

  iftDestroyAdjRel(&A);
#ifdef _DEBUG
  iftDestroyAdjRel(&adjA);iftDestroyAdjRel(&adjB);iftDestroyAdjRel(&adjC);
#endif

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
