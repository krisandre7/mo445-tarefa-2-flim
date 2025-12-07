#include "ift.h"



int main(int argc, char **argv) 
{
  iftImage       *img,   *orig;
  iftMImage      *input, *output;
  char            filename[200];
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
    iftError("Usage: iftTestConvNetwork <image.[ppm,pgm]> <parameters.convnet> <markers.txt (optional)>","main");

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

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  iftRandomSeed(IFT_RANDOM_SEED);

  if (strcmp(ext,"convnet")==0){
    convnet = iftReadConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  /* Compute kernels by unsupervised learning */

  t1     = iftTic(); 
  input  = iftImageToMImage(orig,WEIGHTED_YCbCr_CSPACE);
  iftUnsupLearnKernels(input,convnet,5000,0.01,1);
  
  t2     = iftToc();
  fprintf(stdout,"Kernels learned in %f ms\n",iftCompTime(t1,t2));

  /* Apply convolutional neural network */

  t1     = iftTic();
  output = iftApplyConvNetwork(input,convnet);  
  t2     = iftToc();
  fprintf(stdout,"Convolution network executed in %f ms\n",iftCompTime(t1,t2));


  //  convnet->with_weights = 1;
  //  iftWriteConvNetwork(convnet,"teste1.convnet");  

  /* Enhance edges */
  
  if (argc==4){

    /* Compute watershed and watergray with CNN descriptors */

    t1     = iftTic();
    iftLabeledSet *S=iftReadSeeds2D(argv[3],orig);
    iftLabeledSet *M=iftMAdjustSeedCoordinates(S,input,output);
    iftAdjRel     *A=iftCircular(sqrtf(2.0));
    iftImage *basins = iftMEnhanceEdges(output,A,M,0.5);  
    iftWriteImageP2(basins,"basinsByCNN.pgm");
    iftImage *label  = iftRelaxedWatershed(basins,A,M,5,0.5);
    iftWriteImageP2(label,"labelByCNN.pgm");
    int         size = (input->xsize - output->xsize)/2;
    img              = iftRemFrame(orig,size);
    iftImage *segm   = iftMask(img,label);
    t2     = iftToc();
    fprintf(stdout,"Watershed computed in %f ms\n",iftCompTime(t1,t2));
    iftWriteImageP6(segm,"WatershedByCNN.ppm");
    iftDestroyImage(&label);
    iftDestroyImage(&segm);
    iftDestroyLabeledSet(&S);
    iftDestroyLabeledSet(&M);

    t1     = iftTic();
    iftImage *marker = iftVolumeClose(basins,1000,NULL);
    label            = iftWaterGray(basins,marker,A);
    iftColor RGB, YCbCr;
    RGB.val[0] = 255; RGB.val[1] = 0; RGB.val[2] = 0;
    YCbCr            = iftRGBtoYCbCr(RGB,255);
    iftAdjRel *B     = iftCircular(0.0);
    iftDrawBorders(img,label,A,YCbCr,B);
    t2     = iftToc();
    fprintf(stdout,"Watergray computed in %f ms\n",iftCompTime(t1,t2));
    iftWriteImageP6(img,"WatergrayByCNN.ppm");
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImage(&img);
    iftDestroyImage(&label);
    iftDestroyAdjRel(&B);
    iftDestroyAdjRel(&A);

    /* Compute watershed and watergray with color descriptors */

    t1     = iftTic();
    S=iftReadSeeds2D(argv[3],orig);
    A=iftCircular(sqrtf(2.0));
    basins = iftEnhanceEdges(orig,A,S,0.5);  
    iftWriteImageP2(basins,"basins.pgm");
    label  = iftRelaxedWatershed(basins,A,S,5,0.5);
    iftWriteImageP2(label,"label.pgm");
    segm   = iftMask(orig,label);
    t2     = iftToc();
    fprintf(stdout,"Watershed computed in %f ms\n",iftCompTime(t1,t2));
    iftWriteImageP6(segm,"Watershed.ppm");
    iftDestroyImage(&label);
    iftDestroyImage(&segm);
    iftDestroyLabeledSet(&S);

    t1     = iftTic();
    marker = iftVolumeClose(basins,1000,NULL);
    label  = iftWaterGray(basins,marker,A);
    RGB.val[0] = 255; RGB.val[1] = 0; RGB.val[2] = 0;
    YCbCr = iftRGBtoYCbCr(RGB,255);
    B     = iftCircular(0.0);
    img   = iftCopyImage(orig);
    iftDrawBorders(img,label,A,YCbCr,B);
    t2     = iftToc();
    fprintf(stdout,"Watergray computed in %f ms\n",iftCompTime(t1,t2));
    iftWriteImageP6(img,"Watergray.ppm");
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImage(&label);
    iftDestroyImage(&img);
    iftDestroyAdjRel(&B);
    iftDestroyAdjRel(&A);

  } else {

    /* Compute watergray with CNN and color descriptors */

    t1     = iftTic();
    iftAdjRel     *A=iftCircular(sqrtf(2.0));
    iftImage *basins = iftMImageBasins(output,A);  
    iftWriteImageP2(basins,"basinsByCNN.pgm");
    iftImage *marker = iftVolumeClose(basins,1000,NULL);
    iftImage *label  = iftWaterGray(basins,marker,A);
    iftColor RGB, YCbCr;
    RGB.val[0] = 255; RGB.val[1] = 0; RGB.val[2] = 0;
    YCbCr            = iftRGBtoYCbCr(RGB,255);
    iftAdjRel *B     = iftCircular(0.0);
    int         size = (input->xsize - output->xsize)/2;
    img              = iftRemFrame(orig,size);
    iftDrawBorders(img,label,A,YCbCr,B);
    t2     = iftToc();
    fprintf(stdout,"Watergray computed in %f ms\n",iftCompTime(t1,t2));
    iftWriteImageP6(img,"WatergrayByCNN.ppm");
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImage(&label);
    iftDestroyImage(&img);
    iftDestroyAdjRel(&B);
    iftDestroyAdjRel(&A);

    t1     = iftTic();
    A=iftCircular(sqrtf(2.0));
    basins = iftImageBasins(orig,A);
    iftWriteImageP2(basins,"basins.pgm");
    marker = iftVolumeClose(basins,1000,NULL);
    label  = iftWaterGray(basins,marker,A);
    RGB.val[0] = 255; RGB.val[1] = 0; RGB.val[2] = 0;
    YCbCr = iftRGBtoYCbCr(RGB,255);
    B     = iftCircular(0.0);
    img   = iftCopyImage(orig);
    iftDrawBorders(img,label,A,YCbCr,B);
    t2     = iftToc();
    fprintf(stdout,"Watergray computed in %f ms\n",iftCompTime(t1,t2));
    iftWriteImageP6(img,"Watergray.ppm");
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImage(&label);
    iftDestroyImage(&img);
    iftDestroyAdjRel(&B);
    iftDestroyAdjRel(&A);
  }

  /* Write features */

  for (int i=0; i < convnet->nkernels[convnet->nlayers-1]; i++) {
    img = iftMImageToImage(output,255,i);
    if (img != NULL){
      sprintf(filename,"feat%d.pgm",i);
      iftWriteImageP5(img,filename);
      iftDestroyImage(&img);    
    }
  }

  iftDestroyMImage(&input);
  iftDestroyMImage(&output);
  iftDestroyImage(&orig);
  iftDestroyConvNetwork(&convnet);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);

}
