#include "ift.h"

iftImage* iftWaterGrayIBorder(iftImage* gradient_image,iftImage *label_image,iftAdjRel* A){
  int p,j,q;
  iftVoxel u,v;

  if ( (gradient_image->xsize != label_image->xsize) ||
       (gradient_image->ysize != label_image->ysize) ||
       (gradient_image->zsize != label_image->zsize) )
    iftError("Image dimensions are incompatible","iftWaterGrayBorder");

  for (p=0; p < label_image->n; p++) {
    u.x = iftGetXCoord(label_image,p);
    u.y = iftGetYCoord(label_image,p);
    u.z = iftGetZCoord(label_image,p);
    for (j=0; j < A->n; j++) {
      v.x = u.x + A->dx[j];
      v.y = u.y + A->dy[j];
      v.z = u.z + A->dz[j];
      if (iftValidVoxel(label_image,v)){
	q = iftGetVoxelIndex(label_image,v);
	if (label_image->val[p] < label_image->val[q]){
	  break;
	}
      }
    }
    if (j == A->n)
      gradient_image->val[p] = 0.;
  }

  return gradient_image;
}

iftImage *iftBorderProbImageGT(iftMImage *img,iftImage *GT,int* pnum)
{
  iftAdjRel *A;
  iftImage  *prob;

  if (iftIs3DMImage(img))
    A = iftSpheric(sqrtf(3.0));
  else
    A = iftCircular(sqrtf(2.0));

  prob = iftMImageBasins(img,A);

  (*pnum) = 0;
  for (int p=0; p < prob->n; p++)
    if (GT->val[p] == 0 )
      prob->val[p] = 0;
    else {
      prob->val[p] = 1;
      (*pnum)++;
    }

  iftMaximumValue(prob);
  for (int p=0; p < prob->n; p++)
      prob->val[p] = (int)((float)prob->val[p]/prob->maxval * 100.0);

  iftDestroyAdjRel(&A);

  return(prob);
}


int main(int argc, char **argv) 
{
  iftImage       *img,   *orig,*prob,*GT=NULL;
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

  /* New code for John */

  input  = iftImageToMImage(orig,WEIGHTED_YCbCr_CSPACE);
  int num = 128;
  if (GT == NULL)
    prob   = iftBorderProbImage(input);
  else {
    prob   = iftBorderProbImageGT(input,GT,&num); /* GT - 100% */
    //  if (num <  128) num = 128; // num = min(num,2000)
  }

  //iftUnsupLearnKernels(input,convnet,1000,0.01,1);
  iftUnsupLearnKernelsByKmedoids(input,convnet,num,128,1); // no more using prob - falcão changed on 2013/11/10
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

  iftAdjRel *A = iftCircular(sqrtf(2.0));
  t1     = iftTic();
  img = iftMImageBasins(output, A);
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
  iftWaterGrayIBorder(img,label,A);
  iftWriteImageP2(img,"basins.thin.pgm");

  iftDestroyImage(&marker);
  iftDestroyImage(&label);
  
  iftDestroyImage(&orig);
  iftDestroyImage(&img);
  iftDestroyAdjRel(&A);

  /* end of John's code */

  if (GT != NULL)
    iftDestroyImage(&GT);
  iftDestroyMImage(&input);
  iftDestroyMImage(&output);
  iftDestroyConvNetwork(&convnet);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
