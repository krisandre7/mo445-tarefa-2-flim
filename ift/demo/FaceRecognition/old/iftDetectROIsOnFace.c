#include <ift.h>



iftImage *iftEnhanceByIntegralImage(iftImage *img, int xsize, int ysize)
{
  iftFImage *integ=iftIntegralImage(img);
  iftVoxel   uo, uf, u_up, u_left, u, Delta;
  int        po, pf, p_up, p_left, p, n = xsize*ysize;
  iftImage  *mean=iftCreateImage(img->xsize, img->ysize, img->zsize);

  Delta.z = 0;
  Delta.x = (int)(0.30*img->xsize); 
  Delta.y = (int)(0.10*img->ysize);

  u.z = uo.z = uf.z = u_up.z = u_left.z = 0;
  for (u.y=Delta.y+ysize/2; u.y < img->ysize-Delta.y-ysize/2; u.y++) 
    for (u.x=Delta.x+xsize/2; u.x < img->xsize-Delta.x-xsize/2; u.x++){
      uo.x     = u.x-xsize/2; uo.y     = u.y-ysize/2; 
      uf.x     = u.x+xsize/2; uf.y     = u.y+ysize/2; 
      u_up.x   = uf.x;      u_up.y    = uo.y;
      u_left.x = uo.x;      u_left.y  = uf.y;
      po       = iftGetVoxelIndex(integ,uo);
      pf       = iftGetVoxelIndex(integ,uf);
      p_up     = iftGetVoxelIndex(integ,u_up);
      p_left   = iftGetVoxelIndex(integ,u_left);
      p        = iftGetVoxelIndex(img,u);
      mean->val[p] = (integ->val[pf] - integ->val[p_up] - integ->val[p_left] + integ->val[po])/n; 
    }

  iftDestroyFImage(&integ);
  
  return(mean);
}


iftImage *iftDetectROIsOnFace(iftImage *img)
{
  iftImage  *enha[2];
  iftImage  *cbas, *bin;

  enha[0] = iftEnhanceWhenCrIsGreaterThanCb(img);
  cbas    = iftCloseBasins(enha[0],NULL,NULL);
  enha[1] = iftSub(cbas,enha[0]);
  iftDestroyImage(&cbas);
  iftDestroyImage(&enha[0]);
  enha[0] = iftEnhanceByIntegralImage(enha[1], 10, 5);
  iftDestroyImage(&enha[1]);
  iftMaximumValue(enha[0]);
  bin  = iftThreshold(enha[0],1,enha[0]->maxval,255);
  iftDestroyImage(&enha[0]);

  return(bin);
}


int main(int argc, char *argv[]) 
{
  timer           *t1=NULL,*t2=NULL;
  char             ext[10],*pos,command[200];
  iftColor         YCbCr, RGB;
  iftAdjRel       *A, *B;
  iftImage        *img[2];

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  
  if (argc!=3)
    iftError("Usage: iftDetectROIsOnFace <input.jpg> <output.ppm>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"jpg")==0){
    sprintf(command,"convert %s temp.ppm",argv[1]);
    system(command);
    img[0] = iftReadImageP6("temp.ppm");
    system("rm -f temp.ppm");
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }

  t1 = iftTic();
  
  img[1] = iftDetectROIsOnFace(img[0]);

  t2     = iftToc(); 
  
  fprintf(stdout,"Face ROIs detected in %f ms\n",iftCompTime(t1,t2));
  
  RGB.val[0]=255; RGB.val[1]=0; RGB.val[2]=0;
  YCbCr = iftRGBtoYCbCr(RGB); 
  B = iftCircular(1.0);
  A = iftCircular(sqrtf(2.0));

  iftDrawBorders(img[0], img[1], A, YCbCr, B);
  iftWriteImageP6(img[0],argv[2]);
  iftDestroyImage(&img[0]);
  iftDestroyImage(&img[1]);
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




