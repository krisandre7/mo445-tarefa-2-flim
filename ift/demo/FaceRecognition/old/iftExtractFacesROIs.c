#include <ift.h>

iftImage *iftEnhanceByIntegralImage(iftImage *img, int xsize, int ysize)
{
  iftFImage *integ=iftIntegralImage(img);
  iftVoxel   u, v[4], Delta;
  int        p, n = xsize*ysize;
  iftImage  *mean=iftCreateImage(img->xsize, img->ysize, img->zsize);

  Delta.z = 0;
  Delta.x = (int)(0.30*img->xsize); 
  Delta.y = (int)(0.10*img->ysize);

  u.z = 0;
  for (u.y=Delta.y+ysize/2; u.y < img->ysize-Delta.y-ysize/2; u.y++) 
    for (u.x=Delta.x+xsize/2; u.x < img->xsize-Delta.x-xsize/2; u.x++){
      p = iftGetVoxelIndex(img,u);
      v[0].x     = u.x-xsize/2; v[0].y = u.y-ysize/2; v[0].z = 0; 
      v[1].x     = u.x+xsize/2; v[1].y = u.y-ysize/2; v[1].z = 0; 
      v[2].x     = u.x-xsize/2; v[2].y = u.y+ysize/2; v[2].z = 0; 
      v[3].x     = u.x+xsize/2; v[3].y = u.y+ysize/2; v[3].z = 0; 
      mean->val[p] = (int)(iftGetIntegralValueInRegion(integ,v,4)/n);
    }

  iftDestroyFImage(&integ);
  
  return(mean);
}

iftSet *iftDetectROIsOnFace(iftImage *img)
{
  iftImage  *enha[2];
  iftImage  *cbas, *bin, *geom;
  int p; 
  iftSet *S=NULL;

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
  geom  = iftMarkGeometricCenters(bin);
  iftDestroyImage(&bin);
  for (p=0; p < img->n; p++) 
    if (geom->val[p] != 0)
      iftInsertSet(&S,p);

  iftDestroyImage(&geom);
  return(S);
}


void iftExtractFacesROIs(int xsize, int ysize)
{
  iftVoxel uo, uf, u;
  FILE *fp[2];
  char trash[100],basename[20],imgname[100],roiname[100],command[200];
  int  nclasses, nsamples, class, sample, i, p, subsample;
  iftImage  *img, *roi;
  iftSet *S=NULL;

  fp[1] = fopen("data/rois/coordinates.txt","w");
  fp[0] = fopen("data/data.txt","r");
  fscanf(fp[0],"%s",trash);
  fscanf(fp[0],"%d",&nclasses);
  for (class=1; class <= nclasses; class++) {
    fscanf(fp[0],"%d",&i);
    if (i != class) 
      iftError("data/data.txt is invalid","iftExtractROI1");
    fscanf(fp[0],"%s",trash);
    fscanf(fp[0],"%d",&nsamples);
    for (sample=1; sample <= nsamples; sample++) {
      fscanf(fp[0],"%d",&i);
      fscanf(fp[0],"%s",basename);
      sprintf(imgname,"data/images/%s.jpg",basename);
      sprintf(command,"convert %s temp.ppm",imgname);
      system(command);
      img = iftReadImageP6("temp.ppm");
      S = iftDetectROIsOnFace(img);
      subsample = 1;
      while (S != NULL) {
	p  = iftRemoveSet(&S);
	u  = iftGetVoxelCoord(img,p);
	uo.x = u.x - xsize/2;
	uo.y = u.y - ysize/2;
	uf.x = uo.x+xsize-1; uf.y = uo.y+ysize-1; uf.z = 0;	
	if ((iftValidVoxel(img,uo))&&(iftValidVoxel(img,uf))){	  
	  roi = iftExtractROI(img,uo,uf);        
	  iftWriteImageP6(roi,"temp.ppm");
	  sprintf(roiname,"data/rois/%s_%03d.jpg",basename,subsample);
	  sprintf(command,"convert temp.ppm %s",roiname);
	  system(command);
	  iftDestroyImage(&roi);
	  fprintf(fp[1],"%s %d %d\n",roiname,u.x,u.y);
	  subsample++;
	}
      }
      iftDestroyImage(&img);
    }
  }
  fclose(fp[0]);
  fclose(fp[1]);
  system("rm -f temp.ppm");
}

int main(int argc, char *argv[]) 
{
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  
  if (argc!=3)
    iftError("Usage: iftExtractFacesROIs <xsize> <ysize>","main");


  t1 = iftTic();

  iftExtractFacesROIs(atoi(argv[1]),atoi(argv[2]));

  t2     = iftToc(); 
  
  fprintf(stdout,"ROIs extracted in %f ms\n",iftCompTime(t1,t2));
  

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




