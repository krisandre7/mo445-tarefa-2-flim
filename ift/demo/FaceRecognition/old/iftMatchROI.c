#include <ift.h>

float iftMatchROI(iftImage *img, iftImage *roi, iftVoxel pos)
{
  iftVoxel uo,uf,u,v,opt_pos;
  float    dist, mindist=FLT_MAX;
  iftImage *roi_img;
  iftFeatures *feat1, *feat2;
  iftAdjRel *A=iftCircular(sqrtf(2.0));
  int p,q;

  
  if ((roi->xsize > img->xsize)||
      (roi->ysize > img->ysize)||
      (roi->zsize > img->zsize))
    iftError("Invalid ROI","iftMatchROI");

  feat1 = iftExtractGradient(roi,A);

  opt_pos.x=opt_pos.y=0;
  opt_pos.z=uf.z=uo.z=u.z=0;
    for (u.y=pos.y-roi->ysize/4; u.y < pos.y+roi->ysize/4; u.y++) 
      for (u.x=pos.x-roi->xsize/4; u.x < pos.x+roi->xsize/4; u.x++){	
	uo.x = u.x-roi->xsize/2; uo.y = u.y-roi->ysize/2;
	uf.x = uo.x + roi->xsize - 1;
	uf.y = uo.y + roi->ysize - 1;
	if (iftValidVoxel(img,uo)&&iftValidVoxel(img,uf)){
	  roi_img = iftExtractROI(img,uo,uf);
	  feat2   = iftExtractGradient(roi_img,A);
	  dist    = iftFeatDistL1(feat1,feat2);
	  if (dist < mindist){ 
	    mindist = dist;
	    opt_pos.x = u.x; opt_pos.y = u.y;
	  }
	  iftDestroyFeatures(&feat2);
	  iftDestroyImage(&roi_img);
	}
      }
    iftDestroyAdjRel(&A);
    iftDestroyFeatures(&feat1);
    if (mindist != FLT_MAX){
      printf("(%d,%d)\n",opt_pos.x,opt_pos.y);
      iftImage *tmp=iftCopyImage(img);
      u.z=v.z=0;
      for (u.y=0; u.y < roi->ysize; u.y++) 
	for (u.x=0; u.x < roi->xsize; u.x++){	
	  p = iftGetVoxelIndex(roi,u);
	  v.x = u.x + opt_pos.x - roi->xsize/2;
	  v.y = u.y + opt_pos.y - roi->ysize/2;
	  q   = iftGetVoxelIndex(tmp,v);
	  tmp->val[q] = roi->val[p];
	}
      iftWriteImageP5(tmp,"testmatch.pgm");
      iftDestroyImage(&tmp);
    }else{
      iftWarning("Could not find best matching","iftMatchROI");
    }

  return(mindist);
}

int main(int argc, char *argv[]) 
{
  char             command[200];
  iftImage        *img[2];
  FILE            *fp;
  char             filename[50];
  iftVoxel         u;
  char             ext[10],*pos;
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
    iftError("Usage: iftMatchROI <image.jpg> <roi.jpg>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"jpg")==0){
    sprintf(command,"convert %s temp.pgm",argv[1]);
    system(command);
    img[0]   = iftReadImageP5("temp.pgm");
    system("rm -f temp.pgm");
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  if (strcmp(ext,"jpg")==0){
    sprintf(command,"convert %s temp.pgm",argv[2]);
    system(command);
    img[1]   = iftReadImageP5("temp.pgm");
    system("rm -f temp.pgm");
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }
  
  fp = fopen("data/rois/coordinates.txt","r");
  sprintf(filename," ");
  u.z = 0;
  while((!feof(fp))&&(strcmp(filename,argv[2])!=0)){
    fscanf(fp,"%s %d %d",filename,&u.x,&u.y);
  }
  if (strcmp(filename," ")!=0)
    printf("It found %s %d %d\n",filename,u.x,u.y);
  else
    iftError("Could not find roi coordinates","main");
  fclose(fp);

  t1 = iftTic();

  
  printf("Distance between %s and %s is %f \n",argv[1],argv[2],iftMatchROI(img[0],img[1],u));

  t2     = iftToc(); 
  
  fprintf(stdout,"Matching distance in %f ms\n",iftCompTime(t1,t2));
  
  iftDestroyImage(&img[0]);
  iftDestroyImage(&img[1]);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




