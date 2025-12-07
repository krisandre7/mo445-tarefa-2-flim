#include "ift.h"

int main(int argc, char **argv) {
  iftImage        *image,  *label_image, *hlabel_image;
  iftMImage       *mimage;
  iftImage        *basins, *marker;
  iftAdjRel       *A[3];
  iftRegionHierarchy *rh;
  iftDataSet      *dataset;
  char             ext[10],*pos;
  timer           *t1=NULL,*t2=NULL;
  
  if ((argc != 7)&&(argc != 6))
    iftError("Usage: iftMergeWatergray input.[ppm,pgm,scn] label.[pgm,scn] adjacency_radius volume_thres number_desired_regions input.mig","main");
  
  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"scn")==0){
    image     = iftReadImage(argv[1]);    
  }else{
    if (strcmp(ext,"ppm")==0){
      image     = iftReadImageP6(argv[1]);    
    }else{
      if (strcmp(ext,"pgm")==0){
	image     = iftReadImageP5(argv[1]);
      }else{
	printf("Invalid image format: %s\n",ext);
	exit(-1);
      }
    }
  }

  if (argc == 7)
    mimage = iftReadMImage(argv[6]);
  else {
    if (strcmp(ext,"scn")==0){
      mimage = iftImageToMImage(image,GRAY_CSPACE);
  }else{
    if (strcmp(ext,"ppm")==0){
      mimage = iftImageToMImage(image,WEIGHTED_YCbCr_CSPACE);      
    }else{
      if (strcmp(ext,"pgm")==0){
	mimage = iftImageToMImage(image,GRAY_CSPACE);
      }else{
	printf("Invalid image format: %s\n",ext);
	exit(-1);
      }
    }
  }
  }

  if ((image->xsize!=mimage->xsize)||
      (image->ysize != mimage->ysize)|| 
      (image->zsize != mimage->zsize)){
    iftMImage *maux;
    float sx = ((float)image->xsize/mimage->xsize);
    float sy = ((float)image->ysize/mimage->ysize);
    float sz = ((float)image->zsize/mimage->zsize);
    if (iftIs3DImage(image)){
      maux = iftMInterp(mimage,sx,sy,sz);
    }else{
      maux = iftMInterp2D(mimage,sx,sy);
    }
    iftDestroyMImage(&mimage);
    mimage = maux;
  }

  if ((image->xsize!=mimage->xsize)||
      (image->ysize != mimage->ysize)|| 
      (image->zsize != mimage->zsize)){
    iftError("Could not resolve rescaling","main");
  }
  if (iftIs3DImage(image)){
    A[0] = iftSpheric(atof(argv[3]));
    A[1] = iftSpheric(1.0);
    A[2] = iftSpheric(0.0);
  }else{
    A[0] = iftCircular(atof(argv[3]));
    A[1] = iftCircular(1.0);
    A[2] = iftCircular(0.0);
  }


 t1 = iftTic();

 /* Watergray */

 basins      = iftImageBasins(image,A[0]);
 marker      = iftVolumeClose(basins, atoi(argv[4]));
 label_image = iftWaterGray(basins,marker,A[0]);
 iftDestroyImage(&basins);
 iftDestroyImage(&marker);
 
 /* hierarchical segmentation */

  dataset      = iftRegionMergingDataSet(mimage, label_image);
  iftDestroyMImage(&mimage);
  rh           = iftCreateRegionHierarchy(dataset, A[1], iftRegionMergingFeat);
  hlabel_image = iftFlattenRegionHierarchy(rh, atoi(argv[5]));

  t2     = iftToc(); 
  fprintf(stdout,"watergray+hierarchical segmentation in %f ms\n",iftCompTime(t1,t2));


  iftDestroyImage(&label_image);
  iftDestroyDataSet(&dataset);
  iftDestroyRegionHierarchy(&rh);

  if (strcmp(ext,"scn")==0){
    iftWriteImage(hlabel_image,argv[2]);
  }else{
    if (strcmp(ext,"ppm")==0){
      iftWriteImageP2(hlabel_image,argv[2]);    
      iftColor rgb_color;
      rgb_color.val[0] = 255; rgb_color.val[1] = 0; rgb_color.val[2] = 0;
      iftColor ycbcr_color = iftRGBtoYCbCr(rgb_color);
      iftImage *hlabel_drawn = iftCopyImage(image);
      iftDrawBorders(hlabel_drawn,hlabel_image, A[1], ycbcr_color, A[2]);
      iftWriteImageP6(hlabel_drawn,"result.ppm");
      iftDestroyImage(&hlabel_drawn);
    }else{
      if (strcmp(ext,"pgm")==0){
	iftWriteImageP2(hlabel_image,argv[2]);    
	iftColor rgb_color;
	rgb_color.val[0] = 255; rgb_color.val[1] = 0; rgb_color.val[2] = 0;
	iftColor ycbcr_color = iftRGBtoYCbCr(rgb_color);
	iftImage *hlabel_drawn = iftCopyImage(image);
	iftDrawBorders(hlabel_drawn,hlabel_image, A[0], ycbcr_color, A[2]);
	iftWriteImageP6(hlabel_drawn,"result.ppm");
	iftDestroyImage(&hlabel_drawn);
      }else{
	printf("Invalid image format: %s\n",ext);
	exit(-1);
      }
    }
  }

  iftDestroyImage(&hlabel_image);
  for (int i=0; i < 3; i++) 
    iftDestroyAdjRel(&A[i]);

  iftDestroyImage(&image);

  return(0);
}
