#include "ift.h"


int main(int argc, const char **argv) {
  iftImage *img = NULL, *roi = NULL;
  iftBoundingBox bb;

  if ((argc != 9)&&(argc != 2)) {
    iftError("iftRegionOfInterest <input-image.*> <xmin> <ymin> <zmin> <xmax> <ymax> <zmax> <output-roi.*> or iftRegionOfInterest <input-image.*>","iftRegionOfInterest");
  }

  img = iftReadImageByExt(argv[1]);

  if (argc == 2){    
    printf("xsize %d ysize %d zsize %d dx %.2f dy %.2f dz %.2f\n",img->xsize,img->ysize,img->zsize,img->dx,img->dy,img->dz);
    iftDestroyImage(&img);
    return(0);
  }
    
  bb.begin.x = atoi(argv[2]); 
  bb.begin.y = atoi(argv[3]);
  bb.begin.z = atoi(argv[4]);
  bb.end.x   = atoi(argv[5]);
  bb.end.y   = atoi(argv[6]);
  bb.end.z   = atoi(argv[7]);
  roi = iftExtractROI(img,bb); 

  iftWriteImageByExt(roi,argv[8]);

  iftDestroyImage(&img);
  iftDestroyImage(&roi);

  return 0;
}
