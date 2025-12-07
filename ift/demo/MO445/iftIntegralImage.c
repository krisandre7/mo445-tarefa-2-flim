#include "ift.h"

/* Alexandre Falcao

   Given an original image and a feature image that enhances an object
   of interest in the original image, this code demonstrates the
   computation of the integral image and the cummulative-value image
   using a region Wx x Wy around a pixel v obtained from a
   displacement (Dx,Dy) from each pixel u of the feature image. The
   integral image accelerates this process. Haar-like features may be
   obtained from mathematical operations on multiple cummulative-value
   images.

*/

int main(int argc, char *argv[])
{
  
  /* Example:
     iftIntegralImage orig_image.png feat_image.png 107 1 0 0 0.9 out.png
  */

  if (argc != 9){
    printf("Usage: iftIntegralImage <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8>\n");
    printf("P1: original image with an object of interest (e.g., plate)\n");
    printf("P2: a feature image (e.g., vertical edges of the plate image)\n");
    printf("P3: size of the region in x for cummulative value computation\n");
    printf("P4: size of the region in y for cummulative value computation\n");
    printf("P5: displacement in x for cummulative value computation\n");
    printf("P6: displacement in y for cummulative value computation\n");
    printf("P7: threshold percentage of the maximum cummulative value for object detection\n");
    printf("P8: output image with the detected objects\n");
    exit(-1);
  }

  iftImage  *orig = iftReadImageByExt(argv[1]);
  iftImage  *feat = iftReadImageByExt(argv[2]);
  int Wx          = atoi(argv[3]);
  int Wy          = atoi(argv[4]);
  int Dx          = atoi(argv[5]);
  int Dy          = atoi(argv[6]);
  float thres     = atof(argv[7]);
  char *outname   = argv[8];
    
  iftFImage *integral_image = iftIntegralImage(feat);
  iftDestroyImage(&feat);
  iftFImage *cum_image      = iftCumValueInRegion(integral_image, Dx, Dy, 0,
						  Wx, Wy, 0);
  iftDestroyFImage(&integral_image);  
  iftImage *img  = iftFImageToImage(cum_image,65535);	 
  iftDestroyFImage(&cum_image);
  
  iftImage *bin  = iftThreshold(img,thres*iftMaximumValue(img),
				65535,255);
  iftAdjRel *A   = iftCircular(3.0);
  iftColor RGB, YCbCr;
  RGB.val[0] = 255;  RGB.val[1] = 0;  RGB.val[2] = 0;
  YCbCr = iftRGBtoYCbCr(RGB,255);  
  iftDrawBorders(orig, bin, A, YCbCr, A);
  iftWriteImageByExt(orig,outname);
  iftDestroyImage(&img);
  iftDestroyImage(&bin);
  iftDestroyImage(&orig);
  iftDestroyAdjRel(&A);  
  
  return(0);
}




