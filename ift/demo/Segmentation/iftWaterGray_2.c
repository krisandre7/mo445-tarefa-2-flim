#include "ift.h"


int main(int argc, char *argv[])
{
  iftImage        *img=NULL,*basins;
  iftImage        *marker=NULL;
  iftImageForest  *fst=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  timer           *t1=NULL,*t2=NULL;
  int norm_value;

    if (argc<4)
    iftError("Usage: iftWaterGray <image.ppm(pgm,ppm)> <spatial_radius(ex. 3.0)> <volume thresh> [<label_image>]",
             "main");

  img   = iftReadImageByExt(argv[1]);

  t1 = iftTic();

  if (iftIs3DImage(img)){
    A = iftSpheric(atof(argv[2]));
    basins=iftImageBasins(img,A);
    iftWriteImage(basins,"basins.scn");
  }
  else{
    A = iftCircular(atof(argv[2]));
    iftImage *filtered = iftMedianFilter(img, A);
    basins = iftImageBasins(filtered, A);
    iftWriteImageP2(basins, "basins.pgm");
    iftDestroyImage(&filtered);
  }

  fst = iftCreateImageForest(basins, A);
  marker = iftVolumeClose(basins,atof(argv[3]));
  iftWaterGrayForest(fst,marker);

  t2     = iftToc();

  fprintf(stdout,"%.2f %d\n",iftCompTime(t1,t2),iftMaximumValue(fst->label));

  if (argc == 5){
    iftImage *border = iftBorderImage(fst->label,0);
    iftImage *aux;
    iftImage *gt_img=iftReadImageByExt(argv[4]);
    aux=iftRelabelGrayScaleImage(gt_img,0);
    iftDestroyImage(&gt_img);
    gt_img=aux;
    iftImage *gt_borders=iftBorderImage(gt_img,0);
    printf("br -> %.4f\n",iftBoundaryRecall(gt_borders, border, 2.0));
    printf("ue -> %.4f\n",iftUnderSegmentation(gt_img, fst->label));

    iftDestroyImage(&gt_img);
    iftDestroyImage(&gt_borders);
    iftDestroyImage(&border);
  }

  iftDestroyAdjRel(&A);

  if (iftIs3DImage(img)){
    iftWriteImageByExt(fst->label,"labels.scn");
  }
  else{
    iftWriteImageP2(fst->label,"labels.pgm");

    norm_value = iftNormalizationValue(iftMaximumValue(img));
    RGB.val[0] = 0;
    RGB.val[1] = norm_value;
    RGB.val[2] = norm_value;
    YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
    A          = iftCircular(sqrtf(2.0));
    B          = iftCircular(0.0);
    iftDrawBorders(img,fst->label,A,YCbCr,B);
    iftWriteImageByExt(img,"regions.ppm");

    RGB.val[0] = 255;
    RGB.val[1] = 255;
    RGB.val[2] = 255;
    YCbCr      = iftRGBtoYCbCr(RGB,255);
    iftSetImage(img, 0);
    iftDestroyAdjRel(&A);
    A      = iftCircular(atof(argv[2]));
    iftDrawBorders(img,fst->label,A,YCbCr,B);

    iftWriteImageP2(img,"borders.pgm");
    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
  }

  iftDestroyImage(&img);
  iftDestroyImage(&marker);
  iftDestroyImage(&basins);
  iftDestroyImageForest(&fst);

   return(0);
}

