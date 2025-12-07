#include "ift.h"

/* Data Structures for iftTrainPixels */

typedef struct ift_label_pixel {
  int	  class;
  int 	  p;
} iftLabelPixel;

typedef struct ift_train_pixels_data {
  int          nfilename;
  int          n; /* number of pixels */
  char         *filename;
  iftMImage    *image;
  iftLabelPixel *labelPixel;
} iftLabelPixelsData;

typedef struct ift_train_pixels {
  int                nimages;
  int                nsamplesbi;
  float              radius;
  iftLabelPixelsData* data;
} iftTrainPixels;


iftLabelPixelsData  *iftCreateLabelPixelsData(int npixels);
void  iftDestroyLabelPixelsData(iftLabelPixelsData *le);
iftTrainPixels* iftCreateTrainPixels(int nimages, int nsamplesbi);
void iftDestroyTrainPixels(iftTrainPixels **Z);

iftLabelPixelsData* iftExtractLabelPixels  (iftMImage *mimg,iftImage **imgGT, iftAdjRel* A,int nsamplesbi ,float percBorder,float radius);
iftLabelPixelsData* iftExtractLabelPixelsGTBorderWGNonBorder(iftMImage *mimg,iftImage *imgGT, iftAdjRel* A,int nsamplesbi,float percBorder, float radius);
iftLabelPixelsData* iftExtractLabelPixelsWG(iftMImage *mimg,iftImage * imgGT, iftAdjRel* A,int nsamplesbi ,float percBorder,float radius);
iftTrainPixels*     iftExtractTrainPixels  (char *file_imageNames,char *file_imageNamesGT,int nsamplesbi ,float percBorder,float radius);

iftDataSet*         iftTrainPixelsToDataSet(iftTrainPixels *trainPixels, iftMSConvNetwork* msconvnet);
int                 iftSelectSupTrainPixels(iftDataSet *Z, iftTrainPixels* trainpixels,float train_perc,float border_train_perc);

void                iftWriteTrainPixels(iftTrainPixels *trainPixels,char *filenameTrainPixels);
iftTrainPixels*     iftReadTrainPixels (char *filenameTrainPixels,int bReadImages);
