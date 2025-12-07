//Structre that stores the name of all image and respective class.
#include "ift.h"

typedef struct file_names { 
  char filename[256];
} FileNames;

FileNames * createFileNames (int nimages);
void        destroyFileNames (FileNames * ITS);
int         countImages (char * file_imageNames);
char*       getFolderName (char * file_imageNames);
void        loadFileNames (FileNames *FN, char file_imageNames[]);

iftMSConvNetwork* parseConfigFile (char * file_config);
iftImage*  iftWaterGrayBorder(iftImage *label_image,iftAdjRel* A);
iftImage*  iftWaterGrayIBorder(iftImage* gradient_image,iftImage *label_image,iftAdjRel* A);
iftFImage* iftWaterGrayFBorder(iftFImage* gradient_image,iftImage *label_image,iftAdjRel* A);
