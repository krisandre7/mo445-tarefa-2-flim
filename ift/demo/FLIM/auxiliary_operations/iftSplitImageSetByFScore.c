#include "ift.h"

/* Author: Alexandre Xavier Falcão (September 5th 2024) 

   Description: Reads csv file with image basename and results,
   splitting it into two file sets -- the ones with fscore greater or
   equal to a threshold and the remaining ones with fscore less than
   that threshold.

*/

int main(int argc, char *argv[])
{

  /* Example: iftSplitImageSetByFScore <P1> <P2> <P3> <P4>
     
     P1: csv file with the results of a FLIM model -- image basename; fscore
     P2: fscore threshold to split the image set
     P3: folder with the images
     P4: file extension of the images in that folder (.png, .nii.gz)
  
   */


  if (argc!=5){ 
    iftError("Usage: iftSplitImageSetByFScore <P1> <P2> <P3> <P4>\n"
	     "P1: csv file with the results of a FLIM model -- image basename; fscore\n"
	     "P2: fscore threshold to split the image set\n"
	     "P3: folder with the images\n"
	     "P4: file extension of the images in that folder (.png, .nii.gz)\n",
	     "main");
  }
  
  timer *tstart = iftTic();

  iftCSV *image_set = iftReadCSV(argv[1],';');
  float   thres     = atof(argv[2]);
  char   *image_dir = argv[3];
  char   *ext       = argv[4];

  FILE *fpH = fopen("imageSetHigh.csv","w");
  FILE *fpL = fopen("imageSetLow.csv","w");
  
  for(int i = 0; i < image_set->nrows-1; i++) {
    char *basename = image_set->data[i][0];
    float fscore   = atof(image_set->data[i][1]);
    if (fscore >= thres)
      fprintf(fpH,"%s/%s%s\n",image_dir,basename,ext);
    else
      fprintf(fpL,"%s/%s%s\n",image_dir,basename,ext);      
  }
  fclose(fpH);
  fclose(fpL);
  iftDestroyCSV(&image_set);
  
  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
