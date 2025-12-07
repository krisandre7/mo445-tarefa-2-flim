#include "ift.h"

/* Author: Alexandre Xavier Falcão (December 30th, 2023)

   Description: Substitutes salience maps in a deeper layer L+K by the
   inferior reconstruction of the corresponding salience map in layer
   L using layer L+K as marker image.

*/

#define FILE_EXT ".png"

int main(int argc, char *argv[])
{

  
  if (argc!=4){ 
    iftError("Usage: iftInferiorRecOnSalie <P1> <P2> <P3>\n"
	     "[1] folder with all salience maps\n"
	     "[2] layer number L\n"
	     "[3] layer number L+K\n",
	     "main");
  }
  
  timer *tstart = iftTic();

  char *filename    = iftAllocCharArray(512);
  char suffix[12];
  int first_layer   = atoi(argv[2]);
  int last_layer    = atoi(argv[3]);
  sprintf(suffix,"_layer%d%s",last_layer,FILE_EXT);
  iftFileSet *fs    = iftLoadFileSetFromDirBySuffix(argv[1],suffix, true);
    
  for(int i = 0; i < fs->n; i++) {
    printf("Processing image %d of %ld\r", i + 1, fs->n);
    char *basename = iftFilename(fs->files[i]->path,suffix);
    sprintf(filename,"%s/%s_layer%d%s",argv[1],basename,first_layer,FILE_EXT);
    iftImage *img1  = iftReadImageByExt(filename);
    sprintf(filename,"%s/%s_layer%d%s",argv[1],basename,last_layer,FILE_EXT);
    iftImage *img2  = iftReadImageByExt(filename);
    for (int p=0; p < img2->n; p++) { /* assures img1 >= img2 */
      img2->val[p] = iftMin(img1->val[p],img2->val[p]);
    }
	
    if ((iftMaximumValue(img1)!=0)&&(iftMaximumValue(img2)!=0)){
      iftImage *irec = iftInferiorRec(img1,img2,NULL);
      iftDestroyImage(&img2);
      img2 = irec;
      iftWriteImageByExt(img2,filename); 
    }

    iftFree(basename);
    iftDestroyImage(&img1);
    iftDestroyImage(&img2);
  }
  
  iftFree(filename);
  iftDestroyFileSet(&fs);
  
  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
