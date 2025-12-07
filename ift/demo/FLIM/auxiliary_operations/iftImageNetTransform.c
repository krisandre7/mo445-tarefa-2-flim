#include "ift.h"

int main(int argc, char *argv[])
{
  
  if (argc != 3){
    iftError("Usage: iftImageNetTransform <...>\n"
	     "[1] input folder with images in the png file format. Gray images are converted into RGBNorm color space for the ImageNet transformation. \n"
	     "[2] output folder with mimages transformed by the ImageNet parameters.\n",
	     "main");
  }

  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(argv[1],".png", 1);
  iftMakeDir(argv[2]);
  float mean[3]  = {0.485, 0.456, 0.406};
  float stdev[3] = {0.229, 0.224, 0.225};
  char filename[200];
  int nimages = fs->n;
  
  for (int i=0; i < nimages; i++) {
    iftImage  *img  = iftReadImageByExt(fs->files[i]->path);
    char *basename  = iftFilename(fs->files[i]->path,".png");
    iftMImage *mimg = NULL;
    
    mimg = iftImageToMImage(img, RGBNorm_CSPACE);
#pragma omp parallel 
    for (int p=0; p < mimg->n; p++){
      for (int b=0; b < mimg->m; b++){
	mimg->val[p][b] = (mimg->val[p][b]-mean[b])/stdev[b];
      }
    }
    sprintf(filename,"%s/%s.mimg",argv[2],basename);
    iftWriteMImage(mimg,filename);
    iftFree(basename);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
  }

  iftDestroyFileSet(&fs);
  
  return (0);
}
