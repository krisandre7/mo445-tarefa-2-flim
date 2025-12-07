#include "ift.h"

/* Author: Alexandre Xavier Falcão (May 31st 2024) 

   Description: Preprocess the original images to correct intensity
   variations based on inhomogeneity correction by F. Cappabianco with
   0.0 < alpha <= 2.0 and 0 < beta <= 0.1.
   
*/

iftImage *iftCorrIntensityVariation(iftImage *img, iftAdjRel *A, float alpha, float beta)
{
  int Imax        = iftMaximumValue(img);
  iftFImage *corr = iftCreateFImage(img->xsize,img->ysize,img->zsize);
  
#pragma omp parallel for schedule(auto)
  for (int p =0; p < img->n; p++) {
    iftVoxel u = iftGetVoxelCoord(img,p);
    int Ir     = img->val[p]; 

    for (int i =1; i < A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(img,v)) {
	int q = iftGetVoxelIndex(img,v);	
	if (img->val[q]>Ir)
	  Ir = img->val[q];
      }
    }
    if (Ir >= beta*Imax) {
      float f = pow(((float)Imax/(float)Ir),alpha);
      corr->val[p] = img->val[p]*f;
    } else {
      corr->val[p] = img->val[p];
    }
  }
  iftImage *filt = iftFImageToImage(corr,iftNormalizationValue(Imax));
  iftDestroyFImage(&corr);
  
  return(filt);
}

int main(int argc, char *argv[])
{

  /* Example: iftCorrIntensityVariation orig 5.0 1.1 0.05 filt */


  if (argc!=6){ 
    iftError("Usage: iftCorrIntensityVariation <P1> <P2> <P3> <P4> <P5>\n"
	     "[1] folder with the original images\n"
	     "[2] adjacency radius of the filter\n"
	     "[3] alpha parameter in (0.0,2.0] \n"
	     "[4] beta parameter in (0.0,0.10] \n"
	     "[5] folder with the filtered images\n",	 
	     "main");
  }
  
  timer *tstart = iftTic();

  iftFileSet *fs     = iftLoadFileSetFromDirBySuffix(argv[1],".png", true);
  char *filename     = iftAllocCharArray(512);
  float adj_radius   = atof(argv[2]);
  iftAdjRel  *A      = iftCircular(adj_radius);
  float alpha        = atof(argv[3]);
  float beta         = atof(argv[4]);
  char *output_dir   = argv[5];
  iftMakeDir(output_dir);

  for(int i = 0; i < fs->n; i++) {
    //printf("Processing image %d of %ld\r", i + 1, fs->n);
    char *basename    = iftFilename(fs->files[i]->path,".png");      
    sprintf(filename,"%s/%s.png",output_dir,basename);
    iftImage *orig    = iftReadImageByExt(fs->files[i]->path);
    iftImage *filt    = iftCorrIntensityVariation(orig,A,alpha,beta);
    iftDestroyImage(&orig);
    iftWriteImageByExt(filt,filename);
    iftDestroyImage(&filt);
    iftFree(basename);
  }

  iftFree(filename);
  iftDestroyFileSet(&fs);
  iftDestroyAdjRel(&A);
  
  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
