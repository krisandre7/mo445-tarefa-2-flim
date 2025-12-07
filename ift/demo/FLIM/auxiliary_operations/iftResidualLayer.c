#include "ift.h"

/* Author: Alexandre Xavier Falcão (May 31st, 2024)

   Description: Assuming the activation maps of two given layers have
   the same number of channels, it subtracts the deepest one (P2) from
   the other (P1), and applies ReLU. It also takes into account
   object/background activations in the process since the idea is to
   the important activations, removing intensity variations.

*/

float *LoadKernelWeights(char *filename)
{
  int    number_of_kernels;
  FILE  *fp;
  float *weight;
  
  fp = fopen(filename, "r");
  fscanf(fp, "%d", &number_of_kernels);
  weight = iftAllocFloatArray(number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++) {
    fscanf(fp, "%f ", &weight[k]);
  }
  fclose(fp);

  return(weight);
}

int main(int argc, char *argv[])
{
  
  if (argc!=3){ 
    iftError("Usage: iftResidualLayer <P1> <P2>\n"
	     "[1] first layer number L\n"
	     "[2] last  layer number L+K\n",
	     "main");
  }
  
  timer *tstart = iftTic();

  char *filename    = iftAllocCharArray(512);
  int   first_layer = atoi(argv[1]);
  int   last_layer  = atoi(argv[2]);
  char first_dir[12], last_dir[12];
  sprintf(first_dir,"layer%d",first_layer);
  sprintf(last_dir,"layer%d",last_layer);
  iftFileSet *fs    = iftLoadFileSetFromDirBySuffix(last_dir,".mimg", true);
  float scale[3];
  int norm_value;
  
  for(int i = 0; i < fs->n; i++) {
    printf("Processing image %d of %ld\r", i + 1, fs->n);
    char *basename          = iftFilename(fs->files[i]->path,".mimg");
    sprintf(filename,"%s/%s.mimg",first_dir,basename);
    iftMImage *mimg1 = iftReadMImage(filename);
    sprintf(filename,"flim/conv%d-weights.txt",last_layer);
    float *weight    = LoadKernelWeights(filename);	
    sprintf(filename,"%s/%s.mimg",last_dir,basename);
    iftMImage *mimg2 = iftReadMImage(filename);
    
    if (iftIs3DMImage(mimg1)){
      norm_value = 4095;
      scale[0]   = (float)mimg2->xsize/(float)mimg1->xsize;
      scale[1]   = (float)mimg2->ysize/(float)mimg1->ysize;
      scale[2]   = (float)mimg2->zsize/(float)mimg1->zsize;
    }else{
      norm_value = 255;
      scale[0]   = (float)mimg2->xsize/(float)mimg1->xsize;
      scale[1]   = (float)mimg2->ysize/(float)mimg1->ysize;
      scale[2]   = 1;
    }
    
    for (int b=0; b < mimg1->m; b++){
      iftImage *img1   = iftMImageToImage(mimg1,norm_value,b);
      iftImage *img2   = iftMImageToImage(mimg2,norm_value,b);
      iftImage *interp = NULL;
      if (iftIs3DImage(img1)){
	interp = iftInterp(img1,scale[0],scale[1],scale[2]);
      }else{
	interp = iftInterp2D(img1,scale[0],scale[1]);
      }
      iftDestroyImage(&img1);
      img1     = interp;
      
      if (weight[b]<0.0){
	iftImage *aux1 = iftComplement(img1);
	iftDestroyImage(&img1);
	img1 = aux1;
	iftImage *aux2 = iftComplement(img2);
	iftDestroyImage(&img2);
	img2 = aux2;
      }

      iftImage *res = iftSubReLU(img2,img1);
      if (weight[b]<0.0){
	iftImage *aux1 = iftComplement(res);
	iftDestroyImage(&res);
	res = aux1;
      }
      for (int p=0; p < res->n; p++) {
	mimg2->val[p][b] = (float)res->val[p]/norm_value;		        
      }
      iftDestroyImage(&res);      
      iftDestroyImage(&img1);
      iftDestroyImage(&img2);
    }
    iftWriteMImage(mimg2,filename);
    iftDestroyMImage(&mimg1);
    iftDestroyMImage(&mimg2);
    iftFree(basename);
  }
  
  iftFree(filename);
  iftDestroyFileSet(&fs);
  
  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
