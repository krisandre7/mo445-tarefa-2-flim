#include "ift.h"

iftMatrix *GaborFilters(float adj_radius, float freq)
{
  int n_kernels   = 4;
  iftAdjRel   *A  = iftCircular(adj_radius);
  iftMMKernel *K  = iftCreateMMKernel(A, 1, n_kernels);
  float sigma_x=8, sigma_y=4;
  float *theta = iftAllocFloatArray(n_kernels);

  theta[0] = 0;
  theta[1] = 45*IFT_PI/180.0;
  theta[2] = 90*IFT_PI/180.0;
  theta[3] = 135*IFT_PI/180.0;

  for (int i = 0; i < n_kernels; i++)
    {
      float mag    = 0.0;
      float *gabor = iftGaborFilter(A, theta[i], freq, sigma_x, sigma_y);
      for (int l = 0; l < A->n; l++)
	{
	  K->weight[i][0].val[l] = gabor[l];
	  mag += gabor[l]*gabor[l];
	}
      free(gabor);
      mag = sqrtf(mag);
      for (int l = 0; l < A->n; l++)
	K->weight[i][0].val[l] /= mag;
      
    }

  iftMatrix *M   = iftMMKernelToMatrix(K);
  iftMatrix *M_t = iftTransposeMatrix(M);

  iftDestroyMMKernel(&K);
  iftDestroyAdjRel(&A);
  iftFree(theta);
  
  return(M_t);
}

int main(int argc, const char *argv[])
{

  if (argc != 5)
    iftError("iftDefectByGabor <input_folder> <adj. radius> <output folder>", "main");
  
  iftFileSet *fs      = iftLoadFileSetFromDirBySuffix(argv[1], ".png", true);
  char filename[400];
  iftMatrix *gabor    = GaborFilters(atof(argv[2]),atof(argv[3]));
  iftMakeDir(argv[4]);
  iftAdjRel *A = iftCircular(atof(argv[2])), *B = iftCircular(1.0);
  iftColor RGB, YCbCr;
  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr = iftRGBtoYCbCr(RGB,255);
  iftImage *ref       = iftReadImageByExt("normal_textil.png");
  iftMImage *mimg_ref = iftImageToMImage(ref,GRAYNorm_CSPACE);
  iftMatrix *XI_ref   = iftMImageToFeatureMatrix(mimg_ref,A,NULL);
  iftMatrix *XJ_ref   = iftMultMatrices(XI_ref, gabor);
  iftDestroyMatrix(&XI_ref);
  iftMImage *activ_ref = iftMatrixToMImage(XJ_ref, mimg_ref->xsize,
					   mimg_ref->ysize, mimg_ref->zsize,
					   gabor->ncols, 'c');
  iftDestroyImage(&ref);
  iftDestroyMatrix(&XI_ref);
  iftDestroyMatrix(&XJ_ref);
  iftDestroyMImage(&mimg_ref);

  float *mean_ref = iftAllocFloatArray(activ_ref->m);
  float *std_ref = iftAllocFloatArray(activ_ref->m);
  int *n_ref      = iftAllocIntArray(activ_ref->m);

  for (int p=0; p < activ_ref->n; p++){
    for (int b=0; b < activ_ref->m; b++) {
      activ_ref->val[p][b] = powf(activ_ref->val[p][b],2); 
    }
  }
  
  for (int p=0; p < activ_ref->n; p++){
    for (int b=0; b < activ_ref->m; b++) {
      if (activ_ref->val[p][b]>0){
	mean_ref[b] += activ_ref->val[p][b];
	n_ref[b]++;
      }
    }
  }
  for (int b=0; b < activ_ref->m; b++) {
    mean_ref[b]  /= n_ref[b];
  }

  for (int p=0; p < activ_ref->n; p++){
    for (int b=0; b < activ_ref->m; b++) {      
      if (activ_ref->val[p][b]>0){
	std_ref[b] += powf((activ_ref->val[p][b]-mean_ref[b]),2);
      }
    }
  }
  for (int b=0; b < activ_ref->m; b++) {
    std_ref[b]  /= n_ref[b];
    std_ref[b]   = sqrtf(std_ref[b]);
  }
  
  
  for (int i=0; i < fs->n; i++) {
    char *basename      = iftFilename(fs->files[i]->path,".png");      
    iftImage *orig      = iftReadImageByExt(fs->files[i]->path);
    iftFree(orig->Cb);
    iftFree(orig->Cr);
    orig->Cb = orig->Cr = 0;
    iftMImage *mimg     = iftImageToMImage(orig,GRAYNorm_CSPACE);
    iftMatrix *XI       = iftMImageToFeatureMatrix(mimg,A,NULL);
    iftMatrix *XJ       = iftMultMatrices(XI, gabor);
    iftDestroyMatrix(&XI);
    iftMImage *activ    = iftMatrixToMImage(XJ, mimg->xsize,
					    mimg->ysize, mimg->zsize,
					    gabor->ncols, 'c');
        
    for (int p=0; p < activ_ref->n; p++){
      for (int b=0; b < activ_ref->m; b++) {
	activ->val[p][b] = powf(activ->val[p][b],2); 
	if (fabs(activ->val[p][b]-mean_ref[b])<3.5*std_ref[b])
	  activ->val[p][b]=0;
      }    
      mimg->val[p][0] = 4*(activ->val[p][0]-activ_ref->val[p][0])+
	4*(activ->val[p][2]-activ_ref->val[p][2]) +
	(activ->val[p][1]-activ_ref->val[p][1]) +
	(activ->val[p][3]-activ_ref->val[p][3]); 
      if (mimg->val[p][0]<0)
	mimg->val[p][0]=0;
    }
    
    iftDestroyMImage(&activ);
    iftDestroyMatrix(&XJ);
      
    iftImage *img    = iftMImageToImage(mimg,255,0);     
    iftDestroyMImage(&mimg);
    iftImage *bin    = iftThreshold(img,1,255,255);
    iftDestroyImage(&img);
    iftSet *S        = NULL;
    iftImage *dil    = iftDilateBin(bin,&S,2.0);
    iftDestroyImage(&bin);
    iftDestroySet(&S);
    iftDrawBorders(orig, dil, B, YCbCr, B);
    img = iftCrop2DImageByBorder(orig,45);
    sprintf(filename,"%s/%s.png",argv[4],basename);
    iftWriteImageByExt(img,filename);
    iftDestroyImage(&orig);
    iftDestroyImage(&img);
    iftDestroyImage(&dil);
    iftFree(basename);
  }
  
  iftFree(n_ref);
  iftFree(mean_ref);
  iftDestroyMImage(&activ_ref);
  iftDestroyFileSet(&fs);
  iftDestroyMatrix(&gabor);
  iftDestroyAdjRel(&A);  
  iftDestroyAdjRel(&B);  
  return 0;
}






