#include "ift.h"

typedef struct _weights_per_class {
  int nclasses; 
  int nweights;
  float **weight;
} WeightsPerClass;

WeightsPerClass *CreateWeightsPerClass(int nclasses, int nweights)
{
  WeightsPerClass *wc = (WeightsPerClass *)calloc(1,sizeof(WeightsPerClass));

  wc->nclasses = nclasses;
  wc->nweights = nweights;
  wc->weight   = (float **)calloc(nclasses, sizeof(float *));
  
  for (int i=0; i < wc->nclasses; i++)
    wc->weight[i] = iftAllocFloatArray(nweights);

  return(wc);
}

void DestroyWeightsPerClass(WeightsPerClass **wc)
{
  if (*wc != NULL){
    for (int i=0; i < (*wc)->nclasses; i++)
      iftFree((*wc)->weight[i]);
    iftFree((*wc)->weight);
    iftFree(*wc);
    *wc = NULL;
  }
}

WeightsPerClass *ReadWeightsPerClass(char *weightfile)
{
  WeightsPerClass *wc;
  FILE *fp = fopen(weightfile,"r");
  int nclasses, nweights;
  
  fscanf(fp,"%d %d",&nclasses,&nweights);
  wc = CreateWeightsPerClass(nclasses,nweights);
  for (int c=0; c < nclasses; c++){
    for (int w=0; w < nweights; w++){
      fscanf(fp,"%f",&wc->weight[c][w]);
    }
  }
  fclose(fp);

  return(wc);
}

void NormalizeBandsInPlace(iftMImage *mimg)
{
  for (int b=0; b < mimg->m; b++) {
    float min = IFT_INFINITY_FLT;
    float max = IFT_INFINITY_FLT_NEG;
    for (int p = 0; p < mimg->n; p++) {
      if (mimg->val[p][b] < min)
	min = mimg->val[p][b];
      if (mimg->val[p][b] > max)
	max = mimg->val[p][b];
    }
    if (!iftAlmostZero(max-min))
      for (int p = 0; p < mimg->n; p++) {
	mimg->val[p][b] = (mimg->val[p][b]-min)/(max-min);
      }
  }
}

iftImage *SalienceMap(iftMImage *mimg, WeightsPerClass *wc, int label) 
{
  iftMImage *salie = iftCreateMImage(mimg->xsize,mimg->ysize,mimg->zsize,1);

  for (int p=0; p < mimg->n; p++){
    for (int b=0; b < mimg->m; b++){
      salie->val[p][0] += mimg->val[p][b]*wc->weight[label][b];
    }
    if (salie->val[p][0]<0) salie->val[p][0]=0;
  }

  int Imax;
  if (iftIs3DMImage(mimg))
    Imax = 4095;
  else
    Imax = 255;
  
  iftImage *salie_map = iftMImageToImage(salie,Imax,0);
  iftDestroyMImage(&salie);

  return(salie_map);
}
  
int main(int argc, char *argv[])
{
    if (argc!=5)
        iftError("Usage: iftSalienceMaps <P1> <P2> <P3>\n"
	       "[1] input_layer: Folder with input activation images\n"
	       "[2] weight file: Text file with activation (filter) weights per class\n"
	       "[3] scale factor to return to the original size\n"	 
	       "[4] output_dir: Folder with one salience map per class\n",		 
	       "main");

    timer *tstart = iftTic();

    iftFileSet * fs   = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);
    char *weightfile  = argv[2];
    float scale       = atof(argv[3]);
    char *outdir      = argv[4];
    iftMakeDir(outdir);
    WeightsPerClass *wc = ReadWeightsPerClass(weightfile);
    char filename[200];
    
    for(int i = 0; i < fs->n; i++) {
      printf("Processing image %d of %ld\r", i + 1, fs->n);
      char *basename  = iftFilename(fs->files[i]->path,".mimg");
      iftMImage *mimg = iftReadMImage(fs->files[i]->path);
      //NormalizeBandsInPlace(mimg);
      if (mimg->m != wc->nweights)
	iftError("Number of weights differ from the number of bands","main");
      for (int label=0; label < wc->nclasses; label++){
	iftImage *salie_map = SalienceMap(mimg,wc,label);
	iftImage *interp_map = NULL;
	if (iftIs3DMImage(mimg)){
	  interp_map = iftInterp(salie_map,scale,scale,scale);
	  sprintf(filename,"%s/%s_%d.nii.gz",outdir,basename,label);
	}else{
	  interp_map = iftInterp2D(salie_map,scale,scale);	  
	  sprintf(filename,"%s/%s_%d.png",outdir,basename,label);
	}
	iftWriteImageByExt(interp_map,filename);
	iftDestroyImage(&salie_map);
	iftDestroyImage(&interp_map);
      }
      iftDestroyMImage(&mimg);
      iftFree(basename);
    }
    
    iftDestroyFileSet(&fs);
    DestroyWeightsPerClass(&wc);
    
    printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
