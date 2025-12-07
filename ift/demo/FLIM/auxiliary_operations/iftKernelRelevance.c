#include "ift.h"

void NormalizeImagePerChannel(iftMImage *img)
{

  for(int b=0; b < img->m; b++) {
    float Imin=IFT_INFINITY_FLT, Imax=IFT_INFINITY_FLT_NEG;
    for (int p=0; p < img->n; p++){
      if (img->val[p][b]>Imax)
	Imax = img->val[p][b];
      if ((img->val[p][b]<Imin)&&(img->val[p][b]>0))
	Imin = img->val[p][b];
    }
    if (Imin < Imax)
      for (int p=0; p < img->n; p++){
	img->val[p][b] = (img->val[p][b]-Imin)/(Imax-Imin);
      }
  }
}

int main(int argc, char *argv[])
{
  timer     *tstart=NULL;
  char       filename[200];
  
  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  
  if (argc != 6) {
    printf("iftKernelRelevance <P1> <P2> <P3> <P4>\n");
    printf("P1 (in) : folder with activation images\n");
    printf("P2 (in) : folder with object masks\n");
    printf("P3 (out): folder with object saliency maps\n");
    printf("P4 (out): file (.txt) with the relevance of each kernel in [0,1]\n");
    printf("P5 (out): file (.txt) with the influence of each image in [0,1]\n");
    exit(0);
  }

  tstart = iftTic();

  char *activ_dir     = argv[1];
  char *mask_dir      = argv[2];
  iftFileSet *fs      = iftLoadFileSetFromDirBySuffix(activ_dir,".mimg", 1);  
  int nimages         = fs->n;
  char *salie_dir     = argv[3];
  iftMakeDir(salie_dir);
  char *relfilename   = argv[4];
  char *inffilename   = argv[5];
  
  iftMImage *activ    = iftReadMImage(fs->files[0]->path);
  int nchannels       = activ->m;
  float *weight       = iftAllocFloatArray(nchannels);
  iftDestroyMImage(&activ);

  iftMakeDir("tmp");
  
  for (int i=0; i < nimages; i++) {
    printf("Processing image %s\n",fs->files[i]->path);
    activ             = iftReadMImage(fs->files[i]->path);
    NormalizeImagePerChannel(activ);
    
    char *basename    = iftFilename(fs->files[i]->path,".mimg");
    if (iftIs3DMImage(activ)){      
      sprintf(filename,"%s/%s.nii.gz",mask_dir,basename);
    } else {
      sprintf(filename,"%s/%s.png",mask_dir,basename);
    }
    if (!iftFileExists(filename)){
      iftDestroyMImage(&activ);
      iftFree(basename);
      continue;
    }

    /* Read mask and scale the activation image to be the same size of
       the mask */ 

    iftImage *mask      = iftReadImageByExt(filename);
    iftMImage *aux_activ = NULL;
    float sx, sy, sz;
    sx = (float)mask->xsize/(float)activ->xsize;
    sy = (float)mask->ysize/(float)activ->ysize;
    if (iftIs3DMImage(activ)){
      sz        = (float)mask->zsize/(float)activ->zsize;
      aux_activ = iftMInterp(activ,sx,sy,sz);
    }else{
      aux_activ = iftMInterp2D(activ,sx,sy);
    }
    iftDestroyMImage(&activ);
    activ = aux_activ;
    sprintf(filename,"./tmp/%s.mimg",basename);
    iftWriteMImage(activ,filename);
		   
    /* Compute relevance per kernel */

    float area_obj=0;
    float *weight_obj=iftAllocFloatArray(activ->n);
    float *weight_bkg=iftAllocFloatArray(activ->n);
    for (int p=0; p < activ->n; p++) {
      if (mask->val[p] != 0){
	area_obj++;
	for (int b=0; b < nchannels; b++){
	  weight_obj[b] += activ->val[p][b];
	}
      } else {
	for (int b=0; b < nchannels; b++){
	  weight_bkg[b] += (1.0 - activ->val[p][b]);
	}
      }
    }
    for (int b=0; b < nchannels; b++){
      weight[b] += 0.5*(weight_obj[b]/area_obj + weight_bkg[b]/(activ->n-area_obj));
    }
    iftFree(weight_obj);
    iftFree(weight_bkg);
    iftDestroyImage(&mask);
    iftDestroyMImage(&activ);
    iftFree(basename);
  }
  
  /* centralize and save weights */

  FILE *fp = fopen(relfilename,"w");
  
  float wT=0.0;
  for (int b=0; b < nchannels; b++){
    wT += weight[b];
  }

  float mean_weight = wT/nchannels;
  float max_weight=0;
  
  for (int b=0; b < nchannels; b++){
    weight[b] = (weight[b] - mean_weight);
    if (weight[b]>0){
      if (weight[b]>max_weight)
	max_weight=weight[b];      
    }else{
      if (-weight[b]>max_weight)
	max_weight=-weight[b];      
    }
  }

  fprintf(fp,"%d\n",nchannels);
  for (int b=0; b < nchannels; b++){
    fprintf(fp,"%f\n",weight[b]/max_weight);
  }
  fclose(fp);

  /* generate saliency maps */

  fp = fopen(inffilename,"w");
  fprintf(fp,"%d\n",nimages);
  
  for (int i=0; i < nimages; i++) {
    char *basename     = iftFilename(fs->files[i]->path,".mimg");
    sprintf(filename,"./tmp/%s.mimg",basename);
    printf("Processing image %s\n",filename);
    activ              = iftReadMImage(filename);

    if (iftIs3DMImage(activ)){      
      sprintf(filename,"%s/%s.nii.gz",mask_dir,basename);
    } else {
      sprintf(filename,"%s/%s.png",mask_dir,basename);
    }
    if (!iftFileExists(filename)){
      iftDestroyMImage(&activ);
      iftFree(basename);
      continue;
    }

    iftImage  *mask  = iftReadImageByExt(filename);
    iftFImage *salie = iftCreateFImage(activ->xsize,activ->ysize,activ->zsize);

    for (int p=0; p < activ->n; p++) {
      for (int b=0; b < nchannels; b++){
	salie->val[p] += activ->val[p][b] * weight[b];
      }
      salie->val[p] = iftMax(salie->val[p],0);
    }

    if (iftIs3DMImage(activ)){      
      sprintf(filename,"%s/%s.nii.gz",salie_dir,basename);
    } else {
      sprintf(filename,"%s/%s.png",salie_dir,basename);
    }
    
    iftImage *nsalie = iftFImageToImage(salie,255);
    iftWriteImageByExt(nsalie,filename);

    /* compute image influnce based on DICE */
    iftImage *aux_salie = iftThreshold(nsalie,1,255,1);
    iftDestroyImage(&nsalie);
    nsalie     = aux_salie;
    float dice = iftDiceSimilarity(nsalie,mask);
    
    fprintf(fp,"%f\n",dice);

    iftDestroyImage(&nsalie);
    iftDestroyFImage(&salie);
    iftFree(basename);
    iftDestroyMImage(&activ);
  }
  fclose(fp);

  iftRemoveDir("tmp");
  iftFree(weight);
  iftDestroyFileSet(&fs);

  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return 0;
}








