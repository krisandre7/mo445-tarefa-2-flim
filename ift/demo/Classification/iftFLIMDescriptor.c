#include "ift.h"

/* It requires the same number of superpixels per image and creates a
   dataset whose number of features per image is the number of
   superpixels multiplied by the number of activation channels. The
   features are sorted in the decreasing order per channel */

void iftCreateImgRefData(char *origImgsDir, char *imgBasename, char *origImgExt, iftFileSet *fsRefData, int fileId)
{
    char filename[2048];
    sprintf(filename, "%s/%s%s", origImgsDir, imgBasename, origImgExt);
    fsRefData->files[fileId] = iftCreateFile(filename);
}

void iftWriteKernelDataSetWithFileSet(iftDataSet *Z, iftFileSet *fs, iftSampleStatus status, char *outDatasetName)
{
    iftSetRefData(Z, fs, IFT_REF_DATA_FILESET);
    iftSetStatus(Z, status);
    iftAddStatus(Z, IFT_SUPERVISED);
    iftWriteDataSet(Z, outDatasetName);
    printf("File '%s' created\n", outDatasetName);
}
  
int main(int argc, char *argv[])
{
    timer *tstart = NULL;

    if (argc != 5)
        iftError("Usage: iftFLIMDescriptor <P1> <P2> <P3> <P4>\n"
                 "P1: input folder with the activations (.mimg images)\n"
                 "P2: input folder with the superpixels (.png images)\n"
		 "P3: input folder with the original images (.png images)\n"
                 "P4: output dataset file (.zip)\n",
                 "main");

    tstart = iftTic();

    iftFileSet *fs   = iftLoadFileSetFromDirBySuffix(argv[1],".mimg", 1);
    int first        = 0;
    int last         = fs->n - 1;
    int nclasses     = iftFileSetLabelsNumber(fs);
    
    iftMImage *mimg  = iftReadMImage(fs->files[0]->path);
    int nchannels    = mimg->m; 
    char *basename   = iftFilename(fs->files[0]->path, ".mimg");
    char  filename[200];
    sprintf(filename,"%s/%s.png",argv[2],basename);    
    iftImage *img    = iftReadImageByExt(filename);    
    int nsuperpixels = iftMaximumValue(img);
    float *ch_feat   = iftAllocFloatArray(nsuperpixels);
    int   *index     = iftAllocIntArray(nsuperpixels);
    int   *nelems    = iftAllocIntArray(nsuperpixels);
    int nfeats       = nsuperpixels * nchannels;
    float sx         = (float)img->xsize/(float)mimg->xsize;
    float sy         = (float)img->ysize/(float)mimg->ysize;
    float sz         = (float)img->zsize/(float)mimg->zsize;
    iftFree(basename);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);

    iftDataSet *Z         = iftCreateDataSet(fs->n, nfeats);
    iftFileSet *fsRefData = iftCreateFileSet(fs->n);
    Z->nclasses           = nclasses;

    char *origImgsDir = NULL, *origImgExt = NULL;
    iftFileSet *fsAux     = iftLoadFileSetFromDirOrCSV(argv[3], 1, 1);
    origImgsDir           = iftDirname(fsAux->files[0]->path);
    origImgExt            = iftCopyString(iftFileExt(fsAux->files[0]->path));   
    iftDestroyFileSet(&fsAux);

    for (int i = first; i <= last; i++) {
      basename = iftFilename(fs->files[i]->path, ".mimg");
      printf("Processing file %s: %d of %ld files\r", basename, i + 1, fs->n); fflush(stdout);
      mimg     = iftReadMImage(fs->files[i]->path);
      sprintf(filename,"%s/%s.png",argv[2],basename);    
      img      = iftReadImageByExt(filename);
      iftMImage *interp = NULL;
      if (iftIs3DMImage(mimg)){
	interp = iftMInterp(mimg,sx,sy,sz);
      }else{
	interp = iftMInterp2D(mimg,sx,sy);
      }
      
      Z->sample[i].id = i;
      Z->sample[i].truelabel = fs->files[i]->label;
      
      for (int b=0; b < interp->m; b++){ /* for each channel */
	for (int k=0; k < nsuperpixels; k++){ /* initialize the channel's
						 features */
	  ch_feat[k]=0.0; index[k]=k; nelems[k]=0;
	}
	/* compute the average activation per superpixel */
	for (int p=0; p < interp->n; p++) { 
	  if (img->val[p]>0){
	    ch_feat[img->val[p]-1] += interp->val[p][b];
	    nelems[img->val[p]-1]   += 1;
	  }
	}
	for (int k=0; k < nsuperpixels; k++){
	  if (nelems[k]>0)
	    ch_feat[k] = ch_feat[k]/nelems[k];
	}
	/* Sort the values */
	iftFHeapSort(ch_feat, index, nsuperpixels, IFT_DECREASING);
	  
	/* Copy the channel's features to the sample's features */

	for (int k=0,j=b*nsuperpixels; k < nsuperpixels; k++,j++){
	  Z->sample[i].feat[j] = ch_feat[k];
	}
      }

      /* float *feat = iftAllocFloatArray(Z->nfeats); */
      /* int   *ind  = iftAllocIntArray(Z->nfeats); */

      /* for (int f=0; f < Z->nfeats; f++){ */
      /* 	feat[f] = Z->sample[i].feat[f]; */
      /* 	ind[f]  = f; */
      /* } */
      /* iftFHeapSort(feat, ind, Z->nfeats, IFT_DECREASING); */
      /* for (int f=0; f < Z->nfeats; f++){ */
      /* 	Z->sample[i].feat[f] = feat[f]; */
      /* } */
      /* iftFree(feat); */
      /* iftFree(ind); */

      
      iftCreateImgRefData(origImgsDir, basename, origImgExt, fsRefData, i);
      iftDestroyImage(&img);
      iftDestroyMImage(&mimg);
      iftDestroyMImage(&interp);
      iftFree(basename);
    }
    /* Write the output dataset */
    iftWriteKernelDataSetWithFileSet(Z, fsRefData, IFT_TRAIN, argv[4]);

    /* Free memory */
    iftDestroyFileSet(&fs);
    iftDestroyFileSet(&fsRefData);
    iftDestroyDataSet(&Z);
    iftFree(origImgsDir);
    iftFree(origImgExt);
    iftFree(ch_feat);
    iftFree(index);
    iftFree(nelems);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
