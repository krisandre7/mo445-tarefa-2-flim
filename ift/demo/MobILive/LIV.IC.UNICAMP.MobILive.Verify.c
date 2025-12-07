#include <stdio.h>
#include <stdlib.h>

#include "ift.h"
#include "tiffio.h"
#include "iftTIFF.h"

#include "LIV.IC.UNICAMP.MobILive.h"

//#define _SILENCE

/************* BODIES *************/
int iftCompareImageName(  void * a,   void * b) {
  iftImageNames *first = (iftImageNames*) a;
  iftImageNames *second = (iftImageNames*) b;
  if ( strcmp(first->image_name, second->image_name) < 0 ) return -1;
  if ( strcmp(first->image_name, second->image_name) == 0 ) return 0;

  return 1;
}


int main(int argc, char* argv[])
{
  int i,f,cl;

  char msg[200];
  if (argc < 2) {
    sprintf(msg,"usage: %s test_images_path\n",argv[0]);
    iftError(msg,"opentiff");
  }

  // is argv[1] a directory?
  char directory[200],fullpath[200];
  strcpy(directory,argv[1]);
  if (!iftDirExists(directory))
    iftError("Error: Images directory doesn't exist!!!", "LIV.IC.UNICAMP.MobILive.Verify");

  int nimages;
  iftImageNames *img_names;
  nimages = iftCountImageNames(directory,".tiff"); // tiff
  img_names = iftCreateAndLoadImageNames(nimages, directory,".tiff"); // tiff
  qsort(img_names, nimages, sizeof(iftImageNames), iftCompareImageName);

  iftConvNetwork **convnet   = (iftConvNetwork**) malloc(sizeof(iftConvNetwork*)*nCNNs);
  int            *nfeatures = iftAllocIntArray(nCNNs);

  iftSample       *w         = (iftSample*)       malloc(sizeof(iftSample)*nCNNs);
  float           *rho       = iftAllocFloatArray(nCNNs);
  double         **class     = (double**)         malloc(sizeof(double*) * nCNNs);

  float          **mean      = (float**)          malloc(sizeof(float*)  * nCNNs);
  float          **stdev     = (float**)          malloc(sizeof(float*)  * nCNNs);

  char *file,*fileConvNet,scl[10];
  for(cl=0;cl<nCNNs;cl++) {
    // load convnet 
    sprintf(scl,"%d",cl);
    fileConvNet=replace_str(CONVNET_NAME,"#",scl);
    convnet[cl] = iftReadConvNetwork(fileConvNet);
    free(fileConvNet);

    int xsize[convnet[cl]->nstages], ysize[convnet[cl]->nstages], zsize[convnet[cl]->nstages], nbands[convnet[cl]->nstages];
    iftImageDimensionsAlongNetwork(convnet[cl], xsize, ysize, zsize, nbands);
    nfeatures[cl] = xsize[convnet[cl]->nstages - 1] * ysize[convnet[cl]->nstages - 1] *
                    zsize[convnet[cl]->nstages - 1] * nbands[convnet[cl]->nstages - 1];

    if (nimages <= 0)
      iftError("No images found to be processed","LIV.IC.UNICAMP.MobILive.Verify");

    // Loading SVM model (hyperplane, rho, and class data)
    w[cl].feat = (float*) iftAllocFloatArray(nfeatures[cl]);
    class[cl] = iftAllocDoubleArray(2);

    char fileModel[200];
    sprintf(scl,"%d",cl);
    file=replace_str(MODEL_NAME,"#",scl);
    sprintf(fileModel,"%s.%d.data",file,nfeatures[cl]);
    free(file);
    FILE *pFM = fopen(fileModel,"rb");
    if ( (pFM == NULL) ) {
      sprintf(msg,"Can't read model classification file (%s)",fileModel);
      iftError(msg,"LIV.IC.UNICAMP.MobILive.Verify");
    }
    if ( fread((w[cl].feat),sizeof(float),nfeatures[cl],pFM) != nfeatures[cl])
      iftError("Can't read hyperplane values"     ,"LIV.IC.UNICAMP.MobILive.Verify");
    if ( fread(&(rho[cl]),sizeof(float),1,pFM) != 1)
      iftError("Can't read bias hyperplane value" ,"LIV.IC.UNICAMP.MobILive.Verify");
    //rho -= 0.05; // best for standard
    //rho -= 0.05;

    if ( fread(&(class[cl][0]),sizeof(double),1,pFM) != 1)
      iftError("Can't read class FAKE information","LIV.IC.UNICAMP.MobILive.Verify");
    if ( fread(&(class[cl][1]),sizeof(double),1,pFM) != 1)
      iftError("Can't red class REAL information" ,"LIV.IC.UNICAMP.MobILive.Verify");
    //  fprintf(stdout,"rho: %f, cl0: %lf, cl1: %lf\n",rho,svm->class[0],svm->class[1]);
    fclose(pFM);


    // Loading feature normalization files
    mean[cl]  = iftAllocFloatArray(nfeatures[cl]);
    stdev[cl] = iftAllocFloatArray(nfeatures[cl]);

    char fileNorm[200];
    sprintf(scl,"%d",cl);
    file=replace_str(NORM_NAME,"#",scl);
    sprintf(fileNorm,"%s.%d.data",file,nfeatures[cl]);
    free(file);

    FILE* pFn = fopen(fileNorm,"rb");      
    if ( (pFn == NULL) ) {
      sprintf(msg,"Can't open feature normalization file (%s)",fileNorm);
      iftError(msg,"LIV.IC.UNICAMP.MobILive.Verify");
    }
    if ( fread(mean[cl] ,sizeof(float),nfeatures[cl],pFn) != nfeatures[cl])
      iftError("Can't read mean normalization vector" ,"LIV.IC.UNICAMP.MobILive.Verify");
    if ( fread(stdev[cl],sizeof(float),nfeatures[cl],pFn) != nfeatures[cl])
      iftError("Can't read stdev normalization vector","LIV.IC.UNICAMP.MobILive.Verify");
    fclose(pFn);


    // fprintf(stdout,"rho: %f\n",rho[cl]);
    // fprintf(stdout,"feat: ");for(f=0;f<10;f++) fprintf(stdout,"%6.4f ",w[cl].feat[f]); fprintf(stdout,"\n");
    // fprintf(stdout,"mean: ");for(f=0;f<10;f++) fprintf(stdout,"%6.4f ",mean[cl][f]);   fprintf(stdout,"\n");
    // fprintf(stdout,"std : ");for(f=0;f<10;f++) fprintf(stdout,"%6.4f ",stdev[cl][f]);  fprintf(stdout,"\n");
  }
      

  float *weight = iftAllocFloatArray(nCNNs+1); // last vector position is used by the combiner
  int   *label  = iftAllocIntArray(nCNNs+1);   // last vector position is used by the combiner
  int   votes[2];

  // Open 'classification.csv' file
  FILE* pFcsv = fopen("classification.csv","wt");
  if ( pFcsv == NULL) {
    iftError("Impossible to create the 'classification.csv' file!","main");
  }
#ifndef _SILENCE
  fprintf(stdout,"in path %s, there are %d images to be processed\n",directory,nimages);
#endif

  for (i = 0; i <nimages ; i++) {
    //#ifndef _SILENCE
    //    fprintf(stdout,"(%d/%d): %s\n",i+1,nimages,img_names[i].image_name);
    //#endif

    // Reading TIFF Image and converting it to PGM
    sprintf(fullpath,"%s/%s",directory,img_names[i].image_name);
    iftImage* imgIn = TIFFRGB2IFT(fullpath);
    if (imgIn) {
      if (iftIsColorImage(imgIn)) {
	iftImage* imgTmp=iftImageGray(imgIn);
	iftDestroyImage(&imgIn);
	imgIn=imgTmp;
      }
    }
    else {
      fprintf(stderr,"error while reading %s image - deciding for Fake\n",fullpath);
      fprintf(pFcsv ,"%s\tFake\n",img_names[i].image_name);
      continue;
    }
    
    // feature extraction
    iftMImage *input,*output;
    if (imgIn->maxval <= 255 )
      input  = iftImageToMImage(imgIn, GRAY_CSPACE);
    else
      input  = iftImageToMImage(imgIn, GRAY_CSPACE); // to be implemented
    iftDestroyImage(&imgIn);

    for(cl=0;cl<nCNNs;cl++) {
      output = iftApplyConvNetwork(input, convnet[cl]);

      iftFeatures *features = iftMImageToFeatures(output);
      iftDestroyMImage(&output);

      // Feature normalization
      for ( f = 0 ; f < nfeatures[cl]; f++) { 
	features->val[f] = features->val[f] - mean[cl][f];
	if (stdev[cl][f]> Epsilon)
	  features->val[f] /= stdev[cl][f];
      }

      // Apply classification
      float maxpred, prediction[2];
      int imaxpred;

      prediction[0] = 0.;
      for( f = 0; f < nfeatures[cl]; f++) 
	prediction[0] += w[cl].feat[f] * features->val[f];
      iftDestroyFeatures(&features);

      prediction[0] -= rho[cl];

      prediction[1] = -prediction[0];

      if (prediction[0] > prediction[1] )
	imaxpred = 0;
      else
	imaxpred = 1;

      if (cl == 0) { weight[nCNNs] = 0.; votes[0] = 0; votes[1] = 0;}

      label[cl]  = class[cl][imaxpred];
      weight[cl] = prediction[imaxpred];
      //fprintf(stdout,"%6.4f %6.4f %6.4f\t",prediction[0],prediction[1],weight[cl]);

      // class 1 is positive and class 2 is negative, by definition.
      if (label[cl] != 1) weight[cl] = -weight[cl];

#ifndef _SILENCE
      fprintf(stdout,"%7.4f(%d) ",weight[cl],label[cl]);
#endif

      switch (COMB_RULE) {
      case COMB_SUM:
	weight[nCNNs] += weight[cl];
	break;
      case COMB_MAX:
	if (fabs(weight[nCNNs]) < fabs(weight[cl])) weight[nCNNs] = weight[cl];
	break;
      case COMB_MV:
	if (label[cl] == 1)
	  votes[0] += 1.;
	else
	  votes[1] += 1.;
	break;
      }
    }
    iftDestroyMImage(&input);

    //    define label
    switch (COMB_RULE) {
    case COMB_SUM:
      if (weight[nCNNs] > 0) label[nCNNs] = 1;
      else                   label[nCNNs] = 2;
      break;
    case COMB_MAX:
      if (weight[nCNNs] > 0) label[nCNNs] = 1;
      else                   label[nCNNs] = 2;
      break;
    case COMB_MV:
      if (votes[0] > votes[1]) label[nCNNs] = 1;
      else                     label[nCNNs] = 2;
      break;
    }

#ifndef _SILENCE
    switch (COMB_RULE) {
    case COMB_SUM:
    case COMB_MAX:
      fprintf(stdout,"%7.4f(F%d) "   ,weight[nCNNs]        ,label[nCNNs]); break;
    case COMB_MV:
      fprintf(stdout,"  %3d (F%d)   ",votes[label[nCNNs]-1],label[nCNNs]); break;
    }
#endif

    if (label[nCNNs] == 1) {
      fprintf(pFcsv ,"%s\tFake\n",img_names[i].image_name);
      fprintf(stdout,"%s\tFake\n",img_names[i].image_name);
    } else { // (label[nCNNs] == 2)
      fprintf(pFcsv ,"%s\tReal\n",img_names[i].image_name);
      fprintf(stdout,"%s\tReal\n",img_names[i].image_name);
    }
  }

  fclose(pFcsv);

  iftDestroyImageNames(img_names);

  for(cl=0;cl<nCNNs;cl++) {
    iftDestroyConvNetwork(&(convnet[cl]));

    free(w[cl].feat);
    free(class[cl]);

    free(stdev[cl]);
    free(mean[cl]);
  }
  free(convnet);
  free(nfeatures);

  free(w);free(rho);free(class);

  free(stdev);free(mean);

  free(label);
  free(weight);

  return 0;
}
