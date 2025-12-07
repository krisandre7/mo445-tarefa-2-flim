#include "ift.h"

iftMImage *iftCreateMImageWithLinearConvolutionOfGaussianFiltersOfAnImage(iftImage *img,int nscales){

  iftMImage *temp_mimg;
  if (iftIsColorImage(img))
    temp_mimg = iftImageToMImage(img, LABNorm2_CSPACE);
  else
    temp_mimg=iftImageToMImage(img,GRAYNorm_CSPACE);

  iftMImage *out_mimg= iftCreateMImage(img->xsize,img->ysize,img->zsize,nscales*temp_mimg->m);

  /* create the gaussian kernels */
  iftKernel ** gaussian_kernels=iftAlloc(nscales,sizeof(iftKernel *));
  for (int k=1;k<=nscales;k++){
    gaussian_kernels[k-1]=iftGaussianKernel2D((float)k,(float)k/3);
  }

  /* do the convolution with the gaussian kernels*/
#pragma omp parallel for
  for (int p=0;p<temp_mimg->n;p++){
    iftVoxel u=iftMGetVoxelCoord(temp_mimg,p);
    iftVoxel v;
    iftAdjRel *A;
    int q;
    for (int k=0;k<nscales;k++){
      A=gaussian_kernels[k]->A;
      for (int i=0;i<A->n;i++){
        v=iftGetAdjacentVoxel(A,u,i);
        if (iftMValidVoxel(temp_mimg,v)){
          q=iftMGetVoxelIndex(temp_mimg,v);
          for (int m=0;m<temp_mimg->m;m++){
            out_mimg->band[k*temp_mimg->m+m].val[p]+=gaussian_kernels[k]->weight[i]*temp_mimg->val[q][m];
          }
        }
      }
    }
  }

  for (int k=0;k<nscales;k++)
    iftDestroyKernel(&gaussian_kernels[k]);
  iftFree(gaussian_kernels);
  iftDestroyMImage(&temp_mimg);

  return out_mimg;
}


int main(int argc, char *argv[])
{
  iftImage        *img=NULL;
  iftDataSet      *Z=NULL;
  iftAdjRel       *A=NULL;

  if (argc !=4)
    iftError("Usage: iftImage2DDatasetVisualization <image.ppm(pgm)> <label_image.pgm> <nb_samples>","main");

  img=iftReadImageByExt(argv[1]);

  /* convert the image to multi-image*/
  iftMImage *mimg;
  iftMImage *eimg;

  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAYNorm_CSPACE);
    A  = iftCircular(sqrtf(2.0));
    printf("Adjacency size -> %d\n",A->n);
    eimg = iftExtendMImageByAdjacencyAndVoxelCoord(mimg,A,1);
    iftDestroyAdjRel(&A);
    iftDestroyMImage(&mimg);
    mimg = eimg;
  }
  else {
    mimg = iftImageToMImage(img,LABNorm2_CSPACE);
//    mimg = iftCreateMImageWithLinearConvolutionOfGaussianFiltersOfAnImage(img,3);
    eimg = iftExtendMImageByVoxelCoord(mimg, 1);
    iftDestroyMImage(&mimg);
    mimg = eimg;
  }

  /* create a dataset with the image*/
  Z= iftMImageToDataSet(mimg);
  printf("Created a dataset with %d features\n",Z->nfeats);
  iftSelectUnsupTrainSamples(Z,atof(argv[3]),5);

  /* read the gt image*/
  iftImage *gt=NULL;
  gt=iftReadImageByExt(argv[2]);
  iftImageGTToDataSet(gt,Z);

  printf("The number of classes are %d\n",Z->nclasses);

  /* create a new dataset with the training samples*/
  iftDataSet *Z_train=iftCreateDataSet(Z->ntrainsamples,Z->nfeats);
  Z_train->nclasses=Z->nclasses;
  Z_train->ref_data=Z->ref_data;
  Z_train->ref_data_type=Z->ref_data_type;

  int n=0;
  for (int p=0;p<Z->nsamples;p++)
    if (Z->sample[p].status==IFT_TRAIN){
      iftCopySample(&Z->sample[p],&Z_train->sample[n],Z->nfeats,true);
      n++;
    }

  iftSetStatus(Z_train,IFT_TEST);

//  iftWriteOPFDataSet(Z_train,"opf_dataset.zip");
  iftDataSet* Z1 = iftDimReductionByTSNE(Z_train, 2, 40, 1000, false);

  iftImage *outputImage = iftDraw2DFeatureSpace(Z1,CLASS,0);
  iftWriteImageP6(outputImage,"projection.ppm");

  iftDestroyDataSet(&Z_train);
  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&Z1);
  iftDestroyImage(&outputImage);
  iftDestroyImage(&gt);

  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);
  iftDestroyMImage(&mimg);

  return(0);
}
