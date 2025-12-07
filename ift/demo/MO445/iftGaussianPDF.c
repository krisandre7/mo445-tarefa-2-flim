#include "ift.h"

#define STRIDE 3

int main(int argc, char *argv[])
{
  iftImage        *pdf[4];
  iftDataSet      *Z;
  
  /* it creates a pdf image in the 2D x and y feature space. The pdf
     contains three 2D Gaussian functions using the identity
     covariance matrix and the same variance for the x and y features,
     as specified by the three respective mean and standard deviation
     values. It also generates an opf dataset with a given number of
     samples, using stride of 3, according to the provided pdf. */
  
  if (argc != 14)
    iftError("Usage: iftGaussianPDF <xsize> <ysize> <mean1-xcoord> <mean1-ycoord> <stdev1> <mean2-xcoord> <mean2-ycoord> <stdev2> <mean3-xcoord> <mean3-ycoord> <stdev3> <output_pdf.[pgm,png]> <output_opfdataset.zip>","main");

  iftVoxel mean[3];
  float    stdev[3];

  mean[0].x = atoi(argv[3]);
  mean[0].y = atoi(argv[4]);
  stdev[0]  = atof(argv[5]);
  mean[1].x = atoi(argv[6]);
  mean[1].y = atoi(argv[7]);
  stdev[1]  = atof(argv[8]);
  mean[2].x = atoi(argv[9]);
  mean[2].y = atoi(argv[10]);
  stdev[2]  = atof(argv[11]);
  mean[0].z = mean[1].z = mean[2].z = 0;

  pdf[0] = iftCreateImage(atoi(argv[1]),atoi(argv[2]), 1);
  for (int i=0; i <= 2; i++) {
    if (!iftValidVoxel(pdf[0],mean[i]))
      iftError("mean value is out of the image domain","main");
  }
  iftDestroyImage(&pdf[0]);

  for (int i=0; i <= 2; i++) 
    pdf[i] = iftCreateGaussian(atoi(argv[1]),atoi(argv[2]), 1, mean[i], stdev[i], 255);  

  pdf[3] = iftCreateImage(atoi(argv[1]),atoi(argv[2]), 1);
  
  for (int p=0; p < pdf[3]->n; p++){
    pdf[3]->val[p]  = pdf[0]->val[p] + pdf[1]->val[p] + pdf[2]->val[p];
  }

  int nsamples = 0;
  for (int p=0; p < pdf[3]->n; p=p+STRIDE){
    nsamples += pdf[3]->val[p];
  }
  
  Z      = iftCreateDataSet(nsamples,2);

  int s = 0;
  for (int i=0; i <= 2; i++) {
    for (int p = 0; p < pdf[i]->n; p=p+STRIDE) {
      for (int t=0; t < pdf[i]->val[p]; t++,s++) {
	Z->sample[s].truelabel = i+1;
	Z->sample[s].weight    = pdf[i]->val[p]/255.0;
	iftVoxel u = iftGetVoxelCoord(pdf[i],p);
	Z->sample[s].feat[0] = u.x;
	Z->sample[s].feat[1] = u.y;
	Z->sample[s].id      = p;
      }
    }
  }
  Z->nclasses      = 3;
  Z->ref_data      = pdf[3];
  Z->ref_data_type = IFT_REF_DATA_IMAGE;

  iftSampler* sampler = iftRandomSubsampling(Z->nsamples, 10, 1000);
  iftSampleDataSet(Z, sampler, 5);
  iftDataSet* Zsmall  = iftExtractSamples(Z,IFT_TRAIN);
  iftSetStatus(Zsmall,IFT_TEST);
  
  iftDestroyDataSet(&Z);
  iftDestroySampler(&sampler);
  
  iftWriteOPFDataSet(Zsmall,argv[13]);
  iftWriteImageByExt(pdf[3],argv[12]);

  for (int i=0; i < 4; i++)    
    iftDestroyImage(&pdf[i]);

  iftDestroyDataSet(&Zsmall);
  
  return(0);
}




