#include "ift.h"

// iftTestV1likeKernels 43 43 16 6 2 3 4 6 11 18

int main(int argc, char** argv) {
  if (argc <= 6) {
    fprintf(stderr,"use: %s <sizex> <sizey> <norients> <nfreqs> <dfreq_1> <dfreq_2> ... <defreq_nfreqs>\n",argv[0]);
    return -1;
  }

  int sizex = atoi(argv[1]);
  int sizey = atoi(argv[2]);
  int norients = atoi(argv[3]);
  int nfreqs = atoi(argv[4]);

  float* freqs = iftAllocFloatArray(nfreqs);
  if (argc != 5+nfreqs) {
    fprintf(stderr,"number of dfreqs argument is incorrect!\n");
    return -2;
  }
  for (int f=0;f<nfreqs; f++)
    freqs[f] = 1./atof(argv[5+f]);


  iftMMKernel* v1like = iftV1likeMMKernel2D(sizex,sizey,1,norients,nfreqs,freqs);

  float max=-1.,min=1.;
  for (int k=0; k < norients*nfreqs; k++) {
    for (int i=0; i < v1like->A->n; i++) {
      if (v1like->weight[k][0].val[i] > max) max = v1like->weight[k][0].val[i];
      if (v1like->weight[k][0].val[i] < min) min = v1like->weight[k][0].val[i];
    }
    //    fprintf(stdout,"kernel: %2d (%f,%f)\n",k,min,max);
  }

  // creating and normalizing output image
  iftImage* img = iftCreateImage(sizex*norients,sizey*nfreqs,1);
  for (int k=0; k < norients*nfreqs; k++) {
    for (int i=0; i < v1like->A->n; i++) {
      img->val[(k/norients)*sizex*sizey*norients + (k%norients)*sizex + (v1like->A->dy[i]+sizey/2)*sizex*norients + v1like->A->dx[i]+sizex/2 ]
        = (int)round( 255.*(v1like->weight[k][0].val[i]-min)/(max-min) ); 
    }
  }
  char tmp[300];
  sprintf(tmp,"ift-v1like-%dx%d.pgm",sizex,sizey);
  iftWriteImageP5(img,tmp);

  free(freqs);
  iftDestroyMMKernel(&v1like);
}
