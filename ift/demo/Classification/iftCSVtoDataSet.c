#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftDataSet      *Z=NULL;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  int MemDinInicial, MemDinFinal;
  MemDinInicial = iftMemoryUsed();

  /*--------------------------------------------------------*/

  if (argc!=3)
    iftError("Usage: iftCSVtoDataSet <input.csv> <output.zip>","main");

  t1     = iftTic();

  iftCSV *csv      = iftReadCSV(argv[1],',');
  int     nfeats   = csv->ncols-1;
  int     nsamples = csv->nrows;

  Z  = iftCreateDataSet(nsamples,nfeats);

  printf("Total number of samples  %d\n",Z->nsamples);
  printf("Total number of features %d\n",Z->nfeats);
  
  for (int row=0, s=0; row < csv->nrows; row++, s++){
    Z->sample[s].truelabel = atoi(csv->data[row][0])+1;
    if (Z->nclasses < Z->sample[s].truelabel)
      Z->nclasses = Z->sample[s].truelabel;
    /* Important normalization: Taylla */
    float max=0;
    for (int col=1; col < csv->ncols; col++){
      Z->sample[s].feat[col-1] = atof(csv->data[row][col]);
      if (Z->sample[s].feat[col-1]>max)
	max = Z->sample[s].feat[col-1];      
    }
    for (int col=1; col < csv->ncols; col++){
      Z->sample[s].feat[col-1]/=max;
    }    
  }
  iftSetStatus(Z, IFT_TRAIN);
  iftAddStatus(Z, IFT_SUPERVISED);
  iftWriteDataSet(Z,argv[2]);
  
  t2     = iftToc();


  fprintf(stdout,"DataSet converted in %f ms\n",iftCompTime(t1,t2));

  iftDestroyDataSet(&Z);
  
  /* ---------------------------------------------------------- */

  MemDinFinal = iftMemoryUsed();
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




