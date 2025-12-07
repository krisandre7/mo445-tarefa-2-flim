#include <ift.h>


iftDataSet *iftExtractFacesFeatures()
{
  char         command[200], filename[50], class[4];
  FILE        *fp;
  int          i,j,nsamples;
  iftAdjRel   *A;
  iftImage    *roi;
  iftFeatures *feat;
  iftDataSet  *Z;

  sprintf(command,"ls -v data/rois/*.jpg > temp.txt");
  system(command);
  fp = fopen("temp.txt","r");
  nsamples  = 0; 
  while (!feof(fp)){
    fscanf(fp,"%s",filename);
    nsamples++;
  }
  nsamples--;
  fclose(fp);

  A = iftCircular(sqrtf(2.0));
  Z = iftCreateDataSet(nsamples,A->n-1);
  fp = fopen("temp.txt","r");
  Z->nclasses = 0;

  for (i=0; i < nsamples; i++) {

    fgets(filename,50,fp);
    filename[strlen(filename)-1]='\0';
    for (j=10; j < 13; j++) 
      class[j-10]=filename[j];
    class[3]='\0';
    Z->sample[i].truelabel = atoi(class);
    printf("sample[%d] class %d\n",i+1,Z->sample[i].truelabel);
    if (Z->sample[i].truelabel > Z->nclasses)
      Z->nclasses++;
    sprintf(command,"convert %s temp.pgm",filename);
    system(command);
    roi   = iftReadImageP5("temp.pgm");    
    feat  = iftExtractLBP(roi,A);    
    for (j=0; j < Z->nfeats; j++) 
      Z->sample[i].feat[j] = feat->val[j];
    iftDestroyFeatures(&feat);
    iftDestroyImage(&roi);

  }
  fclose(fp);
  system("rm -f temp.txt temp.pgm");
  iftDestroyAdjRel(&A);

  return(Z);
}

int main(int argc, char *argv[]) 
{
  iftDataSet      *Z;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=1)
    iftError("Usage: iftExtractFacesFeatures","main");

  t1 = iftTic();

  Z  = iftExtractFacesFeatures();

  t2 = iftToc(); 
  
  fprintf(stdout,"Features extracted in %f ms\n",iftCompTime(t1,t2));
  
  iftWriteOPFDataSet(Z,"data/FacesDataSet.opf");

  iftDestroyDataSet(&Z);
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




