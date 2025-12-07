#include <ift.h>

int iftNumberOfFaceSamples(char *dirname)
{
  char  command[200],  filename[200];
  FILE *fp;
  int nsamples;

  sprintf(command,"ls -v %s/images > temp.txt",dirname);
  system(command);
  fp = fopen("temp.txt","r");
  nsamples = 0;
  while(!feof(fp)){
    fscanf(fp,"%s",filename);
    nsamples++;
  }
  nsamples--;
  system("rm -f temp.txt");
  fclose(fp);

  return(nsamples);
}

int iftNumberOfFaceClasses(char *dirname)
{
  char  command[200], filename[200], *token;
  FILE *fp;
  int   nclasses, class, prev_class;

  sprintf(command,"ls -v %s/images > temp.txt",dirname);
  system(command);
  fp = fopen("temp.txt","r");
  nclasses = 0; prev_class = NIL;
  while(!feof(fp)){
    fscanf(fp,"%s",filename);
    token = strtok(filename,"_");
    class = atoi(token);
    if (class != prev_class){
      nclasses++;
      prev_class = class;
    }
  }
  
  system("rm -f temp.txt");
  fclose(fp);
  
  return(nclasses);
}

int iftNumberOfFaceSamplesPerClass(char *dirname, int class)
{
  char  command[200], filename[200], *token;
  FILE *fp;
  int   nsamples_per_class, cur_class=-1;

  sprintf(command,"ls -v %s/images > temp.txt",dirname);
  system(command);
  fp = fopen("temp.txt","r");
  nsamples_per_class = 0; 
  while(!feof(fp)){
    fscanf(fp,"%s",filename);
    token = strtok(filename,"_");
    cur_class = atoi(token);
    if (class == cur_class){
      nsamples_per_class++;
    }
  }
  if (class == cur_class)
    nsamples_per_class--;
  
  system("rm -f temp.txt");
  fclose(fp);
  
  return(nsamples_per_class);
}

void iftSplitFacesData(char *dirname, float train_perc)
{
  int    class, nclasses, nsamples, i, ntrainsamples;
  char   command[200], filename[200];
  uchar *sample;
  
  if ((train_perc<=0.0)||(train_perc>=1.0))
    iftError("Invalid percentage of training samples","iftSplitFacesData");


  nclasses = iftNumberOfFaceClasses(dirname);

  iftRandomSeed(IFT_RANDOM_SEED);
  
  for (class=1; class <= nclasses; class++) {
    nsamples=iftNumberOfFaceSamplesPerClass(dirname, class);
    sample  = iftAllocUCharArray(nsamples);
    for (i=0; i < nsamples; i++) 
      sample[i]=1;
    ntrainsamples = (int)(nsamples*train_perc); 

    /* Create training set */
    
    while (ntrainsamples > 0) {
      i = iftRandomInteger(0,nsamples-1);
      if (sample[i]==1) {
	sprintf(filename,"%03d_%03d.jpg",class,i+1);
	sprintf(command,"cp -f %s/images/%s %s/training/%s",dirname,filename,dirname,filename);
	system(command);
	sample[i] = 0;
      }
      ntrainsamples--;
    }

    /* Create testing dataset */

    for (i=0; i < nsamples; i++) {
      if (sample[i]==1){
	sprintf(filename,"%03d_%03d.jpg",class,i+1);
	sprintf(command,"cp -f %s/images/%s %s/testing/%s",dirname,filename,dirname,filename);
	system(command);
	sample[i]=0;
      }
    }
    free(sample);
  }
  
}

int main(int argc, char *argv[]) 
{
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  
  if (argc!=3)
    iftError("Usage: iftSplitFacesData <directory with face images> <train_perc>","main");


  t1 = iftTic();

  iftSplitFacesData(argv[1],atof(argv[2]));

  t2 = iftToc(); 
  
  fprintf(stdout,"Training and test sets created in %f ms\n",iftCompTime(t1,t2));
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




