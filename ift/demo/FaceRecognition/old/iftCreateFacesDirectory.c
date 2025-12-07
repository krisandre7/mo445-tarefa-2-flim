#include <ift.h>

int main(int argc, char *argv[]) 
{
  char             command[200], individual[50], *basename=NULL, imgname[30];
  FILE            *fp1, *fp2, *fp3;
  int              class,sample,nclasses,nsamples,i;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=2)
    iftError("Usage: iftCreateFacesDirectory <directory with face images>","main");

  t1 = iftTic();
  sprintf(command,"mkdir data");
  system(command);
  sprintf(command,"mkdir data/images");
  system(command);
  sprintf(command,"mkdir data/training");
  system(command);
  sprintf(command,"mkdir data/testing");
  system(command);
  sprintf(command,"mkdir data/rois");
  system(command);
  sprintf(command,"ls -v %s > data/individuals.txt",argv[1]);
  system(command);
  fp1 = fopen("data/individuals.txt","r");
  nclasses  = 0;
  while(!feof(fp1)){
    fscanf(fp1,"%s",individual);
    nclasses++;
  }
  nclasses--;
  fclose(fp1);
  fp1 = fopen("data/individuals.txt","r");
  fp3 = fopen("data/data.txt","w");
  fprintf(fp3,"%s ",argv[1]);
  fprintf(fp3,"%d\n",nclasses);
  
  for (i=0,class=1; i < nclasses; i++,class++) {
    fscanf(fp1,"%s",individual);
    fprintf(fp3,"%03d %s ",class,individual);
    sprintf(command,"ls -v %s/%s > temp",argv[1],individual); 
    system(command); 
    fp2 = fopen("temp","r");
    nsamples  = 0;
    while(!feof(fp2)){
      fscanf(fp2,"%s",imgname);
      nsamples++;
    }
    nsamples--;
    fclose(fp2);
    fp2 = fopen("temp","r");
    fprintf(fp3,"%d \n",nsamples);
    for (sample=1; sample <= nsamples; sample++) {
      fscanf(fp2,"%s",imgname);
      basename = strtok(imgname,"."); 
      fprintf(fp3,"%s ",basename);
      fprintf(fp3,"%03d_%03d\n",class,sample);
      sprintf(command,"cp %s/%s/%s.jpg data/images/%03d_%03d.jpg",argv[1],individual,imgname,class,sample); 
      system(command);
    }
    fclose(fp2);
  }
  fclose(fp1);
  fclose(fp3);

  system("rm -f temp");

  t2 = iftToc(); 
  
  fprintf(stdout,"Data directory was created in %f ms\n",iftCompTime(t1,t2));
  

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




