#include <ift.h>


void iftEncodeImages(char *dirname, iftKnnGraph *graph, iftAdjRel *A)
{
  char        command[200],filename[100];
  FILE       *fp;
  int         i,s,nimages=0;
  iftImage   *img,*eimg;
  iftDataSet *Z;


  /* Count number of training images */

  sprintf(command,"ls -v data/images > temp.txt");
  system(command);
  fp = fopen("temp.txt","r");
  while(!feof(fp)){
    fscanf(fp,"%s",filename);
    nimages++;
  }
  nimages--;
  fclose(fp);

  /* Encode images */

  fp = fopen("temp.txt","r");
  for (i=0; i < nimages; i++) {

    /* Read image */

    fscanf(fp,"%s",filename);
    sprintf(command,"convert %s/images/%s temp.pgm",dirname,filename);
    system(command);
    img = iftReadImageP5("temp.pgm");    
    system("rm -f temp.pgm");
    
    /* Create Image DataSet */

    Z = iftImageToDataSetUsingAdjacency(img,A);

    /* Propagate cluster labels */

    iftUnsupClassify(graph,Z);

    /* Create encoded image */

    eimg = iftCreateImage(img->xsize,img->ysize,img->zsize);
    for (s=0; s < Z->nsamples; s++) 
      eimg->val[Z->sample[s].id] = Z->sample[s].label;

    iftWriteImageP2(eimg,"temp.pgm");
    sprintf(command,"convert temp.pgm %s/codebook/%s",dirname,filename);
    system(command);
    
    iftDestroyDataSet(&Z);
    iftDestroyImage(&img);
    iftDestroyImage(&eimg);
  }

  system("rm -f temp.txt");
  fclose(fp);

}


int main(int argc, char *argv[]) 
{
  char             dirname[200];
  iftDataSet      *Z;
  iftAdjRel       *A;
  iftKnnGraph     *graph;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  
  if (argc!=5)
    iftError("Usage: iftUnsupFeatLearn <directory with face images> <number of samples per image> <perc. for training> <perc. for kmax>","main");

  t1 = iftTic();

  A = iftCircular(sqrtf(2.0));
  
  sprintf(dirname,"%s/training",argv[1]); 
  Z = iftPGMKernelDataSet(dirname,A,atoi(argv[2]));
  printf("nsamples %d\n",Z->nsamples);
  graph = iftUnsupLearn(Z,atof(argv[3]),atof(argv[4]),iftNormalizedCut,3);
  iftUnsupClassify(graph,Z);
  printf("nclusters %d\n",Z->nlabels);
  iftEncodeImages(argv[1],graph,A);

  t2 = iftToc(); 
  
  fprintf(stdout,"Unsupervised feature learning in %f ms\n",iftCompTime(t1,t2));
  
  iftDestroyDataSet(&Z);
  iftDestroyAdjRel(&A);
  iftDestroyKnnGraph(&graph);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




