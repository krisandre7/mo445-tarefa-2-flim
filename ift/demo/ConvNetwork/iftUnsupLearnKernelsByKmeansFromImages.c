#include "ift.h"

#define _SILENCE

void iftApplyConvNetworkToImages(char *input_dir, iftConvNetwork *convnet, char *output_dir, int *xsize,  int *ysize , int *zsize,  int *nbands)
{
  int number_of_images;
  iftImageNames *image_names;

  number_of_images  = iftCountImageNames(input_dir, "mig");
  image_names       = iftCreateAndLoadImageNames(number_of_images, input_dir, "mig");

  char filename[200];
  sprintf(filename,"%s/%s",input_dir,image_names[0].image_name);
  iftMImage *input   = iftReadMImage(filename);
  iftMImage *output  = iftApplyConvNetwork(input,convnet);

  DIR *dir=NULL;
  dir = opendir(output_dir);
  if (dir == NULL) {
    char command[200];
    sprintf(command,"mkdir %s",output_dir);
    if (system(command)!=0) 
      iftError("Could not open directory","iftApplyConvNetworkToImages");
  } else
    closedir(dir);

  sprintf(filename,"%s/%s",output_dir,image_names[0].image_name);
  iftWriteMImage(output,filename);
  
  *xsize  = output->xsize;
  *ysize  = output->ysize;
  *zsize  = output->zsize;
  *nbands = output->m;

  iftDestroyMImage(&input);
  iftDestroyMImage(&output);

#pragma omp parallel for shared(number_of_images,image_names,input_dir,convnet,output_dir)
  for (int s = 1; s < number_of_images ; s++){
    char filename[200];
    sprintf(filename,"%s/%s",input_dir,image_names[s].image_name);
    fprintf(stdout,"Processing %s\n",filename);
    iftMImage *input   = iftReadMImage(filename);
    iftMImage *output  = iftApplyConvNetwork(input,convnet);
    sprintf(filename,"%s/%s",output_dir,image_names[s].image_name);
    iftWriteMImage(output,filename);
    iftDestroyMImage(&input);
    iftDestroyMImage(&output);
  }

  iftDestroyImageNames(image_names);

}

int main(int argc, char **argv) 
{
  char            ext[10],*pos, input_dir[200],output_dir[200];
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *input_convnet=NULL;
  int             xsize, ysize, zsize, nbands,npatches,whitening,tKm1,tKm2,target_layer;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=11)
    iftError("Usage: iftUnsupLearnKernelsByKmeansFromImages <seed> <input_dir> <output_dir|NULL> <input.convnet> <target_layer> <output.convnet> <whitening:0|1> <npatches> <Conventional(1)|Spherical(2)> <Kmeans(1)|Kmedoids(2)>","main");

  unsigned int seed = (unsigned int)atoi(argv[1]);
  iftRandomSeed(seed);

  pos = strrchr(argv[4],'.') + 1;
  sscanf(pos,"%s",ext);

  if (strcmp(ext,"convnet")==0){
    input_convnet = iftReadConvNetwork(argv[4]);
  }else{
    fprintf(stderr,"Invalid file format: %s\n",ext);
    exit(-1);
  }

  strcpy(input_dir,argv[2]);
  strcpy(output_dir,argv[3]);
  target_layer = atoi(argv[5]);
  whitening = atoi(argv[7]);
  npatches = atoi(argv[8]);
  tKm1 = atoi(argv[9]);
  tKm2 = atoi(argv[10]);

  if ( (target_layer <= 0) || (target_layer > input_convnet->nlayers) ) {
    char msg[300];
    sprintf(msg,"Input CNN must have at least %d single layer",target_layer);
    iftError(msg,"main");
  }

  t1 = iftTic();

  if (tKm1 == 1) {
    if (tKm2 == 1)
      iftUnsupLearnKernelsByKmeansFromImages  (input_dir,input_convnet,target_layer,npatches,input_convnet->nkernels[target_layer-1],whitening);
    else if  (tKm2 == 2)
      iftUnsupLearnKernelsByKmedoidsFromImages(input_dir,input_convnet,target_layer,npatches,input_convnet->nkernels[target_layer-1],whitening);
    else
      iftError("clustering method type unknown","iftUnsupLearnKernelsByKmeansFromImages");
  } else if (tKm1 == 2) {
    // tkm2 does'nt matter
    iftUnsupLearnKernelsBySpKmeansFromImages(input_dir,input_convnet,target_layer,npatches,input_convnet->nkernels[target_layer-1],whitening);
  } else
    iftError("clustering method type unknown","iftUnsupLearnKernelsByKmeansFromImages");

  if (strcmp(output_dir,"NULL") !=0 ) {
    iftApplyConvNetworkToImages(input_dir,input_convnet,output_dir, &xsize, &ysize , &zsize, &nbands);
#ifndef _SILENCE
    fprintf(stdout,"dimensions for the next CNN: xsize %d, ysize %d, zsize %d, nbands %d\n",xsize,ysize,zsize,nbands); 
#endif
  }
  input_convnet->with_weights = 1;
  iftWriteConvNetwork(input_convnet,argv[6]);

  t2 = iftToc();
#ifndef _SILENCE
  fprintf(stdout,"Kernels learned in %f ms\n",iftCompTime(t1,t2));
#endif


  iftDestroyConvNetwork(&input_convnet);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal) {
#ifndef _SILENCE
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif
  }


  return(0);
}
