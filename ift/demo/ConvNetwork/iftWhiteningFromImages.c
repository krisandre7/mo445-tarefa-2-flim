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
  int             xsize, ysize, zsize, nbands,npatches,whitening,target_layer;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=7)
    iftError("Usage: iftWhiteningFromImages <input_dir> <output_dir|NULL> <input.convnet> <target_layer> <output.convnet> <npatches>","main");

  pos = strrchr(argv[3],'.') + 1;
  sscanf(pos,"%s",ext);

  iftRandomSeed(IFT_RANDOM_SEED);

  if (strcmp(ext,"convnet")==0){
    input_convnet = iftReadConvNetwork(argv[3]);
  }else{
    fprintf(stderr,"Invalid file format: %s\n",ext);
    exit(-1);
  }

  if (input_convnet->nlayers != 1) 
    iftError("Input CNN must have a single layer","main");
  
  t1 = iftTic();
  
  strcpy(input_dir,argv[1]);
  strcpy(output_dir,argv[2]);
  target_layer = atoi(argv[4]);
  npatches = atoi(argv[6]);

  iftWhiteningFromImages(input_dir,input_convnet,target_layer,npatches);  
  if (strcmp(output_dir,"NULL") !=0 ) {
    iftApplyConvNetworkToImages(input_dir,input_convnet,output_dir, &xsize, &ysize , &zsize, &nbands);
#ifndef _SILENCE
    fprintf(stdout,"dimensions for the next CNN: xsize %d, ysize %d, zsize %d, nbands %d\n",xsize,ysize,zsize,nbands); 
#endif
  }
  input_convnet->with_weights = 1;
  iftWriteConvNetwork(input_convnet,argv[5]);

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
