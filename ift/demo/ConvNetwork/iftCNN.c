#include "ift.h"


void iftApplyConvNetworkToImages(char *input_dir, iftConvNetwork *convnet, char *output_dir)
{
  int             number_of_images;
  iftImageNames  *image_names;

  DIR *dir=NULL;

  dir = opendir(output_dir);
  if (dir == NULL){
    char command[200];
    sprintf(command,"mkdir %s",output_dir);
    if (system(command)!=0)
      iftError("Could not open directory","iftApplyConvNetworkToImages");
  }else{
    closedir(dir);
  }

  number_of_images  = iftCountImageNames(input_dir, "mig");
  image_names       = iftCreateAndLoadImageNames(number_of_images, input_dir, "mig");

#pragma omp parallel for shared(number_of_images,image_names,input_dir,output_dir,convnet)
  for (int s = 0; s < number_of_images ; s++){
    char filename[200];
    sprintf(filename,"%s/%s",input_dir,image_names[s].image_name);
    fprintf(stdout,"Processing %s\n",filename);
    iftMImage *input  = iftReadMImage(filename);
    iftMImage *output = iftApplyConvNetwork(input,convnet);
    sprintf(filename,"%s/%s",output_dir,image_names[s].image_name);
    iftWriteMImage(output,filename);
    iftDestroyMImage(&input);
    iftDestroyMImage(&output);
  }

  iftDestroyImageNames(image_names);
}

int main(int argc, char **argv) 
{
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *convnet;


  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=4)
    iftError("Usage: iftCNN <input_directory> <input_parameters.convnet> <output_dir>","main");

  iftRandomSeed(IFT_RANDOM_SEED);

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  if (strcmp(ext,"convnet")==0){
    convnet = iftReadConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  t1 = iftTic();
  iftApplyConvNetworkToImages(argv[1],convnet,argv[3]);
  t2 = iftToc();
  fprintf(stdout,"Convnet applied in %f ms\n",iftCompTime(t1,t2));

  iftDestroyConvNetwork(&convnet);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
