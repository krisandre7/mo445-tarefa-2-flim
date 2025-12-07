#include "ift.h"

#define NLAYERS 2

int main(int argc, char **argv) 
{
  iftImage       *prob;
  iftMImage      *input[NLAYERS], *output[NLAYERS];
  char            filename[200];
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *convnet, *input_convnet;
  int number_of_images;
  iftImageNames *image_names;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=4)
    iftError("Usage: iftUnsupLearnKernels <input directory> <input_parameters.convnet> <output directory>","main");

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  iftRandomSeed(IFT_RANDOM_SEED);

  if (strcmp(ext,"convnet")==0){
    input_convnet = iftReadConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  number_of_images  = iftCountImageNames(argv[1], "mig");
  image_names       = iftCreateAndLoadImageNames(number_of_images, argv[1], "mig");

  DIR *dir=NULL;
  dir = opendir(argv[3]);
  if (dir == NULL) {
    char command[200];
    sprintf(command,"mkdir %s",argv[3]);
    if (system(command)!=0) 
      iftError("Could not open directory","main");
  } else
    closedir(dir);

  t1     = iftTic(); 

  for (int s=0; s < number_of_images; s++) {

    sprintf(filename,"%s/%s",argv[1],image_names[s].image_name);
    input[0] = iftReadMImage(filename);

    convnet = iftCreateConvNetwork(1);
    convnet->input_norm_adj_param = input_convnet->input_norm_adj_param;
    convnet->input_xsize          = input_convnet->input_xsize;
    convnet->input_ysize          = input_convnet->input_ysize;
    convnet->input_zsize          = input_convnet->input_zsize;
    convnet->input_nbands         = input_convnet->input_nbands;
    convnet->k_bank_adj_param[0]  = input_convnet->k_bank_adj_param[0];
    convnet->nkernels[0]          = input_convnet->nkernels[0];
    convnet->pooling_adj_param[0] = input_convnet->pooling_adj_param[0];
    convnet->stride[0]            = input_convnet->stride[0];
    convnet->alpha[0]             = input_convnet->alpha[0];
    convnet->norm_adj_param[0]    = input_convnet->norm_adj_param[0];
    convnet->rescale              = 0;
    convnet->with_weights         = 0;
    iftCreateRandomKernelBanks(convnet);
    iftCreateAdjRelAlongNetwork(convnet);
    iftMImageIndexMatrices(convnet);

    for (int l=0; l < NLAYERS; l++) {

      if (l>0) {
	input[l] = output[l-1];
      }
    
      prob   = iftBorderProbImage(input[l]);

//      iftUnsupLearnKernels(input[l],prob,convnet,2000,0.05,1);
      iftUnsupLearnKernels(input[l],convnet,2000,0.05,1);
      iftDestroyImage(&prob);
    
      output[l] = iftApplyConvNetwork(input[l],convnet);  
      convnet->input_norm_adj_param=0.0;
      convnet->input_xsize=output[l]->xsize;
      convnet->input_ysize=output[l]->ysize;
      convnet->input_zsize=output[l]->zsize;
      convnet->input_nbands=output[l]->m;
      sprintf(filename,"output-layer-%d.convnet",l);
      iftWriteConvNetwork(convnet,filename);  
      iftDestroyConvNetwork(&convnet);
      convnet = iftReadConvNetwork(filename);
    }
    
    iftAdjRel *A=iftCircular(sqrtf(2.0));  
    iftImage  *basins=iftMImageBasins(output[NLAYERS-1],A);
    sprintf(filename,"%s/%s",argv[3],image_names[s].image_name);
    iftMImage *fbasins = iftImageToMImage(basins,GRAY_CSPACE);
    iftWriteMImage(fbasins,filename);

    for (int l=0; l < NLAYERS; l++) 
      iftDestroyMImage(&output[l]);

    iftDestroyMImage(&input[0]);
    
    iftDestroyAdjRel(&A);
    iftDestroyImage(&basins);
    iftDestroyMImage(&fbasins);
    iftDestroyConvNetwork(&convnet);

  }

  t2     = iftToc();
  fprintf(stdout,"Images created in %f ms\n",iftCompTime(t1,t2));


  iftDestroyImageNames(image_names);
  iftDestroyConvNetwork(&input_convnet);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
