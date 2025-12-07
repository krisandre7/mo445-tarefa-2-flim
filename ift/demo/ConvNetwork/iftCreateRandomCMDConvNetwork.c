#include "ift.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>

// for charplates600 database
int   vet_input_norm[8] = {3,5,7,9,11,13,0,0}; //
int   vet_k_bank    [8] = {3,5,7,9,11,13,7,9};
int   vet_nkernels  [4] = {32,64,96,128};
int   vet_pooling   [8] = {3,5,7,9,11,13,0,0}; //
int   vet_stride    [4] = {1,2,3,4};
float vet_alpha     [4] = {1.,2.,3.,10.};
int   vet_norm      [8] = {3,5,7,9,11,13,0,0}; //

// for charplates600 database
//int   vet_input_norm[8] = {3,5,7,9,11,13,0,0}; //
//int   vet_k_bank    [8] = {3,5,7,9,11,13,7,9};
//int   vet_nkernels  [4] = {32,64,96,128};
//int   vet_pooling   [8] = {3,5,7,9,11,13,0,0}; //
//int   vet_stride    [4] = {1,2,3,4};
//float vet_alpha     [4] = {1.,2.,3.,10.};
//int   vet_norm      [8] = {3,5,7,9,11,13,0,0}; //


// for ASLfs database
//int   vet_input_norm[8] = {0,3,0,5,0,7,0,9};
//int   vet_k_bank    [4] = {3,5,7,9};
//int   vet_nkernels  [3][4] = {{16,32, 64, 64},{32,64, 96,128},{32,64,128,256}};
//int   vet_pooling   [8] = {3,5,7,9};
//int   vet_stride    [4] = {2,2,2,2};
//float vet_alpha     [4] = {1.,2.,3.,10.};
//int   vet_norm      [8] = {3,5,7,9};

int iftIsConvNetInList(iftConvNetwork* convnet,iftConvNetwork** lconvnet, int nconvnets) {

  for(int c=0;c<nconvnets;c++) {
    if (convnet->nlayers != lconvnet[c]->nlayers)
      continue;

    if (convnet->input_norm_adj_param != lconvnet[c]->input_norm_adj_param)
      continue;

    int l;
    for(l=0; l < convnet->nlayers; l++) {
      if (convnet->k_bank_adj_param[l]  != lconvnet[c]->k_bank_adj_param[l] ) break;
      if (convnet->nkernels[l]          != lconvnet[c]->nkernels[l]         ) break;
      if (convnet->pooling_adj_param[l] != lconvnet[c]->pooling_adj_param[l]) break;
      if (convnet->stride[l]            != lconvnet[c]->stride[l]           ) break;
      if (convnet->alpha[l]             != lconvnet[c]->alpha[l]            ) break;
      if (convnet->norm_adj_param[l]    != lconvnet[c]->norm_adj_param[l]   ) break;
    }
    if (l == convnet->nlayers)
      return 1;
  }

  return 0;
}

int iftVerifyImageDimensionsAlongNetwork(iftConvNetwork *convnet)
{
  int bx, by, bz, l, i, j, k;
  iftMImage     *img;
  iftFastAdjRel *F;
  int ret = 1;

  /* Compute the number of bands at each stage of each layer along network */
  int* xsize  = iftAllocIntArray(convnet->nstages);
  int* ysize  = iftAllocIntArray(convnet->nstages);
  int* zsize  = iftAllocIntArray(convnet->nstages);
  int* nbands = iftAllocIntArray(convnet->nstages); 

  nbands[0] = convnet->input_nbands; 
  nbands[1] = convnet->input_nbands; 
  for (l=0, i=2, j=i+1, k=i+2; l < convnet->nlayers; l++,i=i+3,j=i+1,k=i+2){    
    nbands[i] = nbands[j] = nbands[k] = convnet->nkernels[l]; 
  }

  /* Compute the image dimensions at each stage of each layer along network */

  if (convnet->input_norm_adj_param > 0)
    iftMaxAdjShifts(convnet->input_norm_adj,&bx,&by,&bz);
  else
    bx = by = bz = 0;

  /* Before and after input normalization stage */

  xsize[0] = convnet->input_xsize; 
  ysize[0] = convnet->input_ysize;
  zsize[0] = convnet->input_zsize;

  xsize[1] = convnet->input_xsize - 2*bx; 
  ysize[1] = convnet->input_ysize - 2*by; 
  zsize[1] = convnet->input_zsize - 2*bz; 

  for (l=0, i=2, j=i+1, k=i+2; l < convnet->nlayers; l++,i=i+3,j=i+1,k=i+2){    
    
    /* After filtering */
    
    if ((nbands[i-1] <= 0) || (xsize[i-1] <= 1) || (ysize[i-1] <= 1) || (zsize[i-1] <= 0)) {
      ret = 0;break;
    } 

    img = iftCreateMImage(xsize[i-1],ysize[i-1],zsize[i-1],nbands[i-1]);
    F   = iftCreateFastAdjRel(convnet->k_bank[l]->A,img->tby,img->tbz);

    xsize[i] = xsize[i-1] - 2*F->bx; 
    ysize[i] = ysize[i-1] - 2*F->by; 
    zsize[i] = zsize[i-1] - 2*F->bz;

    if ((nbands[i] <= 0) || (xsize[i] <= 1) || (ysize[i] <= 1) || (zsize[i] <= 0)) {
      ret = 0;break;
    } 

    iftDestroyMImage(&img);
    iftDestroyFastAdjRel(&F);

    /* After pooling */

    if (convnet->pooling_adj_param[l] > 0)
      iftMaxAdjShifts(convnet->pooling_adj[l],&bx,&by,&bz);
    else
      bx = by = bz = 0;
      
    xsize[j] = xsize[i] - 2*bx; 
    ysize[j] = ysize[i] - 2*by; 
    zsize[j] = zsize[i] - 2*bz; 

    if ((nbands[j] <= 0) || (xsize[j] <= 1) || (ysize[j] <= 1) || (zsize[j] <= 0)) {
      ret = 0;break;
    } 

    if (convnet->stride[l] > 1){ // reduce scale 
      xsize[j] = ceilf(((float)xsize[j])/convnet->stride[l]);
      ysize[j] = ceilf(((float)ysize[j])/convnet->stride[l]);
      if (convnet->input_zsize != 1) // 3D
	zsize[j] = ceilf(((float)zsize[j])/convnet->stride[l]);
    }

    /* After normalization */

    if (convnet->norm_adj_param[l] > 0)
      iftMaxAdjShifts(convnet->norm_adj[l],&bx,&by,&bz);
    else
      bx = by = bz = 0;

    xsize[k] = xsize[j] - 2*bx; 
    ysize[k] = ysize[j] - 2*by; 
    zsize[k] = zsize[j] - 2*bz; 

    if ((nbands[k] <= 0) || (xsize[k] <= 1) || (ysize[k] <= 1) || (zsize[k] <= 0)) {
      ret = 0;break;
    } 
  }

  for (i=0; i < convnet->nstages; i++){    
    if ((nbands[i] <= 0) || (xsize[i] <= 1) || (ysize[i] <= 1) || (zsize[i] <= 0)) {
      ret = 0; break;
    }
  }

  if (nbands[convnet->nstages-1] * xsize[convnet->nstages-1] * ysize[convnet->nstages-1] * zsize[convnet->nstages-1] > 5000) {
    ret = 0;
  }

  free(xsize);free(ysize);free(zsize);free(nbands);

  return ret;
}


int iftGetFirstNBits(int* value,int nbits)
{
  int ret;

  ret = *value & (1<<(nbits))-1;
  *value = *value >> nbits;

  return ret;
}


int main(int argc, char **argv) {
  if (argc != 7)
    iftError("Please provide the following parameters:\n<random_seed> <xsize> <ysize> <nbands> <output_directory> <num_convnets>\n\n", "main");

  unsigned int random_seed = atoi(argv[1]);
  int xsize  = atoi(argv[2]);
  int ysize  = atoi(argv[3]);
  int nbands = atoi(argv[4]);

  char outputfile[100],dirArchOut[100]; strcpy(dirArchOut, argv[5]);
  if (dirArchOut[strlen(dirArchOut)-1] == '/')
    dirArchOut[strlen(dirArchOut)-1] = '\0';
  int nconvnets = atoi(argv[6]);

  srand(random_seed);

  iftConvNetwork *convnet,**lconvnet;
  int randvalue,nlayers,hlayers[4];

  for(int l=1;l<=3;l++) hlayers[l] = 0;

  lconvnet = (iftConvNetwork**)malloc(nconvnets*sizeof(iftConvNetwork*));
  if (lconvnet == NULL)
    iftError("allocating list of convnets","iftCreateRandomCMDConvNetwork");

  int ncurconvnet = 0,trials=0;
  while(ncurconvnet < nconvnets ) {
    trials++;

    randvalue = rand();
//    nlayers = iftGetFirstNBits(&randvalue,2);
//    if (nlayers < 1 || nlayers > 3)
//      continue;
    nlayers = 3;

    convnet = iftCreateConvNetwork(nlayers);

    convnet->input_norm_adj_param = vet_input_norm[iftGetFirstNBits(&randvalue,3)];

    for(int l=0; l < nlayers; l++) {
      randvalue = rand();
      convnet->k_bank_adj_param[l]  = vet_k_bank      [iftGetFirstNBits(&randvalue,3)];
      convnet->nkernels[l]          = vet_nkernels    [iftGetFirstNBits(&randvalue,2)];
      convnet->pooling_adj_param[l] = vet_pooling     [iftGetFirstNBits(&randvalue,3)];
      convnet->stride[l]            = vet_stride      [iftGetFirstNBits(&randvalue,2)];
      convnet->alpha[l]             = vet_alpha       [iftGetFirstNBits(&randvalue,2)];
      convnet->norm_adj_param[l]    = vet_norm        [iftGetFirstNBits(&randvalue,3)];

      if (convnet->pooling_adj_param[l] == 0) {
	convnet->stride[l] = 1;
	convnet->alpha[l]  = 1.;
      }
    }

    convnet->input_xsize     = xsize;
    convnet->input_ysize     = ysize;
    convnet->input_zsize     = 1;
    convnet->input_nbands    = nbands;
    convnet->rescale         = 0;
    convnet->with_weights    = 0;

    iftCreateRandomKernelBanks(convnet);
    iftCreateAdjRelAlongNetwork(convnet);

    if (  (iftVerifyImageDimensionsAlongNetwork(convnet) ) && 
	  (!iftIsConvNetInList(convnet,lconvnet,ncurconvnet)) ){
      /*
      iftWriteIntValue(stdout,convnet->nlayers,"NLAYERS");
      iftWriteIntValue(stdout,convnet->input_norm_adj_param,"INPUT_NORM_ADJ_PARAM");
      iftWriteIntValue(stdout,convnet->input_xsize,"INPUT_XSIZE");
      iftWriteIntValue(stdout,convnet->input_ysize,"INPUT_YSIZE");
      iftWriteIntValue(stdout,convnet->input_zsize,"INPUT_ZSIZE");
      iftWriteIntValue(stdout,convnet->input_nbands,"INPUT_NBANDS");
      iftWriteIntValues(stdout,convnet->k_bank_adj_param,convnet->nlayers,"K_BANK_ADJ_PARAM");
      iftWriteIntValues(stdout,convnet->nkernels,convnet->nlayers,"NKERNELS");
      iftWriteIntValues(stdout,convnet->pooling_adj_param,convnet->nlayers,"POOLING_ADJ_PARAM");
      iftWriteIntValues(stdout,convnet->stride,convnet->nlayers,"STRIDE");
      iftWriteFloatValues(stdout,convnet->alpha,convnet->nlayers,"ALPHA");
      iftWriteIntValues(stdout,convnet->norm_adj_param,convnet->nlayers,"NORM_ADJ_PARAM");
      iftWriteIntValue(stdout,convnet->rescale,"RESCALE");
      iftWriteIntValue(stdout,convnet->with_weights,"WITH_WEIGHTS");
      */
      hlayers[nlayers]++;
      sprintf(outputfile,"%s/random.%06d.convnet",dirArchOut,ncurconvnet+1);
      fprintf(stderr,"%s, trials:%d\n",outputfile,trials);

      iftWriteConvNetwork(convnet,outputfile);
      lconvnet[ncurconvnet] = iftReadConvNetwork(outputfile);
      // saving memory
      if (lconvnet[ncurconvnet]->input_norm_adj != NULL) iftDestroyAdjRel(&(lconvnet[ncurconvnet]->input_norm_adj));
      for (int l=0; l < lconvnet[ncurconvnet]->nlayers; l++) {
	if (lconvnet[ncurconvnet]->k_bank[l]      != NULL) iftDestroyMMKernel(&(lconvnet[ncurconvnet]->k_bank[l])); 
	if (lconvnet[ncurconvnet]->pooling_adj[l] != NULL) iftDestroyAdjRel(&(lconvnet[ncurconvnet]->pooling_adj[l]));
	if (lconvnet[ncurconvnet]->norm_adj[l]    != NULL) iftDestroyAdjRel(&(lconvnet[ncurconvnet]->norm_adj[l]));
      }
      for (int i=0; i < lconvnet[ncurconvnet]->nlayers; i++) { 
	if (lconvnet[ncurconvnet]->img_ind[i] != NULL) iftDestroyMatrix(&(lconvnet[ncurconvnet]->img_ind[i]));
      }
      //

      ncurconvnet++;
    }

    iftDestroyConvNetwork(&convnet);
  }

  for(int l=1;l<=3;l++)
    fprintf(stdout,"h[%d] = %d, ",l,hlayers[l]);
  fprintf(stdout,"\n");

  for(int c=0; c < nconvnets; c++)
    iftDestroyConvNetwork(&(lconvnet[c]));
  free(lconvnet);

  return 0;
}
