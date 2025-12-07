#include "ift.h"

int main(int argc, char *argv[])
{
  iftImage        *img=NULL,*bg_img=NULL,*label=NULL,*dens=NULL,*aux=NULL;
  iftDataSet      *Z=NULL,*bg_Z=NULL;
  iftKnnGraph     *graph=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  char             ext[10],*pos;
  int              s,p,q; 
  iftVoxel         u;
  timer           *t1=NULL,*t2=NULL;
  int              norm_value;


  if (argc < 7 || argc > 8)
      iftError(
              "Usage: iftSegmentPlantRoots <image.ppm[pgm,png]> <nb_train_samples> <kmax_perc [0,1]> <area> <do_smoothing> <train_block (USED FOR IFT_TRAIN IN THE SAME IMAGE)> <bg_image.ppm[pgm,png](OPTIONAL)>",
              "main");

  iftRandomSeed(IFT_RANDOM_SEED);

  img=iftReadImageByExt(argv[1]);

  /* convert the image to multi-image*/
  iftMImage *mimg=NULL,*bg_mimg=NULL;
  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAY_CSPACE);
  }
  else{
    mimg = iftImageToMImage(img, LAB_CSPACE);
  }

  Z= iftMImageToDataSet(mimg);

  if (argc == 8){
    /* if the bg image was given as a parameter we train the classifier with samples of this image*/
    bg_img=iftReadImageByExt(argv[7]);

    if (!iftIsColorImage(bg_img)) {
      bg_mimg = iftImageToMImage(img, GRAY_CSPACE);
    }
    else{
      bg_mimg = iftImageToMImage(bg_img, LAB_CSPACE);
    }
    bg_Z = iftMImageToDataSet(bg_mimg);
  }
  else{
    /* if we don't have a bg image we train the classifier with samples of a corner block in the root image. We are fixing the tile so be careful and the count of tiles to be divided the image*/
    int block_vol = mimg->n / (1.0 * 92);

    int n_blocks_x, n_blocks_y, n_blocks_z;
    iftNumberOfEquallyDimensionedTilesWithGivenMaximumSize(mimg->xsize, mimg->ysize, mimg->zsize,block_vol, &n_blocks_x, &n_blocks_y, &n_blocks_z);
    iftImageTiles *tiles = iftMImageTilesByEquallyDividingAxes(mimg, n_blocks_x, n_blocks_y, n_blocks_z);

    int tile_selected=atoi(argv[6]);
    iftBoundingBox bb     = tiles->tile_coords[tile_selected];
    bg_Z=iftMImageBoundingBoxToDataSet(Z,bb,true);
    iftDestroyImageTiles(&tiles);
  }

  t1=iftTic();
  iftSelectUnsupTrainSamples(bg_Z,atof(argv[2]),1);
  int kmax  = (int) iftMax((atof(argv[3]) * bg_Z->ntrainsamples), 1);

  graph = iftCreateKnnGraph(bg_Z,kmax);
  int nb_clusters=iftUnsupTrain(graph,iftNormalizedCut);
  printf("train phase terminated with %d number of groups\n",nb_clusters);

  int noutliers = iftUnsupClassifyDataSetLessOutliers(graph,Z);
  printf("classifying phase terminated. The number of outliers was -> %d\n",noutliers);

  /* relabel the dataset. The group 1 will be the root content and the group 0 will be the bg*/
  Z->ngroups=2;
#pragma omp parallel for
  for (int s=0;s<Z->nsamples;s++){
    if (Z->sample[s].status == IFT_OUTLIER)
      Z->sample[s].group=1;
    else
      Z->sample[s].group=0;
  }

  t2 = iftToc();
  printf("Time transcurred %.2f\t",iftCompTime(t1,t2));

  label = iftDataSetClusterInformationToLabelImage(Z, false);

//  iftImage *labels_orig=iftColorizeComp(label);
//  iftWriteImageP6(labels_orig,"labels_orig.ppm");
//  iftDestroyImage(&labels_orig);

  int do_smoothing=atoi(argv[5]);
  if (do_smoothing){
    aux=label;
    label = iftSmoothRegionsByDiffusion(aux,img,0.5,5);
    iftDestroyImage(&aux);
  }

  if (atoi(argv[4])>0){
    aux = iftFastAreaOpen(label,atoi(argv[4]));
      //iftSelectAndPropagateRegionsAboveArea(label,atoi(argv[4]));
    iftDestroyImage(&label);
    label=aux;
  }
  
  iftWriteImageP2(label,"labels.pgm");
//  aux   = iftColorizeComp(label);
//  iftWriteImageP6(aux,"clusters.ppm");
//  iftDestroyImage(&aux);

//  iftImage *border = iftBorderImage(label);
//  iftWriteImageP2(border, "border.pgm");
//  iftDestroyImage(&border);

  /* create the samples image*/
  norm_value = iftNormalizationValue(iftMaximumValue(img));
  RGB.val[0] = 0;
  RGB.val[1] = norm_value;
  RGB.val[2] = norm_value;
  YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
  B          = iftCircular(1.5);

  if (argc == 8)
    aux = iftCopyImage(bg_img);
  else
    aux = iftCopyImage(img);
  for (s=0; s < graph->nnodes; s++) {
    p = graph->Z->sample[graph->node[s].sample].id;
    u = iftGetVoxelCoord(aux,p);
    iftDrawPoint(aux,u,YCbCr,B,255);
  }
  iftWriteImageP6(aux,"samples.ppm");
  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

  /* create the regions image*/
  iftDestroyAdjRel(&A);
  A  = iftCircular(sqrtf(2.0));
  B          = iftCircular(0.0);
  RGB.val[0] = 0;
  RGB.val[1] = norm_value;
  RGB.val[2] = norm_value;
  YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
  aux        = iftCopyImage(img);
  iftDrawBorders(aux,label,A,YCbCr,B);

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  if (strcmp(ext,"png")==0)
    iftWriteImagePNG(aux,"regions.png");
  else
    iftWriteImageP6(aux,"regions.ppm");

  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

  /* create the dens image */
  dens = iftDataSetWeight(Z);
//  RGB.val[0] = norm_value;
//  RGB.val[1] = 0;
//  RGB.val[2] = 0;
//  B          = iftCircular(1.5);
//  aux        = iftCopyImage(dens);
  int maxdens = iftMaximumValue(dens);
//  for (p=0; p < dens->n; p++)
//    aux->val[p] = (int) (((float)dens->val[p]/maxdens)*norm_value);
//  iftWriteImageP2(aux,"dens.pgm");
//  iftDestroyImage(&aux);
//  iftDestroyAdjRel(&B);

  /* create the outliers image */
  RGB.val[0] = norm_value;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
  B          = iftCircular(1.5);
  aux        = iftCopyImage(dens);
  for (p=0; p < dens->n; p++)
    aux->val[p] = (int) (((float)dens->val[p]/maxdens)*norm_value);
  int outlier_nb=0;
  for (s=0; s < Z->nsamples; s++) {
    if (Z->sample[s].status == IFT_OUTLIER){
      q = Z->sample[s].id;
      u = iftGetVoxelCoord(img,q);
      iftDrawPoint(aux,u,YCbCr,B,255);
      outlier_nb++;
    }
  }
  if (outlier_nb >0)
    iftWriteImageP6(aux,"outlier.ppm");

  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);
  iftDestroyMImage(&mimg);
  iftDestroyImage(&dens);
  iftDestroyImage(&label);
  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&bg_Z);
  iftDestroyKnnGraph(&graph);

  iftDestroyImage(&bg_img);
  iftDestroyMImage(&bg_mimg);


  return(0);
}
