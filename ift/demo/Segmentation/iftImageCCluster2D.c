#include "ift.h"

int main(int argc, char *argv[])
{
  iftImage        *img=NULL,*label=NULL,*dens=NULL,*aux=NULL;
  iftDataSet      *Z=NULL;
  iftKnnGraph     *graph=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  char             ext[10],*pos;
  int              s,p,q; 
  iftVoxel         u;
  timer           *t1=NULL,*t2=NULL;
  int              norm_value;


  if (argc< 7 || argc>8)
    iftError("Usage: iftImageCluster2D <image.ppm[pgm]> <train_perc [0,1]> <kmax_perc [0,1]> <number of groups> <area> <do_smoothing> <gt_image(OPTIONAL)>","main");


  iftRandomSeed(IFT_RANDOM_SEED);

  img=iftReadImageByExt(argv[1]);

  /* convert the image to multi-image*/
  iftMImage *mimg;
  iftMImage *eimg;

  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAYNorm_CSPACE);
    A  = iftCircular(sqrtf(2.0));
    eimg=iftExtendMImageByAdjacencyAndVoxelCoord(mimg,A,1);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);
    mimg = eimg;
  }
  else{
    mimg = iftImageToMImage(img, LABNorm2_CSPACE);
    eimg = iftExtendMImageByVoxelCoord(mimg,1);
    iftDestroyMImage(&mimg);
    mimg = eimg;
  }

  Z= iftMImageToDataSet(mimg);

  /* read the gt image if it was given as parameter*/
  iftImage *gt=NULL;
  if (argc==8){
    gt=iftReadImageByExt(argv[7]);
    iftImageGTToDataSet(gt,Z);
  }

  t1=iftTic();
  iftSelectUnsupTrainSamples(Z,atof(argv[2]),5);

  int kmax  = (int) iftMax((atof(argv[3]) * Z->ntrainsamples), 1);

  graph = iftCreateKnnGraph(Z,kmax);
  iftUnsupTrainWithCClusters(graph, atoi(argv[4]));
  if (Z->nclasses > 0){
    iftPropagateClusterTrueLabels(graph);
  }
  iftUnsupClassify(graph,Z);
  t2     = iftToc();
  fprintf(stdout,"%s\t%d\t%.3f\t%.3f\t%d\t%.2f\t",argv[1],Z->nsamples,atof(argv[2]),atof(argv[3]),Z->ngroups,
          iftCompTime(t1,t2));

  if(Z->nclasses>0){
    printf("Accuracy -> %.2f\t",iftTruePositives(Z));
    label = iftDataSetToLabelImage(Z, false);
  }
  else{
    label= iftDataSetClusterInformationToLabelImage(Z, false);
  }

  iftImage *labels_orig=iftColorizeComp(label);
  iftWriteImageP6(labels_orig,"labels_orig.ppm");
  iftDestroyImage(&labels_orig);

  int do_smoothing=atoi(argv[6]);
  if (do_smoothing){
    aux=label;
    label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
    iftDestroyImage(&aux);
  }

  aux   = iftSelectAndPropagateRegionsAboveArea(label,atoi(argv[5]));
  iftDestroyImage(&label);
  label=aux;
  printf("%d",iftMaximumValue(label));
  printf("\n");

  iftWriteImageP2(label,"labels.pgm");
  aux   = iftColorizeComp(label);
  iftWriteImageP6(aux,"clusters.ppm");
  iftDestroyImage(&aux);

  iftImage *border = iftBorderImage(label,0);
  iftWriteImageP2(border, "border.pgm");
  iftDestroyImage(&border);

  dens = iftDataSetWeight(Z);

  norm_value = iftNormalizationValue(iftMaximumValue(img));
  RGB.val[0] = 0;
  RGB.val[1] = norm_value;
  RGB.val[2] = norm_value;
  YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
  B          = iftCircular(1.5);
  aux        = iftCopyImage(img);
  for (s=0; s < graph->nnodes; s++) {
    p = graph->Z->sample[graph->node[s].sample].id;
    u = iftGetVoxelCoord(img,p);
    iftDrawPoint(aux,u,YCbCr,B,255);
  }
  iftWriteImageP6(aux,"samples.ppm");
  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

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

  RGB.val[0] = norm_value;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
  B          = iftCircular(1.5);
  aux        = iftCopyImage(dens);
  int maxdens = iftMaximumValue(dens);
  for (p=0; p < dens->n; p++)
    aux->val[p] = (int) (((float)dens->val[p]/maxdens)*norm_value);
  for (s=0; s < graph->nnodes; s++) {
    if (graph->node[s].root==s){
      q = graph->Z->sample[graph->node[s].sample].id;
      u = iftGetVoxelCoord(img,q);
      iftDrawPoint(aux,u,YCbCr,B,255);
    }
  }
  iftWriteImageP6(aux,"maxima.ppm");
  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

  /* generate an image with the outliers*/
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
  else
    iftWriteImageP2(aux,"outlier.ppm");

  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

  iftDestroyAdjRel(&A);
  iftDestroyImage(&gt);
  iftDestroyImage(&img);
  iftDestroyMImage(&mimg);
  iftDestroyImage(&dens);
  iftDestroyImage(&label);
  iftDestroyDataSet(&Z);
  iftDestroyKnnGraph(&graph);


  return(0);
}

