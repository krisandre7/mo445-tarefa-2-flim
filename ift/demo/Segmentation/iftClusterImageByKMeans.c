#include "ift.h"

int main(int argc, char *argv[])
{
  iftImage        *img=NULL,*label=NULL,*dens=NULL,*aux=NULL;
  iftDataSet      *Z=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  char             ext[10],*pos;
  timer           *t1=NULL,*t2=NULL;
  int              norm_value;
  bool propagate_cluster_true_labels= false;

  iftRandomSeed(IFT_RANDOM_SEED);

  if (argc< 7 || argc>9)
      iftError(
              "Usage: iftClusterImageByKMeans <image.ppm(pgm,png,scn)> <number_groups> <max_iterations(ex:200)> <area> <do_smoothing(0:NO/1:YES)> <type(kmeans:0/kmedoids:1)> [<gt_image>] [<propagate_root_true_labels(0:NO/1:YES)]",
              "main");

  img=iftReadImageByExt(argv[1]);
  int   k = atoi(argv[2]);
  int   maxIterations = atoi(argv[3]);
  float minImprovement = 1E-5;

  /* convert the image to multi-image*/
  iftMImage *mimg;
  iftMImage *eimg;

  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAYNorm_CSPACE);
     if (img->zsize > 1)
      A = iftSpheric(1.0);
    else
      A = iftCircular(sqrtf(2.0));
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
  if (argc>7){
    gt=iftReadImageByExt(argv[7]);
    iftImageGTToDataSet(gt,Z);
    /* check that the gt image is binary*/
//    if (Z->nclasses !=2)
//      iftError("The gt image must be binary","main");
  }

  t1 = iftTic();
  iftClusterDataSetByKMeans(Z, k, maxIterations, minImprovement, atoi(argv[6]), 1, 0);
  t2 = iftToc();

  /*int count_prototypes=0;
  for (int s=0;s<Z->nsamples;s++)
    if (Z->sample[s].status == IFT_PROTOTYPE){
      count_prototypes++;
      printf("%d ",s);
    }
  printf("\nCount prototypes -> %d\n",count_prototypes);*/


  fprintf(stdout,"%s %d %d %.3f ",argv[1],Z->nsamples,Z->ngroups,iftCompTime(t1,t2));

  /* propagate the cluster true labels if neccessary*/
  if (argc > 7 && atoi(argv[8])){
    propagate_cluster_true_labels=true;
    iftPropagateClusterTrueLabels2(Z);
    label = iftDataSetToLabelImage(Z, false);
  }
  else{
    label = iftDataSetClusterInformationToLabelImage(Z, false);
  }


  if (propagate_cluster_true_labels && !iftIs3DImage(img)){
    iftWriteImageP2(label,"labels_orig.pgm");
  }
  else{
    if (!iftIs3DImage(img)){
      iftImage *labels_orig=iftColorizeComp(label);
      iftWriteImageP6(labels_orig,"labels_orig.ppm");
      iftDestroyImage(&labels_orig);
    }
    else{
      iftWriteImage(label,"labels_orig.scn");
    }
  }

  /* check if we do smoothing */
  if (atoi(argv[5])){
    aux=label;
    label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
    iftDestroyImage(&aux);
  }

  /* this is specifically for the liver database experiments*/
//  if (!iftIsColorImage(img)){
//    for (int p=0;p<img->n;p++){
//      label->val[p]=(img->val[p] <= 20)?0:label->val[p];
//    }
//  }

  if(!propagate_cluster_true_labels){
    aux = iftSelectAndPropagateRegionsAboveArea(label,atoi(argv[4]));
    iftDestroyImage(&label);
    label=aux;
  }
  else{
    aux=iftVolumeOpen(label,atoi(argv[4]));
    iftDestroyImage(&label);
    label=aux;
    aux=iftVolumeClose(label,atoi(argv[4]));
    iftDestroyImage(&label);
    label=aux;

    if (iftMinimumValue(label) > 0){
      for (int p=0;p<label->n;p++)
        label->val[p]--;
    }

    aux= iftExtractLargestObjectInLabelImage(label);
    iftDestroyImage(&label);
    label=aux;
  }

  printf("%d ",iftMaximumValue(label));

  if (iftIs3DImage(img)){
    iftWriteImage(label,"labels.scn");
  }
  else{
    iftWriteImageP2(label,"labels.pgm");
    aux   = iftColorizeComp(label);
    iftWriteImageP6(aux,"clusters.ppm");
    iftDestroyImage(&aux);
  }

  /* check if we want a propagation result or a superpixel result*/
  if(propagate_cluster_true_labels){
    iftSetStatus(Z,IFT_TEST);
    printf("%.3f ",iftTruePositives(Z));
    aux=iftRelabelGrayScaleImage(gt,false);
    iftDestroyImage(&gt);
    gt=aux;
    if (iftMinimumValue(label) >0){
      for (int s=0;s<label->n;s++)
        label->val[s]-=1;
    }
    printf("%.3lf\n",iftDiceSimilarity(label,gt));
  }
  else{
    /*compute br and ue*/
    iftImage *border = iftBorderImage(label,0);
    if (argc > 7){
      iftImage *gt_img=iftReadImageByExt(argv[7]);
      aux=iftRelabelGrayScaleImage(gt_img,0);
      iftDestroyImage(&gt_img);
      gt_img=aux;
      iftImage *gt_borders=iftBorderImage(gt_img,0);
      printf("\nbr -> %.4f\n",iftBoundaryRecall(gt_borders, border, 2.0));
      printf("ue -> %.4f\n",iftUnderSegmentation(gt_img, label));

      iftDestroyImage(&gt_img);
      iftDestroyImage(&gt_borders);
    }
    else
      printf("\n");

    if (iftIs3DImage(img)){
      iftWriteImage(border, "border.scn");
    }
    else
      iftWriteImageP2(border, "border.pgm");

    iftDestroyImage(&border);

    if (!iftIs3DImage(img)){
      dens = iftDataSetWeight(Z);

        norm_value = iftNormalizationValue(iftMaximumValue(img));
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
    }
  }

  iftDestroyAdjRel(&A);
  iftDestroyImage(&gt);
  iftDestroyImage(&img);
  iftDestroyMImage(&mimg);
  iftDestroyImage(&dens);
  iftDestroyImage(&label);
  iftDestroyDataSet(&Z);

  return(0);
}
