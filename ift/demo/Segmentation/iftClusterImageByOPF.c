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
  int              norm_value,kmax;
  bool propagate_cluster_true_labels= false;

  if (argc< 7 || argc>9)
    iftError("Usage: iftClusterImageByOPF <image.ppm(pgm,png,scn)> <train_samples_nb> <kmax(percent or value)> <area(0..300)> <type sampling(0:RANDOM/1:GRID)> <do_smoothing(0:NO/1:YES)> [<gt_image]> [<propagate_root_true_labels>(0:NO/1:YES)]","main");

  iftRandomSeed(IFT_RANDOM_SEED);

  img=iftReadImageByExt(argv[1]);

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

  A=NULL;
  Z= iftMImageToDataSet(mimg);

  /* read the gt image if it was given as parameter*/
  iftImage *gt=NULL;
  if (argc > 7){
    gt=iftReadImageByExt(argv[7]);
    iftImageGTToDataSet(gt,Z);
    /* check that the gt image is binary*/
//    if (Z->nclasses !=2)
//      iftError("The gt image must be binary","main");
  }

  t1=iftTic();
  if (!atoi(argv[5])){
    /* do random sampling*/
    iftSelectUnsupTrainSamples(Z,atof(argv[2]),5);
  }
  else{
    /* do grid sampling */
    iftImage *mask1 = iftSelectImageDomain(mimg->xsize,mimg->ysize,mimg->zsize);
    iftImage *mask_sampling = iftGridSampling(mimg, mask1, atoi(argv[2]));

    int train_samples_nb=0;
    #pragma omp parallel for reduction(+:train_samples_nb)
    for (int s=0;s<Z->nsamples;s++){
      int voxel=Z->sample[s].id;
      if (mask_sampling->val[voxel]){
        Z->sample[s].status=IFT_TRAIN;
        train_samples_nb++;
      }
    }
    Z->ntrainsamples=train_samples_nb;
//    printf("Train samples nb -> %d\n",train_samples_nb);
    iftDestroyImage(&mask1);
    iftDestroyImage(&mask_sampling);
  }

  if (atof(argv[3]) < 1.0)
    kmax  = (int) iftMax((atof(argv[3]) * Z->ntrainsamples), 1);
  else
    kmax = iftMax(atoi(argv[3]),1);

  graph = iftCreateKnnGraph(Z,kmax);
  iftUnsupTrain(graph,iftNormalizedCut);
//  iftFastUnsupTrain(graph,iftNormalizedCut);

//  int noutliers = iftUnsupClassifyDataSetLessOutliers(graph, Z);
  iftUnsupClassify(graph,Z);
  t2 = iftToc();

//  if (noutliers >0) {
//    /*modify the alpha values of the dataset to not consider coordenate values*/
//    for (int f = Z->nfeats-3; f < Z->nfeats; f++)
//      Z->alpha[f] = 0.0;
//    iftConqueringOutliersByIFTInMImageDataset(Z);
//    /*reset the alpha values*/
//    for (int f = Z->nfeats-3; f < Z->nfeats; f++)
//      Z->alpha[f] = 1.0;
//  }

  fprintf(stdout,"%s %d %.3f %.3f %d %.2f ",argv[1],Z->nsamples,atof(argv[2]),atof(argv[3]),Z->ngroups,iftCompTime(t1,t2));

  /* propagate the cluster true labels if neccessary*/
  if (argc > 8 && atoi(argv[8])){
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
  if (atoi(argv[6])){
    aux=label;
    label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
    iftDestroyImage(&aux);
  }

  if(!propagate_cluster_true_labels){

    /* this is specifically for the liver database experiments, we try here to identify the bg*/
/*  if (!iftIsColorImage(img)){
    for (int p=0;p<img->n;p++){
      label->val[p]=(img->val[p] <= 20)?0:label->val[p];
    }
  }*/

    aux = iftSelectAndPropagateRegionsAboveArea(label,atoi(argv[4]));
    iftDestroyImage(&label);
    label=aux;
  }
  else{
    aux=iftVolumeOpen(label,atoi(argv[4]),NULL);
    iftDestroyImage(&label);
    label=aux;
    aux=iftVolumeClose(label,atoi(argv[4]),NULL);
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

    if (iftIs3DImage(img)){
      iftWriteImage(border, "border.scn");
    }
    else
      iftWriteImageP2(border, "border.pgm");

    iftDestroyImage(&border);

    if (!iftIs3DImage(img)){

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
#pragma omp parallel for
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
#pragma omp parallel for
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
    }
  }

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
