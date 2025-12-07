#include <ift.h>

bool get_new_val(int *sup_diff_block, int max_labels, iftDataSet *Z, int *regions, int *index, double thresh,int init_row){

  bool found=false;
  double coeff=0;

  float *f1=iftAllocFloatArray(Z->nfeats);
  float *f2=iftAllocFloatArray(Z->nfeats);

  for (int i=init_row;i<max_labels && !found;i++)
    for (int j=i+1;j<max_labels && !found;j++)
      if ((sup_diff_block[i * max_labels + j] > 0) && (regions[i] > 0 && regions[j] > 0)){

        /* compare the probability distributions*/
        for (int m=0;m<Z->nfeats;m++){
          f1[m]= Z->sample[i].feat[m] / regions[i];
          f2[m]= Z->sample[j].feat[m] / regions[j];
        }
        coeff=Z->iftArcWeight(f1,f2,Z->alpha,Z->nfeats);

        if (coeff>=thresh){
          *index=i*max_labels +j;
          found=true;
          break;
        }
        else{
          sup_diff_block[i * max_labels + j]=0;
        }
      }

  iftFree(f1);
  iftFree(f2);

  return found;
}

/* function to try to remove the lines produced by the blocks division. Right now I'm using a matrix of sup_size*sup_size to merge adjacent superpixels. This is a first approach, I need to optimize this.*/

iftImage *remove_block_lines(iftImage *img, iftImage *in_label, iftImage *block_img, double threshold){

  iftImage *new_label_img,*aux;
  iftAdjRel *A;

  new_label_img=iftCreateImage(in_label->xsize,in_label->ysize,in_label->zsize);

  /* put as background all pixels with value less than 20, this is specifically for the liver dataset*/
//  if (!iftIsColorImage(img)){
//    for (int p=0;p<img->n;p++){
//      label->val[p]=(img->val[p] <= 20)?1:label->val[p]+1;
//    }
//    A=iftCircular(sqrtf(2.0));
//
//    aux=iftRelabelRegions(label,A);
//    iftDestroyImage(&label);
//    label=aux;
//
//    iftDestroyAdjRel(&A);
//  }

  int max_label=iftMaximumValue(in_label);

  /* matrix to store the common adjacency (count of superpixels) between neighboring superpixels that are found in different blocks*/
  int *sup_diff_blocks=iftAllocIntArray(max_label*max_label);

  if (!iftIs3DImage(img)){
    A=iftCircular(1.0);
  }
  else{
    A=iftSpheric(1.0);
  }

  int p,q,l1,l2;
  iftVoxel u, v;
  for (int z=0;z < in_label->zsize;z++)
    for (int y=0;y<in_label->ysize;y++){
#pragma omp parallel for private(p,q,l1,l2,u,v)
      for (int x=0;x<in_label->xsize;x++) {
        u.x = x;
        u.y = y;
        u.z = z;
        p = iftGetVoxelIndex(in_label, u);
        for (int i = 1; i < A->n; i++) {
          v=iftGetAdjacentVoxel(A,u,i);
          if (iftValidVoxel(in_label, v)) {
            q = iftGetVoxelIndex(in_label, v);
            if (in_label->val[p] != in_label->val[q] && block_img->val[p] != block_img->val[q]) {
              l1=in_label->val[p]-1;
              l2=in_label->val[q]-1;
#pragma omp atomic
              sup_diff_blocks[l1 * max_label + l2]++;
            }
          }
        }
      }
    }

  /* put all values in the upper triangular matrix to save processing time*/
  for (int i=0;i<max_label;i++)
    for (int j=i+1;j<max_label;j++){
      sup_diff_blocks[i*max_label+j]+=sup_diff_blocks[j*max_label+i];
      sup_diff_blocks[j*max_label+i]=0;
    }

  /*create a dataset to save the histogram features from each superpixel*/
  int *regions=iftAllocIntArray(max_label);
  int nbins;
  if (iftIsColorImage(img))
    nbins=512;
  else
    nbins=16;
  iftDataSet *Z=iftSupervoxelsToLabOrGrayHistogramDataSet(img, in_label, nbins, regions);

  /*join adjacent superpixels */
  int index=-1,row,col,current_row=0;
  int *new_labels=iftAllocIntArray(max_label+1);
  /*init new labels vector*/
  for (int i=1;i<=max_label;i++)
    new_labels[i]=i;

  int curr_val=get_new_val(sup_diff_blocks, max_label, Z, regions, &index, threshold,current_row);

  /* merge superpixels adjacents*/
  while (curr_val){
    /* compute index to merge*/
    row=index/max_label;
    col=index%max_label;

    /* check if there is the same superpixel*/
    if (row == col){
      sup_diff_blocks[row*max_label +col]=0;
    }
    else{

      /*update the histogram dataset and the region vector*/
      for (int m=0;m<Z->nfeats;m++)
        Z->sample[row].feat[m]+=Z->sample[col].feat[m];
      regions[row]+=regions[col];
      regions[col]=0;

      /* the superpixels are merged*/
      sup_diff_blocks[row*max_label +col]=0;
      new_labels[col+1]=row+1;

      /* update the neighbors in the same row, all neighbor superpixels of col will be neighbors of row now */
#pragma omp parallel for
      for (int j=col+1;j<max_label;j++){
        sup_diff_blocks[row*max_label+j]+=sup_diff_blocks[col*max_label+j];
        sup_diff_blocks[col*max_label+j]=0;
      }

      /*update the neighbors in the same col*/
#pragma omp parallel for
      for (int j=row+1;j<max_label;j++){
        sup_diff_blocks[row*max_label+j]+=sup_diff_blocks[j*max_label+col];
        sup_diff_blocks[j*max_label+col]=0;
      }
    }

    current_row=row;
    curr_val=get_new_val(sup_diff_blocks,max_label,Z,regions,&index,threshold,current_row);
  }

  /*update the final labels in the vector*/
#pragma omp parallel for private(l1,l2)
  for (int i=0;i<=max_label;i++){
    l1=new_labels[i];
    l2=new_labels[l1];
    while (l1 != l2){
      l1=l2;
      l2=new_labels[l1];
    }
    new_labels[i]=l1;
  }

  /* substitute the new labels in the image*/
#pragma omp parallel for
  for (int i=0;i<in_label->n;i++)
    new_label_img->val[i]=new_labels[in_label->val[i]];

  /* make the labels consecutively*/
  aux=iftRelabelGrayScaleImage(new_label_img,1);
  iftDestroyImage(&new_label_img);
  new_label_img=aux;

  iftDestroyAdjRel(&A);
  iftFree(new_labels);
  iftDestroyDataSet(&Z);
  iftFree(regions);
  iftFree(sup_diff_blocks);

  return new_label_img;
}

int main(int argc, char *argv[])
{
  iftImage        *img=NULL,*label=NULL,*aux=NULL;
  iftDataSet      *Z=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  char             ext[10],*pos;
  timer           *t1=NULL,*t2=NULL;
  iftKnnGraph *graph;
  bool propagate_cluster_true_labels=false;

  if (argc<8 || argc >10)
    iftError("Usage: iftClusterImageByDivideAndConquerUnsupOPF2Levels <image.ppm(pgm,png,scn)> <num_blocks> <train_samples_nb> <kmax_first_level(percent or value)> <kmax_second_level(percent or value)> <area> <similarity_threshold(0..1, ex 0.85)> [<gt_image>] [<propagate_root_true_labels>(0:NO/1:YES)]","main");

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
  if (argc>8){
    gt=iftReadImageByExt(argv[8]);
    iftImageGTToDataSet(gt,Z);
    /* check that the gt image is binary*/
//    if (Z->nclasses !=2)
//      iftError("The gt image must be binary","main");
  }

  t1 = iftTic();
  graph= iftClusterImageByDivideAndConquerUnsupOPF2Levels(Z, atoi(argv[2]), atof(argv[3]), iftNormalizedCut,
                                                          atof(argv[4]), atof(argv[5]),0);
  t2 = iftToc();
  fprintf(stdout,"%s %d %.2f ",argv[1],Z->ngroups,iftCompTime(t1,t2));

  if (argc > 8 && atoi(argv[9])){
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

  /* do smoothing*/
  aux=label;
  label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
  iftDestroyImage(&aux);

  if (!propagate_cluster_true_labels){

    /* eliminate noise inside blocks*/
    aux   = iftSelectAndPropagateRegionsAboveAreaByColor(img,label,10);
    iftDestroyImage(&label);
    label=aux;

     /* print regions before merging*/
    if (!iftIs3DImage(img)){
      iftDestroyAdjRel(&A);
      A  = iftCircular(sqrtf(2.0));
      B          = iftCircular(0.0);
      RGB.val[0] = 0;
      RGB.val[1] = 255;
      RGB.val[2] = 255;
      YCbCr      = iftRGBtoYCbCr(RGB,255);
      aux        = iftCopyImage(img);
      iftDrawBorders(aux,label,A,YCbCr,B);

      pos = strrchr(argv[1],'.') + 1;
      sscanf(pos,"%s",ext);
      if (strcmp(ext,"png")==0)
        iftWriteImagePNG(aux,"regions_before.png");
      else
        iftWriteImageP6(aux,"regions_before.ppm");

      iftDestroyImage(&aux);
      iftDestroyAdjRel(&B);
    }

    /* remove block lines*/
    iftImage *block_img=iftMImageTilesToLabelImage(mimg,atoi(argv[2]));
    aux= remove_block_lines(img,label, block_img,atof(argv[7]));
    iftDestroyImage(&label);
    label=aux;
    iftDestroyImage(&block_img);

    aux=iftColorizeComp(label);
    iftWriteImageP6(aux,"regions_after.ppm");
    iftDestroyImage(&aux);

      /* eliminate short superpixels*/
    aux   = iftSelectAndPropagateRegionsAboveAreaByColor(img,label,atoi(argv[6]));
    iftDestroyImage(&label);
    label=aux;
  }
  else{
    aux=iftVolumeOpen(label,atoi(argv[6]));
    iftDestroyImage(&label);
    label=aux;
    aux=iftVolumeClose(label,atoi(argv[6]));
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

  /* write labels image */
  if (iftIs3DImage(img)){
    iftWriteImage(label,"labels.scn");
  }
  else{
    iftWriteImageP2(label,"labels.pgm");
    aux   = iftColorizeComp(label);
    iftWriteImageP6(aux,"clusters.ppm");
    iftDestroyImage(&aux);
  }

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
    /*compute br and ue if necessary*/
    iftImage *border = iftBorderImage(label,0);
    if (argc >8){
      iftImage *gt_img=iftReadImageByExt(argv[8]);
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

      int norm_value;
      iftImage *dens=NULL;
      int s,p,q;
      iftVoxel u;

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

      iftDestroyImage(&aux);
      iftDestroyAdjRel(&B);
      iftDestroyImage(&dens);
    }
  }

  iftDestroyDataSetWithoutFeatVectors(&graph->Z);
  iftDestroyKnnGraph(&graph);

  iftDestroyImage(&gt);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);
  iftDestroyMImage(&mimg);
  iftDestroyImage(&label);
  iftDestroyDataSet(&Z);

  return(0);
}
