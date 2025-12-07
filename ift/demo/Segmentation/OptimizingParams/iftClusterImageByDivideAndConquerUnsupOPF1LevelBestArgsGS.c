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

  iftFree(new_labels);
  iftDestroyDataSet(&Z);
  iftFree(regions);

  return new_label_img;
}


double find_best_args(iftDict* problem, iftDict* params) {

  char *orig_img_path = iftGetStrValFromDict("orig_img_path", problem);
  char *gt_labels_path = iftGetStrValFromDict("gt_labels_path", problem);
  char *orig_img_ext= iftGetStrValFromDict("orig_img_ext", problem);

  iftDir *image_files = iftLoadFilesFromDirBySuffix(orig_img_path, orig_img_ext);
  char *gt_img_ext="";
  if (!strcmp(orig_img_ext,"png") || !strcmp(orig_img_ext,"PNG") || !strcmp(orig_img_ext,"ppm") || !strcmp(orig_img_ext,"PPM") || !strcmp(orig_img_ext,"pgm") || !strcmp(orig_img_ext,"PGM"))
    gt_img_ext="pgm";
  else
  if (!strcmp(orig_img_ext,"scn"))
    gt_img_ext="scn";
  else
      iftError("Invalid extension for orig image directory", "find_best_args");
  iftDir *gt_label_files = iftLoadFilesFromDirBySuffix(gt_labels_path, gt_img_ext);

  int file_count = iftGetLongValFromDict("count_train_img", problem);
  int limit_nb_superp = iftGetLongValFromDict("limit_nb_superp", problem);
  int limit_secs = iftGetLongValFromDict("limit_secs", problem);
  int type_measure = iftGetLongValFromDict("type_measure", problem);

  float measure_acc = 0.0;
  int superpixels_acc = 0;

  iftImage *img = NULL, *label = NULL, *aux = NULL, *border = NULL, *gt_label = NULL, *gt_border = NULL;
  iftDataSet *Z = NULL;
  timer *t1 = NULL, *t2 = NULL;
  iftAdjRel *A;
  iftMImage *mimg, *eimg;

  int nb_blocks=iftGetLongValFromDict("nb_blocks_arg", problem);
  float train_perc_arg=iftGetLongValFromDict("train_perc_arg",problem);
  float kmax_arg = (float)iftGetDblValFromDict("kmax_arg", params);
  int area_arg=iftGetLongValFromDict("area_arg",problem);
  float threshold=(float)iftGetDblValFromDict("thresh_arg", problem);
  bool exited=false;

  for (int i = 0; i < file_count; i++) {

    img = iftReadImageByExt(image_files->files[i]->path);

    /* convert the image to multi-image*/
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
    else {
      mimg = iftImageToMImage(img, LABNorm2_CSPACE);
      eimg = iftExtendMImageByVoxelCoord(mimg, 1);
      iftDestroyMImage(&mimg);
      mimg = eimg;
    }

    Z = iftMImageToDataSet(mimg);

    t1 = iftTic();
    iftClusterImageByDivideAndConquerUnsupOPF1Level(Z, nb_blocks, train_perc_arg, iftNormalizedCut, kmax_arg,0);
    t2 = iftToc();

    float time = iftCompTime(t1, t2) / 1000;

    /* if the method execution is more than the minutes passed as argument discard this configuration of parameters*/
    if (time > (float) limit_secs) {
      measure_acc =IFT_INFINITY_FLT_NEG;
      exited=true;
      break;
    }

    label = iftDataSetClusterInformationToLabelImage(Z, false);

    /* do smoothing*/
    aux=label;
    label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
    iftDestroyImage(&aux);

    /* eliminate noise inside blocks*/
    aux   = iftSelectAndPropagateRegionsAboveAreaByColor(img,label,10);
    iftDestroyImage(&label);
    label=aux;

    /* remove block lines*/
    iftImage *block_img=iftMImageTilesToLabelImage(mimg,nb_blocks);
    aux= remove_block_lines(img,label, block_img,threshold);
    iftDestroyImage(&label);
    label=aux;
    iftDestroyImage(&block_img);

    /* eliminate short superpixels*/
    aux   = iftSelectAndPropagateRegionsAboveAreaByColor(img,label,area_arg);
    iftDestroyImage(&label);
    label=aux;

    int nb_sup=iftMaximumValue(label);
    if (nb_sup > 2*limit_nb_superp){
      measure_acc =IFT_INFINITY_FLT_NEG;
      exited=true;
      break;
    }
    superpixels_acc+=nb_sup;

    gt_label=iftReadImageByExt(gt_label_files->files[i]->path);
    aux=iftRelabelGrayScaleImage(gt_label,0);
    iftDestroyImage(&gt_label);
    gt_label=aux;

    if (type_measure){
      /*computing under segmentation error*/
      /*here we are minimizing not maximizing*/
      measure_acc -= iftUnderSegmentation(gt_label,label);
    }
    else{
      /*compute boundary recall*/
      border  = iftBorderImage(label,0);
      gt_border=iftBorderImage(gt_label,0);
      measure_acc += iftBoundaryRecall(gt_border, border, 2.0);

      iftDestroyImage(&border);
      iftDestroyImage(&gt_border);
    }

    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyDataSet(&Z);
    iftDestroyImage(&label);

    iftDestroyImage(&gt_label);

  }
  if (!exited) {
    float mean_sup = superpixels_acc / file_count;
    if (mean_sup > (float) limit_nb_superp)
      measure_acc =IFT_INFINITY_FLT_NEG;
    else
      measure_acc /= file_count;
  }

  iftDestroyDir(&image_files);
  iftDestroyDir(&gt_label_files);

  printf("\nWith kmax = %.5f ->fun_obj = %.5f\n",kmax_arg,measure_acc);

  return measure_acc;
}

int main(int argc, char *argv[])
{
  iftRandomSeed(IFT_RANDOM_SEED);

  if (argc != 15)
    iftError("Usage: iftClusterImageByDivideAndConquerUnsupOPF1LevelBestArgsGS <dir_orig_img> <dir_label_img> <count_train_img> <limit_nb_superp> <limit_secs> <optimizing measure 0(BR)/1(UE)> <count_train_samples> <orig_img_ext (ppm:png,...)> <init_nb_block> <init_kmax1> <final_kmax1> <step_kmax1> <init_area> <threshold>","main");

  iftDict* params = iftCreateDict();
  iftDict* problem = iftCreateDict();

  iftInsertIntoDict("kmax_arg", iftRange(atof(argv[10]), atof(argv[11]), atof(argv[12])),params);


  iftInsertIntoDict("train_perc_arg",atoi(argv[7]),problem);
  iftInsertIntoDict("orig_img_ext",argv[8],problem);
  iftInsertIntoDict("nb_blocks_arg",atoi(argv[9]),problem);
  iftInsertIntoDict("orig_img_path", argv[1],problem);
  iftInsertIntoDict("gt_labels_path", argv[2],problem);
  iftInsertIntoDict("count_train_img", atoi(argv[3]),problem);
  iftInsertIntoDict("limit_nb_superp", atoi(argv[4]),problem);
  iftInsertIntoDict("limit_secs", atoi(argv[5]),problem);
  iftInsertIntoDict("type_measure", atoi(argv[6]),problem);
  iftInsertIntoDict("area_arg", atoi(argv[13]),problem);
  iftInsertIntoDict("thresh_arg", atof(argv[14]),problem);

  iftDict *result =iftGridSearch(params, find_best_args, problem);

  printf("Best parameters for %s with %d images an limit number superpixels %d and %d seconds with measure-> %s and train samples by block %d\n",argv[1],atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6])?"ue":"br",atoi(argv[7]));

  printf("best_nb_blocks_arg -> %.4f\n",(float)iftGetLongValFromDict("nb_blocks_arg",problem));
  printf("best_train_perc_arg -> %.4f\n",(float)iftGetLongValFromDict("train_perc_arg",problem));
  printf("best_kmax_arg -> %.4f\n",(float)iftGetDblValFromDict("kmax_arg",result));
  float res=(float)iftGetDblValFromDict("best_func_val",result);
  printf("best_objetive_func -> %.4f\n",atoi(argv[6])?-res:res);

  iftDestroyDict(&params);
  iftDestroyDict(&problem);
  iftDestroyDict(&result);

  return(0);
}
