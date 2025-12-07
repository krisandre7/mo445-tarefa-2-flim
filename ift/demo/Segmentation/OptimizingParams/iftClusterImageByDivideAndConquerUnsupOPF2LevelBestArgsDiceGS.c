#include <ift.h>

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
    iftError("Invalid extension for orig image directory","find_best_args");
  iftDir *gt_label_files = iftLoadFilesFromDirBySuffix(gt_labels_path, gt_img_ext);

  int file_count = iftGetLongValFromDict("count_train_img", problem);
  int limit_nb_groups = iftGetLongValFromDict("limit_nb_groups", problem);
  int limit_secs = iftGetLongValFromDict("limit_secs", problem);
  int type_measure = iftGetLongValFromDict("type_measure", problem);

  float measure = 0.0;
  int groups_acc = 0;

  iftImage        *img=NULL,*label=NULL,*aux=NULL,*border=NULL,*gt_label=NULL,*gt_border=NULL;
  iftDataSet      *Z=NULL;
  timer           *t1=NULL,*t2=NULL;
  iftAdjRel *A;
  iftMImage *mimg,*eimg;

  int nb_blocks=iftGetLongValFromDict("num_blocks_arg", problem);
  float train_perc_arg=iftGetLongValFromDict("train_perc_arg",problem);
  float kmax1_arg=iftGetDblValFromDict("kmax1_arg",params);
  float kmax2_arg=iftGetDblValFromDict("kmax2_arg",params);
  int area_arg=iftGetLongValFromDict("area_arg",problem);
  bool exited=false;

  for (int i=0;i<file_count && !exited;i++) {

    img = iftReadImageByExt(image_files->files[i]->path);

    /* convert the image to multi-image*/
    if (!iftIsColorImage(img)) {
      mimg = iftImageToMImage(img, GRAYNorm_CSPACE);
      if (img->zsize > 1)
        A = iftSpheric(1.0);
      else
        A = iftCircular(1.0);
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

    Z= iftMImageToDataSet(mimg);

    gt_label=iftReadImageByExt(gt_label_files->files[i]->path);
    iftImageGTToDataSet(gt_label,Z);
    if (Z->nclasses !=2)
      iftError("The gt image must be binary","main");

    t1 = iftTic();
    iftKnnGraph* graph= iftClusterImageByDivideAndConquerUnsupOPF2Levels(Z, nb_blocks, train_perc_arg,
                                                                         iftNormalizedCut,
                                                                         kmax1_arg, kmax2_arg,0);
    t2 = iftToc();
    float time = iftCompTime(t1, t2) / 1000;

    if (Z->ngroups > 2*limit_nb_groups || time > (float)limit_secs){
      measure =IFT_INFINITY_FLT_NEG;
      exited=true;
      break;
    }

    groups_acc += Z->ngroups;
    iftPropagateClusterTrueLabels2(Z);

    if (type_measure){
      /*computing acc*/
      iftSetStatus(Z,IFT_TEST);
      measure += iftTruePositives(Z);
    }
    else{
      /*compute dice*/
      label = iftDataSetToLabelImage(Z,false);

      /* do smoothing*/
      aux=label;
      label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
      iftDestroyImage(&aux);

      aux=iftVolumeOpen(label,area_arg);
      iftDestroyImage(&label);
      label=aux;
      aux=iftVolumeClose(label,area_arg);
      iftDestroyImage(&label);
      label=aux;

      if (iftMinimumValue(label) > 0){
        for (int p=0;p<label->n;p++)
          label->val[p]--;
      }

      aux= iftExtractLargestObjectInLabelImage(label);
      iftDestroyImage(&label);
      label=aux;

      aux=iftRelabelGrayScaleImage(gt_label,false);
      iftDestroyImage(&gt_label);
      gt_label=aux;

      if (iftMinimumValue(label) >0){
        for (int s=0;s<label->n;s++)
          label->val[s]-=1;
      }
      measure+=iftDiceSimilarity(label,gt_label);
    }

    iftDestroyImage(&label);
    iftDestroyKnnGraph(&graph);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyDataSet(&Z);
    iftDestroyImage(&gt_label);
  }

  if (!exited){
    int mean_groups = groups_acc / file_count;
    if (mean_groups > limit_nb_groups)
      measure =IFT_INFINITY_FLT_NEG;
    else
      measure /=file_count;
  }

  iftDestroyDir(&image_files);
  iftDestroyDir(&gt_label_files);

  printf("\nWith kmax1 = %.5f and kmax2 = %.5f ->fun_obj = %.5f\n", kmax1_arg, kmax2_arg, measure);
  fflush(stdout);

  return measure;

}

int main(int argc, char *argv[])
{
  iftRandomSeed(IFT_RANDOM_SEED);

  if (argc != 17)
    iftError("Usage: iftClusterImageByDivideAndConquerUnsupOPF2LevelBestArgsDiceGS <dir_orig_img> <dir_label_img> <count_train_img> <limit_nb_groups> <limit_secs> <optimizing measure 0(DICE)/1(ACC)> <count_train_samples> <orig_img_ext (ppm:png,...)> <init_nb_block> <init_kmax1> <final_kmax1> <step_kmax1> <init_kmax2> <final_kmax2> <step_kmax2> <area>","main");

  iftDict* params = iftCreateDict();
  iftDict* problem = iftCreateDict();

  iftInsertIntoDict("kmax1_arg", iftRange(atof(argv[10]), atof(argv[11]), atof(argv[12])),params);
  iftInsertIntoDict("kmax2_arg", iftRange(atof(argv[13]),atof(argv[14]), atof(argv[15])),params);

  iftInsertIntoDict("train_perc_arg",atoi(argv[7]),problem);
  iftInsertIntoDict("orig_img_ext",argv[8],problem);
  iftInsertIntoDict("num_blocks_arg",atoi(argv[9]),problem);
  iftInsertIntoDict("orig_img_path", argv[1],problem);
  iftInsertIntoDict("gt_labels_path", argv[2],problem);
  iftInsertIntoDict("count_train_img", atoi(argv[3]),problem);
  iftInsertIntoDict("area_arg", atoi(argv[16]),problem);
  iftInsertIntoDict("limit_nb_groups", atoi(argv[4]),problem);
  iftInsertIntoDict("limit_secs", atoi(argv[5]),problem);
  iftInsertIntoDict("type_measure", atoi(argv[6]),problem);

  iftDict *result =iftGridSearch(params, find_best_args, problem);

  printf("Best parameters for %s with %d train images images, limited number of groups %d, %d seconds, with measure-> %s, and train samples by block %d\n",argv[1],atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6])?"dice":"acc",atoi
          (argv[7]));

  printf("best_num_blocks_arg -> %d\n",(int)iftGetLongValFromDict("num_blocks_arg",problem));
  printf("best_train_perc_arg -> %.4f\n",(float)iftGetLongValFromDict("train_perc_arg",problem));
  printf("best_kmax1_arg -> %.4f\n",(float)iftGetDblValFromDict("kmax1_arg",result));
  printf("best_kmax2_arg -> %.4f\n",(float)iftGetDblValFromDict("kmax2_arg",result));
  float res=(float)iftGetDblValFromDict("best_func_val",result);
  printf("best_objetive_func -> %.4f\n",atoi(argv[6])?-res:res);

  iftDestroyDict(&params);
  iftDestroyDict(&problem);
  iftDestroyDict(&result);

  return(0);
}
