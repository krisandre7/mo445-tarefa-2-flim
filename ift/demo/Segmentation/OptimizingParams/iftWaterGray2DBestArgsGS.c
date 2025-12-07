#include "ift.h"

double find_best_args(iftDict* problem, iftDict* params) {

  char *orig_img_path=iftGetStrValFromDict("orig_img_path",problem);
  char *gt_labels_path=iftGetStrValFromDict("gt_labels_path",problem);
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

  int file_count=iftGetLongValFromDict("count_train_img",problem);
  int limit_nb_superp=iftGetLongValFromDict("limit_nb_superp",problem);
  int limit_secs=iftGetLongValFromDict("limit_secs",problem);
  int type_measure = iftGetLongValFromDict("type_measure", problem);

  float measure_acc =0.0;
  int superpixels_acc=0;

  iftImage        *img=NULL,*aux=NULL,*border=NULL,*gt_label=NULL,*gt_border=NULL,*basins=NULL,*marker;
  timer           *t1=NULL,*t2=NULL;
  iftAdjRel *A,*B;

  float spatial_radius_arg=(float)iftGetDblValFromDict("spatial_radius_arg",problem);
  int vol_arg=(int)iftGetDblValFromDict("vol_thresh_arg",params);
  bool exited=false;

  A      = iftCircular(spatial_radius_arg);

  for (int i=0;i<file_count;i++){

    img=iftReadImageByExt(image_files->files[i]->path);

    t1=iftTic();

    iftImage *filtered = iftMedianFilter(img, B);
    basins = iftImageBasins(filtered, A);

    iftImageForest *fst = iftCreateImageForest(basins, A);
    marker = iftVolumeClose(basins,vol_arg);
    iftWaterGrayForest(fst,marker);

    t2= iftToc();
    float time = iftCompTime(t1, t2) / 1000;
    /* if the method execution is more than 1 minute discard this configuration of parameters*/
    if (time > (float)limit_secs){
      measure_acc =IFT_INFINITY_FLT_NEG;
      exited=true;
      break;
    }

    int nb_sup=iftMaximumValue(fst->label);
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
      measure_acc -= iftUnderSegmentation(gt_label,fst->label);
    }
    else{
      /*compute boundary recall*/
      border  = iftBorderImage(fst->label,0);
      gt_border=iftBorderImage(gt_label,0);
      measure_acc += iftBoundaryRecall(gt_border, border, 2.0);

      iftDestroyImage(&border);
      iftDestroyImage(&gt_border);
    }

    iftDestroyImage(&img);
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImage(&gt_label);
    iftDestroyImageForest(&fst);
  }

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);

  if (!exited){
    float mean_sup=superpixels_acc/file_count;
    if (mean_sup > (float)limit_nb_superp)
      measure_acc =IFT_INFINITY_FLT_NEG;
    else
      measure_acc /=file_count;
  }

  iftDestroyDir(&image_files);
  iftDestroyDir(&gt_label_files);

  return measure_acc;

}

int main(int argc, char *argv[])
{
  iftRandomSeed(IFT_RANDOM_SEED);

  if (argc != 12)
    iftError("Usage: iftWaterGray2DBestArgsGS <dir_orig_img> <dir_label_img> <count_train_img> <limit_nb_superp> <limit_secs> <optimizing_measure(0:BR/1:UE> <orig_img_ext(ppm:png,...)> <spatial_radius> <init_volume> <final_volume> <step_volume>","main");

  iftDict* params = iftCreateDict();
  iftDict* problem = iftCreateDict();

  iftInsertIntoDict("vol_thresh_arg", iftRange(atoi(argv[9]), atoi(argv[10]), atoi(argv[11])),params);

  iftInsertIntoDict("orig_img_ext",argv[7],problem);
  iftInsertIntoDict("orig_img_path", argv[1],problem);
  iftInsertIntoDict("gt_labels_path", argv[2],problem);
  iftInsertIntoDict("count_train_img", atoi(argv[3]),problem);
  iftInsertIntoDict("limit_nb_superp", atoi(argv[4]),problem);
  iftInsertIntoDict("limit_secs", atoi(argv[5]),problem);
  iftInsertIntoDict("type_measure", atoi(argv[6]),problem);
  iftInsertIntoDict("spatial_radius_arg", atof(argv[8]),problem);

  iftDict *result =iftGridSearch(params, find_best_args, problem);

  printf("Best parameters for %s with %d images an limit number superpixels %d and %d seconds with measure-> %s\n",argv[1],atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6])?"ue":"br");

  printf("best_vol_thresh_arg -> %d\n",(int)iftGetDblValFromDict("vol_thresh_arg",result));
  printf("best_spatial_radius_arg -> %.2f\n",iftGetDblValFromDict("spatial_radius_arg",problem));
  float res=(float)iftGetDblValFromDict("best_func_val",result);
  printf("best_objetive_func -> %.4f\n",atoi(argv[6])?-res:res);

  iftDestroyDict(&params);
  iftDestroyDict(&problem);
  iftDestroyDict(&result);

  return(0);
}
