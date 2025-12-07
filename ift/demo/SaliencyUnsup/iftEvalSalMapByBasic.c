#include "ift.h"

#define MAX_BIN_VAL 1
#define MIN_BIN_VAL 0

//-------------------------------------------------------------------------------------------------

void usage();
float iftFloatArrayMeanValue(iftFloatArray *array);
iftFloatArray *iftThrFBetaScores(iftImage *objsm, iftImage *gt, float beta);
iftFloatArray *iftThrRecallScores(iftImage *objsm, iftImage *gt);
iftFloatArray *iftThrFalseAlarmScores(iftImage *objsm, iftImage *gt);
iftFloatArray *iftThrPrecisionScores(iftImage *objsm, iftImage *gt);
iftFloatArray *iftThrPASCALScores(iftImage *objsm, iftImage *gt);
float iftAreaUnderTheCurve(iftFloatArray *recall, iftFloatArray *false_alarm);
float iftAveragePrecision(iftFloatArray *precision, iftFloatArray *recall);


//-------------------------------------------------------------------------------------------------

int main(int argc, const char *argv[]) 
{
  iftImage *objsm, *gt;
  float ap, auc;
  iftFloatArray *fscores, *recall, *false_alarm, *precision, *pascal;

  if(argc != 3) usage();

  objsm = iftReadImageByExt(argv[1]);
  gt = iftReadImageByExt(argv[2]);

  #pragma omp parallel
  {
    #pragma omp single
    fscores = iftThrFBetaScores(objsm, gt, 0.3);

    #pragma omp single
    recall = iftThrRecallScores(objsm, gt);

    #pragma omp single
    false_alarm = iftThrFalseAlarmScores(objsm, gt);

    #pragma omp single
    precision = iftThrPrecisionScores(objsm, gt);

    #pragma omp single
    pascal = iftThrPASCALScores(objsm, gt);

    #pragma omp barrier

    #pragma omp single
    auc = iftAreaUnderTheCurve(recall, false_alarm);

    #pragma omp single
    ap = iftAveragePrecision(precision, recall);
  }

  printf("Mean Recall measure: %0.3f\n", iftFloatArrayMeanValue(recall));
  printf("Mean False-Alarm measure: %0.3f\n", iftFloatArrayMeanValue(false_alarm));
  printf("Mean Precision measure: %0.3f\n", iftFloatArrayMeanValue(precision));
  printf("Mean F(0.3)-measure: %0.3f\n", iftFloatArrayMeanValue(fscores));
  printf("Mean PASCAL measure: %0.3f\n", iftFloatArrayMeanValue(pascal));
  printf("Area Under The Curve measure: %0.3f\n", auc);
  printf("Average Precision measure: %0.3f\n", ap);

  return (0);
}

//-------------------------------------------------------------------------------------------------

void usage()
{
  printf("Usage: iftEvalSalMap [1] [2] [3] [4] {5}\n");
  printf("---------------------------------------------------------\n");
  printf("[1] - Gray-level object saliency map\n");
  printf("[2] - Binary ground-truth image\n");

  iftError("Too many/few args!", "main");
}

void _iftComputeBasicMetricsFromImage(iftImage *bin, iftImage *gt, int *tp, int *tn, int *fp, int *fn)
{
  iftVerifyImageDomains(bin, gt, "_iftComputeBasicMetricsFromImage");

  int tmp_tp, tmp_tn, tmp_fp, tmp_fn;

  tmp_tp = 0; tmp_tn = 0; tmp_fp = 0; tmp_fn = 0;

  #pragma omp parallel for reduction(+:tmp_tp,tmp_tn,tmp_fp,tmp_fn)
  for (int p = 0; p < bin->n; p++) {
    if(bin->val[p] == gt->val[p]) { // The binary map is correct
      if(bin->val[p] == MAX_BIN_VAL) tmp_tp++; // Correct object classification
      else tmp_tn++; // Correct background classification
    } else { // The binary map is incorrect
      if(bin->val[p] == MAX_BIN_VAL) tmp_fp++; // Incorrect object classification
      else tmp_fn++; // Incorrect background classification
    }
  }

  *tp = tmp_tp; *tn = tmp_tn; *fp = tmp_fp; *fn = tmp_fn;
}

iftImage *_iftNormalizeImage(iftImage *gray, int min, int max) 
{
  int curr_max, curr_min;
  iftImage *norm;

  curr_max = iftMaximumValue(gray);
  curr_min = iftMinimumValue(gray);

  norm = iftCreateImage(gray->xsize, gray->ysize, gray->zsize);

  #pragma omp parallel for
  for(int p = 0; p < gray->n; p++) {
    norm->val[p] = iftRound(gray->val[p] * (max - min)/(float)(curr_max - curr_min));
  }

  return norm;
}

iftIntArray *_iftGetDistinctThreshVals(iftImage *gray) 
{
  int curr_max, num_threshs, thresh_count;
  iftBMap *added;
  iftIntArray *thresh_vals;

  curr_max = iftMaximumValue(gray);
  added = iftCreateBMap(curr_max + 1);
  num_threshs = 0;

  for(int p = 0; p < gray->n; p++) {
    if(iftBMapValue(added, gray->val[p]) == 0) {
      iftBMapSet1(added, gray->val[p]);
      num_threshs++;
    }
  }

  thresh_vals = iftCreateIntArray(num_threshs);
  thresh_count = 0;

  for(int i = 0; i < added->n; i++) {
    if(iftBMapValue(added, i) == 1) thresh_vals->val[thresh_count++] = i;
  }

  // Free
  iftDestroyBMap(&added);

  return thresh_vals;
}

//-------------------------------------------------------------------------------------------------

float iftFloatArrayMeanValue(iftFloatArray *array)
{
  float mean;

  mean = 0.0;

  #pragma omp parallel for reduction(+:mean)
  for(int i = 0; i < array->n; i++) mean += array->val[i];

  mean /= (float)array->n;

  return mean;
}

iftFloatArray *iftThrFBetaScores(iftImage *objsm, iftImage *gt, float beta)
{
  iftVerifyImageDomains(objsm, gt, "iftMeanFBetaScore");

  int max_thresh;
  float beta2;
  iftIntArray *thresh_vals;
  iftFloatArray *fscores;
  iftImage *norm_gt;

  beta2 = beta * beta;

  thresh_vals = _iftGetDistinctThreshVals(objsm);
  max_thresh = thresh_vals->val[thresh_vals->n - 1];
  fscores = iftCreateFloatArray(thresh_vals->n);
  norm_gt = _iftNormalizeImage(gt, MIN_BIN_VAL, MAX_BIN_VAL);

  #pragma omp parallel for
  for(int i = 0; i < thresh_vals->n; i++) {
    int tp, tn, fp, fn;
    iftImage *bin;

    bin = iftThreshold(objsm, thresh_vals->val[i], max_thresh, MAX_BIN_VAL);

    _iftComputeBasicMetricsFromImage(bin, norm_gt, &tp, &tn, &fp, &fn);

    fscores->val[i] = (1.0 + beta2) * tp /(float)((1.0 + beta2) * tp + beta2 * fn + fp);

    //Free
    iftDestroyImage(&bin);
  }

  // Free
  iftDestroyImage(&norm_gt);
  iftDestroyIntArray(&thresh_vals);

  return (fscores);
}

iftFloatArray *iftThrRecallScores(iftImage *objsm, iftImage *gt)
{
  iftVerifyImageDomains(objsm, gt, "iftThrRecallScores");
  int max_thresh;
  iftIntArray *thresh_vals;
  iftFloatArray *recall;
  iftImage *norm_gt;

  thresh_vals = _iftGetDistinctThreshVals(objsm);
  max_thresh = thresh_vals->val[thresh_vals->n - 1];
  recall = iftCreateFloatArray(thresh_vals->n);
  norm_gt = _iftNormalizeImage(gt, MIN_BIN_VAL, MAX_BIN_VAL);

  #pragma omp parallel for
  for(int i = 0; i < thresh_vals->n; i++) {
    int tp, tn, fp, fn;
    iftImage *bin;

    bin = iftThreshold(objsm, thresh_vals->val[i], max_thresh, MAX_BIN_VAL);

    _iftComputeBasicMetricsFromImage(bin, norm_gt, &tp, &tn, &fp, &fn);

    recall->val[i] = tp/((float)tp + fn);

    //Free
    iftDestroyImage(&bin);
  }

  // Free
  iftDestroyImage(&norm_gt);
  iftDestroyIntArray(&thresh_vals);

  return (recall);
}

iftFloatArray *iftThrFalseAlarmScores(iftImage *objsm, iftImage *gt)
{
  iftVerifyImageDomains(objsm, gt, "iftThrFalseAlarmScores");
  int max_thresh;
  iftIntArray *thresh_vals;
  iftFloatArray *false_alarm;
  iftImage *norm_gt;

  thresh_vals = _iftGetDistinctThreshVals(objsm);
  max_thresh = thresh_vals->val[thresh_vals->n - 1];
  false_alarm = iftCreateFloatArray(thresh_vals->n);
  norm_gt = _iftNormalizeImage(gt, MIN_BIN_VAL, MAX_BIN_VAL);

  #pragma omp parallel for
  for(int i = 0; i < thresh_vals->n; i++) {
    int tp, tn, fp, fn;
    iftImage *bin;

    bin = iftThreshold(objsm, thresh_vals->val[i], max_thresh, MAX_BIN_VAL);

    _iftComputeBasicMetricsFromImage(bin, norm_gt, &tp, &tn, &fp, &fn);

    false_alarm->val[i] = fp/((float)tn + fp);

    //Free
    iftDestroyImage(&bin);
  }

  // Free
  iftDestroyImage(&norm_gt);
  iftDestroyIntArray(&thresh_vals);

  return (false_alarm);
}

iftFloatArray *iftThrPrecisionScores(iftImage *objsm, iftImage *gt)
{
  iftVerifyImageDomains(objsm, gt, "iftThrPrecisionScores");
  int max_thresh;
  iftIntArray *thresh_vals;
  iftFloatArray *precision;
  iftImage *norm_gt;

  thresh_vals = _iftGetDistinctThreshVals(objsm);
  max_thresh = thresh_vals->val[thresh_vals->n - 1];
  precision = iftCreateFloatArray(thresh_vals->n);
  norm_gt = _iftNormalizeImage(gt, MIN_BIN_VAL, MAX_BIN_VAL);

  #pragma omp parallel for
  for(int i = 0; i < thresh_vals->n; i++) {
    int tp, tn, fp, fn;
    iftImage *bin;

    bin = iftThreshold(objsm, thresh_vals->val[i], max_thresh, MAX_BIN_VAL);

    _iftComputeBasicMetricsFromImage(bin, norm_gt, &tp, &tn, &fp, &fn);

    precision->val[i] = tp/((float)tp + fp);

    //Free
    iftDestroyImage(&bin);
  }

  // Free
  iftDestroyImage(&norm_gt);
  iftDestroyIntArray(&thresh_vals);

  return (precision);
}

iftFloatArray *iftThrPASCALScores(iftImage *objsm, iftImage *gt)
{
  iftVerifyImageDomains(objsm, gt, "iftThrPASCALScores");
  int max_thresh;
  iftIntArray *thresh_vals;
  iftFloatArray *pascal;
  iftImage *norm_gt;

  thresh_vals = _iftGetDistinctThreshVals(objsm);
  max_thresh = thresh_vals->val[thresh_vals->n - 1];
  pascal = iftCreateFloatArray(thresh_vals->n);
  norm_gt = _iftNormalizeImage(gt, MIN_BIN_VAL, MAX_BIN_VAL);

  #pragma omp parallel for
  for(int i = 0; i < thresh_vals->n; i++) {
    int tp, tn, fp, fn;
    iftImage *bin;

    bin = iftThreshold(objsm, thresh_vals->val[i], max_thresh, MAX_BIN_VAL);

    _iftComputeBasicMetricsFromImage(bin, norm_gt, &tp, &tn, &fp, &fn);

    pascal->val[i] = tp/((float)tp + fn + fp);

    //Free
    iftDestroyImage(&bin);
  }

  // Free
  iftDestroyImage(&norm_gt);
  iftDestroyIntArray(&thresh_vals);

  return (pascal);
}

float iftAreaUnderTheCurve(iftFloatArray *recall, iftFloatArray *false_alarm)
{
  if(recall->n != false_alarm->n) iftError("Recall and False-alarm does not have the same size!", "iftAreaUnderTheCurve");

  float sum;

  sum = 0.0;
  for(int i = 1; i < false_alarm->n; i++) {
    float height;

    // Trapezoidal area
    height = fabs(false_alarm->val[i] - false_alarm->val[i-1]);

    sum += (recall->val[i] + recall->val[i-1]) * height/2.0;
  }

  return sum;
}

float iftAveragePrecision(iftFloatArray *precision, iftFloatArray *recall)
{
  if(precision->n != recall->n) iftError("Precision and Recall does not have the same size!", "iftAveragePrecision");

  int num_points;
  float ap;
  iftFloatArray *ap_array;

  num_points = precision->n;
  
  ap_array = iftCreateFloatArray(num_points);

  #pragma omp parallel for
  for(int i = 0; i < num_points; i++) {
    float max_precision;

    max_precision = 0.0;

    for(int j = 0; j < num_points; j++) {
      if(recall->val[j] >= recall->val[i]) {
        if(precision->val[j] > max_precision) max_precision = precision->val[j];
      }
    }

    ap_array->val[i] = max_precision;
  }

  ap = iftFloatArrayMeanValue(ap_array);


  return ap;
}