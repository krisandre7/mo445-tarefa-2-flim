#include "ift.h"

#define MAX_BIN_VAL 1
#define MIN_BIN_VAL 0

//-------------------------------------------------------------------------------------------------

void usage();
float iftFbwScore(iftImage *objsm, iftImage *gt, float beta);

//-------------------------------------------------------------------------------------------------

int main(int argc, const char *argv[]) 
{
  iftImage *objsm, *gt;

  if(argc != 3) usage();

  objsm = iftReadImageByExt(argv[1]);
  gt = iftReadImageByExt(argv[2]);

  printf("%f\n", iftFbwScore(objsm, gt, 1));

  // Free
  iftDestroyImage(&objsm);
  iftDestroyImage(&gt);

  return (0);
}

//-------------------------------------------------------------------------------------------------

void usage()
{
  printf("Usage: iftEvalSalMapByFbw [1] [2] [3] [4] {5}\n");
  printf("---------------------------------------------------------\n");
  printf("[1] - Gray-level object saliency map\n");
  printf("[2] - Binary ground-truth image\n");

  iftError("Too many/few args!", "main");
}

iftMatrix *_iftCreatePixelImportanceMatrix(iftMatrix *gt_mat, iftImage *gt)
{ 
  float ALPHA = log(0.5)/5.0; // Value defined in the paper

  iftMatrix *B_mat;

  B_mat = iftCreateMatrix(gt_mat->n, 1);

  #pragma omp parallel for
  for(int i = 0; i < B_mat->ncols; i++) {
    float g_i;

    g_i = iftMatrixElem(gt_mat, i, 1);
    
    if(g_i == 1.0) iftMatrixElem(B_mat, i, 1) = 1.0;
    else {
      float delta_i;

      delta_i = IFT_INFINITY_FLT;

      for(int j = 0; j < gt->n; j++) {
        float g_j;

        g_j = g_i = iftMatrixElem(gt_mat, j, 1);

        if(g_j == 1 && i != j) {
          float dist_ij;
          iftVoxel voxel_i, voxel_j;

          voxel_i = iftGetVoxelCoord(gt, i);
          voxel_j = iftGetVoxelCoord(gt, j);

          dist_ij = iftVoxelDistance(voxel_i, voxel_j);

          if(dist_ij < delta_i) delta_i = dist_ij;
        }
      }

      iftMatrixElem(B_mat, i, 1) = 2.0 - exp(ALPHA * delta_i);
    }    
  }

  return B_mat;
}

iftMatrix *_iftCreateErrorMatrix(iftMatrix *objsm_mat, iftMatrix *gt_mat)
{
  iftMatrix *error_mat;

  error_mat = iftCreateMatrix(objsm_mat->n, 1);

  #pragma omp parallel for
  for(int i = 0; i < objsm_mat->ncols; i++) {
    float gt_i, objsm_i;

    gt_i = iftMatrixElem(gt_mat, i, 1);
    objsm_i = iftMatrixElem(objsm_mat, i, 1);
    iftMatrixElem(error_mat, i, 1) = fabs(gt_i - objsm_i);
  }

  return error_mat;
}

iftMatrix *_iftMatricesMinimumPointWise(iftMatrix *mat_a, iftMatrix *mat_b)
{
  iftMatrix *min_mat;

  min_mat = iftCreateMatrix(mat_a->ncols, 1);

  #pragma omp parallel for
  for(int i = 0; i < min_mat->nrows; i++) {
    float mat_a_i, mat_b_i;

    mat_a_i = iftMatrixElem(mat_a, i, 1);
    mat_b_i = iftMatrixElem(mat_b, i, 1); 
    
    iftMatrixElem(min_mat, i, 1) = iftMin(mat_a_i, mat_b_i);
  }

  return min_mat;
}

iftMatrix *_iftConvertImageToMatrix(iftImage *img)
{
  int max, min;
  iftMatrix *mat;

  mat = iftCreateMatrix(img->n, 1);
  max = iftMaximumValue(img);
  min = iftMinimumValue(img);

  #pragma omp parallel for
  for(int i = 0; i < img->n; i++) {
    float norm;

    norm = (img->val[i] - min)/((float)max - min);

    iftMatrixElem(mat, i, 1) = norm;
  }

  return mat;
}

iftMatrix *_iftCreateWeightedErrorMatrix(iftMatrix *objsm_mat, iftMatrix *gt_mat)
{
  iftMatrix *error_w_mat;

  error_mat = iftCreateMatrix(objsm_mat->n, 1);

  #pragma omp parallel for
  for(int i = 0; i < objsm_mat->ncols; i++) {
    float gt_i, objsm_i;

    gt_i = iftMatrixElem(gt_mat, i, 1);
    objsm_i = iftMatrixElem(objsm_mat, i, 1);
    iftMatrixElem(error_mat, i, 1) = fabs(gt_i - objsm_i);
  }

  return error_mat;
}

float iftFbwScore(iftImage *objsm, iftImage *gt, float beta)
{
  float tp_w, fp_w, fn_w, precision_w, recall_w, fbw, beta2;
  iftMatrix *objsm_mat, *gt_mat, *tp_w_mat, *fp_w_mat, *fn_w_mat, *inv_error_w_mat, *inv_gt_mat;

  objsm_mat = _iftConvertImageToMatrix(objsm);
  gt_mat = _iftConvertImageToMatrix(gt);
  
  error_w_mat = _iftCreateWeightedErrorMatrix(objsm_mat, gt_mat);


  // error_w_mat = iftMatricesMultiplicationPointWise(min_mat, B_mat);
  // iftDestroyMatrix(&min_mat);
  // iftDestroyMatrix(&B_mat);

  // inv_error_w_mat = iftCopyMatrix(error_w_mat);
  // iftComputeSubtractionBetweenMatrixScalar(inv_error_w_mat, 1.0, 'b');

  // inv_gt_mat = iftCopyMatrix(gt_mat);
  // iftComputeSubtractionBetweenMatrixScalar(inv_gt_mat, 1.0, 'b');

  // tp_w_mat = iftMatricesMultiplicationPointWise(inv_error_w_mat, gt_mat);
  // tp_w = iftMatrixSum(tp_w_mat);
  // iftDestroyMatrix(&tp_w_mat);

  // fp_w_mat = iftMatricesMultiplicationPointWise(error_w_mat, inv_gt_mat);
  // fp_w = iftMatrixSum(fp_w_mat);
  // iftDestroyMatrix(&fp_w_mat);

  // fn_w_mat = iftMatricesMultiplicationPointWise(error_w_mat, gt_mat);
  // fn_w = iftMatrixSum(fn_w_mat);
  // iftDestroyMatrix(&fn_w_mat);

  // iftDestroyMatrix(&error_w_mat);
  // iftDestroyMatrix(&gt_mat);
  // iftDestroyMatrix(&inv_error_w_mat);
  // iftDestroyMatrix(&inv_gt_mat);

  // precision_w = tp_w/(tp_w + fp_w);
  // recall_w = tp_w/(tp_w + fn_w);

  // beta2 = beta*beta;

  // fbw = (1.0 + beta2)*((precision_w * recall_w)/(beta2 * precision_w + recall_w));

  // return fbw;

  return 0;
}

