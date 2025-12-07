#include "ift.h"

float iftAchievableSegmentationAccuracy(iftImage *orig_gt, iftImage *orig_label)
{
  iftImage *label, *gt;
  int nobjs, nsuperpixels, ncorrect_pixels=0, npixels_best_obj, index_best_obj;
  int min_val_superpixels, min_val_gt, is_min_gt_0 = 0, is_min_label_0 = 0;
  float acc;
  // Number of pixel in the intersection superpixel gt_object (rows: superpixels, columns: objects)
  iftMatrix *mat_npixels; 
  int *npixels_object_gt, *npixels_object_intersect;

  assert(orig_label->n == orig_label->n);

  // Get min values
  min_val_superpixels = iftMinimumValue(orig_label);
  min_val_gt = iftMinimumValue(orig_gt);
  // Set minimum value of gt and label to 1
  if (min_val_gt == 0) {
    gt = iftAddValue(orig_gt, 1);
    is_min_gt_0 = 1;
  } else {
    gt = orig_gt;
  }
  if (min_val_superpixels == 0) {
    label = iftAddValue(orig_label, 1);
    is_min_label_0 = 1;
  } else {
    label = orig_label;
  }

  // Alloc matrix to count number of pixels in the intersection (superpixels, gt_object)
  nobjs = iftMaximumValue(gt);
  nsuperpixels = iftMaximumValue(label); 
  mat_npixels = iftCreateMatrix(nobjs, nsuperpixels);
  npixels_object_gt = iftAllocIntArray(nobjs);
  npixels_object_intersect = iftAllocIntArray(nobjs);
  
  // Fill intersection matrix
  for (int p = 0; p < label->n; ++p) {
    int index_sup = label->val[p]-1;
    int index_obj = gt->val[p]-1;
    iftMatrixElem(mat_npixels, index_obj, index_sup) += 1;
    npixels_object_gt[gt->val[p]-1]++;
  }
  // Compute accuracy
  for (int i = 0; i < nsuperpixels; ++i)
  {
    npixels_best_obj = -1;
    index_best_obj = 0;
    for (int j = 0; j < nobjs; ++j)
    {
      if (npixels_best_obj < iftMatrixElem(mat_npixels, j, i)) {
        npixels_best_obj = iftMatrixElem(mat_npixels, j, i);
        index_best_obj = j;
      }
    }
    npixels_object_intersect[index_best_obj] += npixels_best_obj;
    ncorrect_pixels += npixels_best_obj;
  }
  acc = (float)ncorrect_pixels / (float)label->n;

  // Print accuracies by class
  for (int i = 0; i < nobjs; ++i)
  {
    float acc_by_class = (float)npixels_object_intersect[i] / (float)npixels_object_gt[i];
    printf("ASA class %d : %f \n", i+1, acc_by_class);
  }

  // Free
  if (is_min_gt_0) iftDestroyImage(&gt);
  if (is_min_label_0) iftDestroyImage(&label);
  iftDestroyMatrix(&mat_npixels);
  return acc;
}


int main(int argc, char *argv[]) {

	iftImage *label, *gt_img;

	if (argc != 3) {
        iftError("Usage: iftWriteColorClassmap <label_image> <gt_image>", "main");
	}

	label = iftReadImageByExt(argv[1]);
	gt_img = iftReadImageByExt(argv[2]);

  int max_value = iftMaximumValue(gt_img);
  if (max_value == 255) {
    for (int p = 0; p < gt_img->n; ++p) {
      if (gt_img->val[p] != 0)
        gt_img->val[p] = 1;
      else
        gt_img->val[p] = 0;
    }
  }

	float asa = iftAchievableSegmentationAccuracy(gt_img, label);
	printf("Overall Achievable Segmentation Accuracy (ASA) : %f\n", asa);


  printf("max val gt %d\n", iftMaximumValue(gt_img));
  printf("max val label %d\n", iftMaximumValue(label));

	

	iftDestroyImage(&gt_img);
	iftDestroyImage(&label);

	return 0;
}