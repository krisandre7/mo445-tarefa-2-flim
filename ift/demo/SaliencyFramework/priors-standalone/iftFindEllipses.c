#include "iftSaliency.h"
#include "iftSaliencyPriors.h"

iftImage **readImageSet(iftFileSet *fs);

float EllipseEquation(iftVoxel u, iftTensorScale *tensor_scale)
{
  float ellipse_eq = sqrtf(powf((u.x - tensor_scale->pos_focus->val[1].x), 2) +
			   (powf((u.y - tensor_scale->pos_focus->val[1].y), 2))) +
    sqrtf(powf((u.x - tensor_scale->neg_focus->val[1].x), 2) +
	  (powf((u.y - tensor_scale->neg_focus->val[1].y), 2)));

  return(ellipse_eq);
}

int main(int argc, const char *argv[]) {

    if (argc != 3) {
        iftError("Usage: iftFindEllipses <bin_path> <out_path>", "main");
    }

    char out_path[250];
    iftFileSet *fs_binary = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);

    strcpy(out_path, argv[2]);
    iftImage **bin = readImageSet(fs_binary);
    int nimages    = fs_binary->n;

    /* for each binary image */
    
    for(int i = 0; i < nimages; i++){
      /* label its components from 1 to n */
      iftImage *label_img = iftLabelComp(bin[i],NULL);
      int       nobjs     = iftMaximumValue(label_img);

      /* for each component */
      for (int o=1; o <= nobjs; o++) {
	/* extract the component from the label image */
	iftImage *obj = iftExtractObject(label_img, o);
	/* relabel it as though it were a superpixel image */
	for (int p=0; p < obj->n; p++){
	  if (obj->val[p]!=0)
	    obj->val[p]=2;
	  else
	    obj->val[p]=1;
	}
	/* Estimate centroids and ellipses for object and background */
	iftVoxelArray *centroids     = iftGeometricCentersFromLabelImage(obj);
	iftTensorScale *tensor_scale = iftSuperpixelToTensorScale(obj, 32,
								  0, 100000);
	/* Create an ellipse image */
	iftImage *ellipsesImage      = iftCreateImageFromImage(obj);
	for(int p = 0; p < obj->n; p++) {
	  iftVoxel u       = iftGetVoxelCoord(obj, p);
	  float ellipse_eq = EllipseEquation(u,tensor_scale); 
	  if (ellipse_eq > 2*tensor_scale->major_axis->val[1]) {
	    ellipsesImage->val[p] = 0;
	  }else{
	    ellipsesImage->val[p] = 255;
	  }
	}
	/* compute error */
	float error = 0.0, n = 0;
	for(int p = 0; p < obj->n; p++) {
	  if (((obj->val[p]==1)&&(ellipsesImage->val[p]!=0))||
	      ((obj->val[p]==2)&&(ellipsesImage->val[p]==0))){
	    error++;
	  }
	  if ((obj->val[p]==2)||(ellipsesImage->val[p]!=0))
	    n++;
	}
	error /= n;
        char *basename = iftFilename(fs_binary->files[i]->path,
				     iftFileExt(fs_binary->files[i]->path));
	printf("basename %s object %d error %f\n",basename,o,error);
	
        char filename[200];
        sprintf(filename, "%s/%s_%d.png", out_path, basename,o);
        iftWriteImageByExt(ellipsesImage, filename);
	iftDestroyTensorScale(&tensor_scale);
        iftDestroyVoxelArray(&centroids);
        iftDestroyImage(&ellipsesImage);
	iftDestroyImage(&obj);
	iftFree(basename);
      }
      iftDestroyImage(&label_img);
    }
    
    iftDestroyFileSet(&fs_binary);
    for(int i = 0; i < nimages; i++){
      iftDestroyImage(&bin[i]);
    }
    free(bin);
}

iftImage **readImageSet(iftFileSet *fs){
    int first = 0;
    int last  = fs->n - 1;

    iftImage **orig = (iftImage **) calloc(fs->n, sizeof(iftImage *));

    for (int i = first; i <= last; i++) {
        char *basename = iftFilename(fs->files[i]->path, iftFileExt(fs->files[i]->path));
        printf("Reading file %s: %d of %ld files\r", basename, i + 1, fs->n);
        fflush(stdout);
        orig[i] = iftReadImageByExt(fs->files[i]->path); // it seems better than YCbCrNorm_CSPACE and LAB_CSPACE
    }
    return(orig);
}
