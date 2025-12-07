#include "ift.h"

#define stdev_factor 0.001

void SaveImage(char *dir, char *basename, iftMImage *mimg)
{
    char filename[200];

    sprintf(filename, "%s/%s.mimg", dir, basename);
    iftWriteMImage(mimg, filename);
}

void iftWriteMImageAsTxt(iftMImage *mimg, char *filename)
{
    FILE *f = NULL;
    f = fopen(filename,"w");
    if (f){
        for (int b=0; b < mimg->m; b++){
            fprintf(f,"%f\n",mimg->val[0][b]);
        }
    }
    fclose(f);
}

iftMImage *ReadImageWithColorConv(char *filename, const char *ext, iftColorSpace cspace)
{
  iftMImage *input = NULL;
  if (strcmp(ext,".mimg") == 0){
    input = iftReadMImage(filename);
  }else{
    iftImage  *img = iftReadImageByExt(filename);
    input          = iftImageToMImage(img, cspace); 
    iftDestroyImage(&img);    
  }

  return(input);
}

void MeanValuesPerImage(iftMImage *img, iftImage *label, iftMImage *mean)
{
  float nelems = 0;

  if (label) {
    for (int p=0; p < img->n; p++){
      if (label->val[p]) {
	nelems++;
      }
    }
  }else{
    nelems=img->n;
  }
  
  for (int b=0; b < img->m; b++){
    mean->val[0][b] = 0.0;
  }
  
#pragma omp parallel for
  for (int p=0; p < img->n; p++){
    for (int b=0; b < img->m; b++){
        if (label) {
            if (label->val[p]) {
                mean->val[0][b] += img->val[p][b];
          }
        }else{
            mean->val[0][b] += img->val[p][b];
        }
    }	   
  }

  for (int b=0; b < img->m; b++){
    mean->val[0][b] = mean->val[0][b]/nelems;
  }
}

void StdevValuesPerImage(iftMImage *img, iftImage *label, iftMImage *mean, iftMImage *stdev)
{
  float nelems = 0;

  if (label) {
    for (int p=0; p < img->n; p++){
      if (label->val[p]) {
	nelems++;
      }
    }
  }else{
    nelems=img->n;
  }

  for (int b=0; b < img->m; b++){
    stdev->val[0][b] = 0.0;
  }
  
#pragma omp parallel for
  for (int p=0; p < img->n; p++){
    for (int b=0; b < img->m; b++){
        if (label) {
            if (label->val[p]) {
                stdev->val[0][b] += (img->val[p][b] - mean->val[0][b]) *
                                    (img->val[p][b] - mean->val[0][b]);
          }
        } else {
            stdev->val[0][b] += (img->val[p][b] - mean->val[0][b]) *
                                (img->val[p][b] - mean->val[0][b]);
        }
    }	   
  }
  for (int b=0; b < img->m; b++){
    stdev->val[0][b] = sqrt(stdev->val[0][b]/nelems) + stdev_factor;
  }
}

void MyNormalizeImageByZScore(iftMImage *img, iftImage *label, iftMImage *mean, iftMImage *stdev)
{
#pragma omp parallel for
  for (int p=0; p < img->n; p++){
    for (int b=0; b < img->m; b++){
        if (label){
            if (label->val[p]) {
	      img->val[p][b] = (img->val[p][b] - mean->val[0][b]) / stdev->val[0][b];
            } else {
	      img->val[p][b] = 0;
            }
        } else {
	  img->val[p][b] = (img->val[p][b] - mean->val[0][b]) / stdev->val[0][b];
        }
    }
  }
}


void MeanValuesPerBand(iftFileSet *fs, iftFileSet *fs_label,
		       const char *ext, iftMImage *mean)
{
  float nelems = 0;
  
  for (int i=0; i < fs->n; i++) {
    iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);
    char *basename1 = iftFilename(fs->files[i]->path,ext);
    iftImage *label = NULL;
    
    if (fs_label){
      const char *ext2 = iftFileExt(fs_label->files[i]->path);
      char *basename2 = iftFilename(fs_label->files[i]->path,ext2);
      if (strcmp(basename1,basename2)==0){
        label = iftReadImageByExt(fs_label->files[i]->path);
	for (int p=0; p < img->n; p++){
	  if (label->val[p]) {
	    nelems++;
	  }
	}
      }else{
	iftError("Basenames %s and %s must be the same",
		 "MeanValuesPerBand",basename1,basename2);
      }
      iftFree(basename2);
    }else{
      nelems += img->n;
    }
    iftFree(basename1);

#pragma omp parallel for
    for (int p=0; p < img->n; p++){
      for (int b=0; b < img->m; b++){
	if (label){
	  if (label->val[p]){
	    mean->val[0][b] += img->val[p][b];
	  }
	} else {
	  mean->val[0][b] += img->val[p][b];
	}
      }  
    }
    
    iftDestroyMImage(&img);
    if (label)
      iftDestroyImage(&label);
  }
      
  for (int b=0; b < mean->m; b++){
    mean->val[0][b] = mean->val[0][b]/nelems;
  }
}

void StdevValuesPerBand(iftFileSet *fs, iftFileSet *fs_label, const char *ext, iftMImage *mean, iftMImage *stdev)
{
  float nelems = 0;
  
  for (int i=0; i < fs->n; i++) {
    iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);
    char *basename1 = iftFilename(fs->files[i]->path,ext);
    iftImage *label = NULL;
    
    if (fs_label){
      const char *ext2 = iftFileExt(fs_label->files[i]->path);
      char *basename2 = iftFilename(fs_label->files[i]->path,ext2);
      if (strcmp(basename1,basename2)==0){
        label = iftReadImageByExt(fs_label->files[i]->path);
	for (int p=0; p < img->n; p++){
	  if (label->val[p]) {
	    nelems++;
	  }
	}
      }else{
	iftError("Basenames %s and %s must be the same",
		 "StdevValuesPerBand",basename1,basename2);
      }
      iftFree(basename2);
    }else{
      nelems += img->n;
    }
    iftFree(basename1);

#pragma omp parallel for
    for (int p=0; p < img->n; p++){
      for (int b=0; b < img->m; b++){
	if (label) {
	  if (label->val[p]) {
	    stdev->val[0][b] += (img->val[p][b] - mean->val[0][b]) *
	      (img->val[p][b] - mean->val[0][b]);
	  }	  
	} else {
	  stdev->val[0][b] += (img->val[p][b] - mean->val[0][b]) *
	    (img->val[p][b] - mean->val[0][b]);
	}
      }
    }
    
    iftDestroyMImage(&img);
    if (label)
      iftDestroyImage(&label);
  }
      
  for (int b=0; b < stdev->m; b++){
    stdev->val[0][b] = sqrt(stdev->val[0][b]/nelems)+stdev_factor;
  }    
}

void NormalizeAndSavePerBand(iftFileSet *fs, iftFileSet *fs_label, const char *ext, iftMImage *mean, iftMImage *stdev, char *output_dir)
{
  for (int i=0; i < fs->n; i++) {
    iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);
    iftImage *label = NULL;
    if (fs_label){
        label = iftReadImageByExt(fs_label->files[i]->path);
    }
    
#pragma omp parallel for
    for (int p=0; p < img->n; p++){
      for (int b=0; b < img->m; b++){
	if (label) {
	  if (label->val[p]) {
	    img->val[p][b] = (img->val[p][b] - mean->val[0][b]) / stdev->val[0][b];
	  } else {
	    img->val[p][b] = 0;
	  }
	} else {
	  img->val[p][b] = (img->val[p][b] - mean->val[0][b]) / stdev->val[0][b];
	}
      }
    }
    char *basename  = iftFilename(fs->files[i]->path, ext);
    SaveImage(output_dir, basename, img);
    iftDestroyMImage(&img);
    iftDestroyImage(&label);
  }
}

void MeanValuesPerVoxel(iftFileSet *fs, const char *ext, iftMImage *mean)
{

  for (int i=0; i < fs->n; i++) {
    iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);
    
#pragma omp parallel for
    for (int p=0; p < img->n; p++){
      for (int b=0; b < img->m; b++){
	mean->val[p][b] += img->val[p][b];
      }	   
    }
    
    iftDestroyMImage(&img);
  }
      
#pragma omp parallel for
    for (int p=0; p < mean->n; p++){
      for (int b=0; b < mean->m; b++){
	mean->val[p][b] = mean->val[p][b]/fs->n;
      }
    }
}

void StdevValuesPerVoxel(iftFileSet *fs, const char *ext, iftMImage *mean, iftMImage *stdev)
{
  
  for (int i=0; i < fs->n; i++) {
    iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);
    
#pragma omp parallel for
    for (int p=0; p < img->n; p++){
      for (int b=0; b < img->m; b++){
	stdev->val[p][b] += (img->val[p][b] - mean->val[p][b]) *
	  (img->val[p][b] - mean->val[p][b]);
      }	   
    }
    
    iftDestroyMImage(&img);
  }
      
#pragma omp parallel for
  for (int p=0; p < stdev->n; p++){
    for (int b=0; b < stdev->m; b++){
      stdev->val[p][b] = sqrt(stdev->val[p][b]/fs->n)+stdev_factor;
    }
  }
}

void NormalizeAndSavePerVoxel(iftFileSet *fs, const char *ext, iftMImage *mean, iftMImage *stdev, char *output_dir)
{
  for (int i=0; i < fs->n; i++) {
    iftMImage *img = ReadImageWithColorConv(fs->files[i]->path, ext, LABNorm2_CSPACE);
    
#pragma omp parallel for
    for (int p=0; p < img->n; p++){
      for (int b=0; b < img->m; b++){
	img->val[p][b] = (img->val[p][b] - mean->val[p][b]) / stdev->val[p][b];
      }	   
    }
    char *basename  = iftFilename(fs->files[i]->path, ext);
    SaveImage(output_dir, basename, img);
    iftDestroyMImage(&img);
  }
}

int main(int argc, char *argv[])
{
    timer *tstart = NULL;

    if ((argc != 4) && (argc != 5))
      iftError("Usage: iftBatchNormalization <P1> <P2> <P3 optional> <P4>\n"
	       "P1: Input folder/csv file with the images \n"
               "    (.mimg, .nii.gz, etc)\n"
	       "P2: 0 - adaptive per image, 1- per band in dataset, 2 - per voxel in dataset\n"
           "P3: Label folder/csv file with the label images (optional)\n"
	       "P4: Output folder with normalized images (.mimg). \n"
               "    Option 1/2 in P2 generates \n"
               "    <input folder>-mean.mimg and <input folder>-stdev.mimg\n" 
               "    with the normalization parameters per band/voxel.\n",
	       "main");

    tstart = iftTic();

    /* Get input parameters */

    iftFileSet *fs_input  = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);
    iftFileSet *fs_label  = NULL;
    int adaptive          = atoi(argv[2]);
    char *output_dir;
    if (argc == 5){
      fs_label   = iftLoadFileSetFromDirOrCSV(argv[3], 1, 1);
      output_dir = argv[4];
    } else {
      output_dir = argv[3];
    }
    const char *ext = iftFileExt(fs_input->files[0]->path);
    iftMakeDir(output_dir);

    /* Verify input parameters and perform initial procedures */

    iftMImage *input=NULL, *mean=NULL, *stdev=NULL;
    iftImage  *label=NULL;

    input = ReadImageWithColorConv(fs_input->files[0]->path, ext,
				   LABNorm2_CSPACE);
    int nImgs=fs_input->n, nBands=input->m;
    char filename[200];
    char *basename;
    
    /* Normalization process */
    
    switch (adaptive) {
      
    case 0: /* adaptive */
      iftDestroyMImage(&input);
      mean  = iftCreateMImage(1,1,1,nBands);
      stdev = iftCreateMImage(1,1,1,nBands);
      for (int i = 0; i < nImgs; i++) {
	input = ReadImageWithColorConv(fs_input->files[i]->path, ext, LABNorm2_CSPACE);
	if (fs_label){
	  label = iftReadImageByExt(fs_label->files[i]->path);
	  if (!iftIsDomainEqual(input,label))
	    iftError("Input and label domains are not equal.","main");
	}
	MeanValuesPerImage(input,label,mean);
	StdevValuesPerImage(input,label,mean,stdev);
	MyNormalizeImageByZScore(input,label,mean,stdev);
	basename  = iftFilename(fs_input->files[i]->path, ext);
	SaveImage(output_dir, basename, input);
	iftFree(basename);
	iftDestroyMImage(&input);
	iftDestroyImage(&label);
      }
      break;
    case 1: /* per band in dataset */      
      iftDestroyMImage(&input);
      mean  = iftCreateMImage(1,1,1,nBands);
      stdev = iftCreateMImage(1,1,1,nBands);
      iftDestroyMImage(&input);
      MeanValuesPerBand(fs_input,fs_label,ext,mean);
      StdevValuesPerBand(fs_input,fs_label,ext,mean,stdev);
      NormalizeAndSavePerBand(fs_input,fs_label,ext,mean,stdev,output_dir);
      basename = iftBasename(argv[1]);
      sprintf(filename,"%s-mean.txt",basename);
      iftWriteMImageAsTxt(mean,filename);
      sprintf(filename,"%s-stdev.txt",basename);
      iftWriteMImageAsTxt(stdev,filename);
      iftFree(basename);
      break;
    case 2: /* per voxel in dataset */
      mean  = iftCreateMImage(input->xsize,input->ysize,input->zsize,nBands);
      stdev = iftCreateMImage(input->xsize,input->ysize,input->zsize,nBands);
      iftDestroyMImage(&input);
      MeanValuesPerVoxel(fs_input,ext,mean);
      StdevValuesPerVoxel(fs_input,ext,mean,stdev);
      NormalizeAndSavePerVoxel(fs_input,ext,mean,stdev,output_dir);
      basename = iftBasename(argv[1]);
      sprintf(filename,"%s-mean.mimg",basename);
      iftWriteMImage(mean,filename);
      sprintf(filename,"%s-stdev.mimg",basename);
      iftWriteMImage(stdev,filename);
      iftFree(basename);
      break;
    }
    
    /* Destroy memory */
    
    iftDestroyFileSet(&fs_input);
    iftDestroyFileSet(&fs_label);
    iftDestroyMImage(&mean);
    iftDestroyMImage(&stdev);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    
    return (0);
}
