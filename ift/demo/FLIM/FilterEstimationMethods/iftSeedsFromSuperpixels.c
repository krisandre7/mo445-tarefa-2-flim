#include "ift.h"

/* 

Creates seed files from superpixel images -- each center of superpixel
becomes a seed. The seeds may be unlabeled (they receive label 1),
labeled with the image class (label), or labeled with the object label
in a given ground-truth image. A folder with multichannel images and
an adjacency relation are given to convert the seed set of each image
into a patch dataset and reduce the number of seeds by clustering,
before outputting the resulting seeds in each seed file.

*/

int GetImageLabel(char *basename)
{
  iftSList *info  = iftSplitString(basename,"_");
  iftSNode *node  = info->head;
  int image_label = atoi(node->elem);
  iftDestroySList(&info);
  return(image_label);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    /* if ((argc != 7)&&(argc != 8)) */
    /*   iftError("Usage: iftSeedsFromSuperpixels P1 P2 P3 P4 (optional)\n" */
    /* 	       "P1: input folder with superpixel (label) images (.png, .nii.gz)\n" */
    /* 	       "P2: output folder with the seed files (-seeds.txt)\n" */
    /* 	       "P3: input folder with the corresponding multichannel images\n" */
    /* 	       "P4: adjacency radius for patch extraction and seed clustering\n" */
    /* 	       "P5: number of clusters per image\n" */
    /* 	       "P6: label type: 0 (unlabeled), 1 (image class), 2 (object label)\n" */
    /* 	       "P7: input folder with ground-truth images for label type 2 (optional)\n", */
    /* 	       "main"); */
    
    /* tstart = iftTic(); */
    
    /* // Load input parameters */
    
    /* iftFileSet  *fs       = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);     */
    /* char ext[10]; */
    /* sprintf(ext,"%s",iftFileExt(fs->files[0]->path)); */
    /* char *out_dir         = argv[2]; */
    /* iftMakeDir(out_dir); */
    /* char *mimg_dir        = argv[3]; */
    /* iftAdjRel *A          = NULL; */
    /* float adj_radius      = atof(argv[4]); */
    /* int  nclusters        = atoi(argv[5]); */
    /* int label_type        = atoi(argv[6]); */
    /* char *gt_dir          = NULL; */

    /* if ((label_type == 2)&&(argc != 8)) */
    /*   iftError("GT images are required for label type 2","main"); */

    /* if ((label_type < 0)||(label_type>2)) */
    /*   iftError("Label type must be in {0,1,2}","main"); */

    /* if (argc == 8) */
    /*   gt_dir = argv[7]; */

    if ((argc != 4)&&(argc != 5))
      iftError("Usage: iftSeedsFromSuperpixels P1 P2 P3 P4 (optional)\n"
	       "P1: input folder with superpixel (label) images (.png, .nii.gz)\n"
	       "P2: output folder with the seed files (-seeds.txt)\n"
	       "P3: label type: 0 (unlabeled), 1 (image class), 2 (object label)\n"
	       "P4: input folder with ground-truth images for label type 2 (optional)\n",
	       "main");
    
    tstart = iftTic();
    
    // Load input parameters
    
    iftFileSet  *fs       = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);    
    char ext[10];
    sprintf(ext,"%s",iftFileExt(fs->files[0]->path));
    char *out_dir         = argv[2];
    iftMakeDir(out_dir);
    int label_type        = atoi(argv[3]);
    char *gt_dir          = NULL;

    if ((label_type == 2)&&(argc != 5))
      iftError("GT images are required for label type 2","main");

    if ((label_type < 0)||(label_type>2))
      iftError("Label type must be in {0,1,2}","main");

    if (argc == 5)
      gt_dir = argv[4];
    
    char filename[300];
    
    for (int i=0; i < fs->n; i++){
      char *basename      = iftFilename(fs->files[i]->path, ext);
      iftImage *simg      = iftReadImageByExt(fs->files[i]->path);
      iftLabeledSet *S    = iftGeodesicCenters(simg);
      iftLabeledSet *seed = S;
      switch (label_type) {
      case 0: /* unlabeled */ 
	while (seed != NULL) {
	  seed->label = 1;
	  seed        = seed->next;
	}
	break;
      case 1:
	while (seed != NULL) {
	  seed->label = GetImageLabel(basename);
	  seed        = seed->next;
	}
	break;
      case 2:
	sprintf(filename,"%s/%s%s",gt_dir,basename,ext);
	iftImage *gt        = iftReadImageByExt(filename);
	bool binary_image   = false; 
	if (iftIsBinaryImage(gt))
	  binary_image = true;
	while (seed != NULL) {
	  int p       = seed->elem;
	  if (binary_image){
	    if (gt->val[p]!=0)
	      seed->label = 1;
	    else
	      seed->label = 0;
	  } else {
	    seed->label = gt->val[p];
	  }
	  seed        = seed->next;
	}
	iftDestroyImage(&gt);
	break;
      }
      
      /* /\* Reduce the number of seeds by clustering *\/ */
      /* sprintf(filename,"%s/%s.mimg",mimg_dir,basename); */
      /* iftMImage *input = iftReadMImage(filename); */
      /* if (iftIs3DMImage(input)) */
      /* 	A = iftSpheric(adj_radius); */
      /* else */
      /* 	A = iftCircular(adj_radius); */
      /* iftDataSet *Z    = iftDataSetFromLabeledSet(S,input,A); */
      /* iftDestroyAdjRel(&A); */
      /* iftDestroyMImage(&input); */
      /* // iftCosineDistance2 */
      /* iftKMeans(Z, nclusters, 100, 0.001, */
      /* 		NULL, NULL, iftEuclideanDistance); */
      /* seed = NULL; */
      /* for (int s=0; s < Z->nsamples; s++){ */
      /* 	if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)){ */
      /* 	  iftInsertLabeledSet(&seed,Z->sample[s].id,Z->sample[s].label); */
      /* 	} */
      /* }       */
      /* sprintf(filename,"%s/%s-seeds.txt",out_dir,basename); */
      /* iftWriteSeeds(seed,simg,filename); */
      /* iftDestroyLabeledSet(&seed); */
      /* iftDestroyDataSet(&Z); */

      sprintf(filename,"%s/%s-seeds.txt",out_dir,basename);
      iftWriteSeeds(S,simg,filename);
      iftDestroyLabeledSet(&S);
      iftDestroyImage(&simg);		  
      iftFree(basename);      
    }
    iftDestroyFileSet(&fs);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}


