#include "ift.h"

/* Assign a subsequent integer number to each marker (connected
   component) of the seed files in a given folder. It will overwrite
   the seed files with the marker label information. */ 

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 3)
      iftError("Usage: iftLabelMarkers P1 P2 \n"
	       "P1: input folder with training images (.png, .nii.gz, or .mimg)\n"
	       "P2: input folder with seed files (-seeds.txt)\n",
	       "main");
    
    tstart = iftTic();

    // Load input parameters
    
    iftFileSet  *fs_seeds = iftLoadFileSetFromDirBySuffix(argv[2], \
							  "-seeds.txt", 1);

    char *basename        = iftFilename(fs_seeds->files[0]->path,	\
					"-seeds.txt");
    char filename[300], ext[10];
    sprintf(filename, "%s/%s%s", argv[1], basename, ".mimg");
    if (iftFileExists(filename)){
      sprintf(ext,".mimg");
    } else {
      sprintf(filename, "%s/%s%s", argv[1], basename, ".png");
      if (iftFileExists(filename)){
	sprintf(ext,".png");
      } else {
	sprintf(filename, "%s/%s%s", argv[1], basename, ".nii.gz");
	if (iftFileExists(filename)){
	  sprintf(ext,".nii.gz");
	} else {
	  iftError("Invalid input image format","main");
	}
      }
    }

    int incr=0;
    
    for (int i=0; i < fs_seeds->n; i++){
      basename         = iftFilename(fs_seeds->files[i]->path,	\
				    "-seeds.txt");
      sprintf(filename, "%s/%s%s", argv[1], basename, ext);
      iftMImage  *mimg = NULL;
      iftImage   *img  = NULL;
      /* An example that might be useful in other codes */
      if (strcmp(ext,".png")==0){
	img  = iftReadImageByExt(filename);
	if (iftIsColorImage(img)){
	  mimg = iftImageToMImage(img, LABNorm2_CSPACE);
	}else{
	  mimg = iftImageToMImage(img, GRAY_CSPACE);
	}
      } else {
	if (strcmp(ext,".nii.gz")==0) {
	  img  = iftReadImageByExt(filename);
	  mimg = iftImageToMImage(img, GRAY_CSPACE); 
	} else {
	  mimg = iftReadMImage(filename);
	}
      }
      iftDestroyImage(&img);
      /* Label markers */
      iftLabeledSet *S  = iftMReadSeeds(mimg, fs_seeds->files[i]->path);      
      iftLabeledSet *M  = NULL, *seed = S;
      iftImage *markers = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
      iftAdjRel *A      = NULL;
      if (iftIs3DMImage(mimg))
	A = iftSpheric(sqrtf(3.0));
      else
	A = iftCircular(sqrtf(2.0));

      /* Create a binary image with the markers */
      
      while (seed != NULL) {
        int p = seed->elem;
        markers->val[p] = 255;
        seed = seed->next;
      }

      /* Label it from 1, 2,..,n */ 
      iftImage *lbmarkers = iftLabelComp(markers, A);
      iftDestroyAdjRel(&A);
      iftDestroyImage(&markers);

      /* Copy the seeds and labels by adding a label increment */ 
      seed = S;
      while (seed != NULL) {
        int p = seed->elem, label = seed->label;
        iftInsertLabeledSetMarkerAndHandicap(&M, p, label, \
					     lbmarkers->val[p]+incr, 0);
        seed = seed->next;
      }
      /* Update the label increment with the number of marker labels
	 already used */
      incr += iftNumberOfMarkers(M);
      /* Overwrite the original seed file */
      iftMWriteSeeds(M, mimg, fs_seeds->files[i]->path);      

      /* Destroy allocated memory */
      iftDestroyImage(&lbmarkers);
      iftFree(basename);
      iftDestroyLabeledSet(&M);
      iftDestroyMImage(&mimg);
    }
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}


