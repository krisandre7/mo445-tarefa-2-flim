#include "ift.h"

#define color_space LABNorm2_CSPACE
/* 
   Author: Alexandre Xavier Falcão (November 2nd, 2023)

   Description: For multiple objects/classes, the strategy is to
   create one model per object/class. The outputs of the models
   (salience map/classes) can then be combined into a final decision
   layer based on comparisons among salience values or class
   probabilities.

   One can then provide markers labeled as object/background and
   distinct numbers of feature points per object and background
   markers can be extracted. This code assumes that labeled markers
   have been drawn on a few images and saved in a separated
   folder. You should draw markers with label 1 for object and label 0
   for background. You may use FLIMBuilder in
   libmo445/demo/Qt/FLIMBuilder/FLIMBuilder to do that. The code
   extracts circular/spheric patches of a given radius from marker
   pixels, computes Ni clusters and selects Ni feature points (cluster
   centers) per marker with label i={0,1}. The numbers Ni are input
   parameters. The code also adds a marker id to each marker pixel to
   facilitate the process.

*/


void AddMarkerLabels(iftImage *img, iftLabeledSet *S) {
    iftLabeledSet *seed    = S;
    iftImage      *markers = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iftAdjRel     *A = NULL;

    if (iftIs3DImage(img))
      A = iftSpheric(sqrtf(3.0));
    else
      A = iftCircular(sqrtf(2.0));

    while (seed != NULL) {
      int p = seed->elem;
      markers->val[p] = 255;
      seed = seed->next;
    }

    iftImage *lbmarkers = iftLabelComp(markers, A);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&markers);

    seed = S;
    while (seed != NULL) {
        int p = seed->elem;
	if (lbmarkers->val[p] != 0)
	  seed->marker = lbmarkers->val[p];
        seed = seed->next;
    }

    iftDestroyImage(&lbmarkers);
}

/* It creates a dataset in which voxels of a same marker are assigned
   to the same truelabel -- the marker id. Such samples are
   represented by 3x3 patches centered at the marker voxels. */

iftDataSet *ExtractMarkerDataSet(iftMImage *img, iftLabeledSet *S, iftAdjRel *A) {
  int nsamples    = iftLabeledSetSize(S);
  int tensor_size = img->m * A->n;
  iftDataSet *Z = iftCreateDataSet(nsamples, tensor_size);
  int ninput_channels = img->m;

  Z->nclasses = 0;
  int s       = 0;
  iftLabeledSet *seed = S;
  while (seed != NULL) {
    int p = seed->elem;
    Z->sample[s].id = seed->elem;
    Z->sample[s].truelabel = seed->marker;
    if (Z->sample[s].truelabel > Z->nclasses)
      Z->nclasses = Z->sample[s].truelabel;
    iftVoxel u = iftMGetVoxelCoord(img, p);
    int j = 0;
    for (int k = 0; k < A->n; k++) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, k);
      if (iftMValidVoxel(img, v)) {
	int q = iftMGetVoxelIndex(img, v);
	for (int b = 0; b < ninput_channels; b++) {
	  Z->sample[s].feat[j] = img->val[q][b];
	  j++;
	}
      } else {
	for (int b = 0; b < img->m; b++) {
	  Z->sample[s].feat[j] = 0;
	  j++;
	}
      }
    }
    s++;
    seed = seed->next;
  }
  
  iftSetStatus(Z, IFT_TRAIN);
  iftAddStatus(Z, IFT_SUPERVISED);
  
  return (Z);
}

iftLabeledSet *SelectClusterCenters(iftImage *img, iftLabeledSet *S, float adj_radius, int nclusters_per_bkg_marker, int nclusters_per_obj_marker)
{
  iftImage      *markers = iftCreateImage(img->xsize, img->ysize, img->zsize);
  iftImage      *labels  = iftCreateImage(img->xsize, img->ysize, img->zsize);
  iftLabeledSet *seed    = S, *T = NULL;

  /* Store marker id and label of each marker voxel */
  while (seed != NULL) {
    markers->val[seed->elem]=seed->marker;
    labels->val[seed->elem]=seed->label;
    seed = seed->next;
  }

  /* Extract the marker dataset: the marker id is the truelabel of the
     sample */
  
  iftMImage *mimg = iftImageToMImage(img, color_space);
  iftAdjRel *A    = NULL;
  if (iftIs3DImage(img)){
    A = iftSpheric(adj_radius);
  }else{
    A = iftCircular(adj_radius);
  }  
  iftDataSet *Z = ExtractMarkerDataSet(mimg,S,A);

  /* Estimate the given number of cluster centers per marker */
  
  for (int c = 1; c <= Z->nclasses; c++) {
    iftDataSet *Z1  = iftExtractSamplesFromClass(Z, c);
    int nclusters_per_marker;
    /* verify the label of a first pixel from the current marker */
    if (labels->val[Z1->sample[0].id]==0){ /* background marker */
      nclusters_per_marker = nclusters_per_bkg_marker;
    }else{ /* object marker */
      nclusters_per_marker = nclusters_per_obj_marker;
    }    
    int ngroups = iftMin(Z1->nsamples, nclusters_per_marker);
    iftKMeans(Z1, ngroups, 100, 0.001, NULL, NULL, iftEuclideanDistance);
    for (int s = 0; s < Z1->nsamples; s++) {
      if (iftHasSampleStatus(Z1->sample[s], IFT_PROTOTYPE)) {
	int p = Z1->sample[s].id;
	iftInsertLabeledSetMarkerAndHandicap(&T,p,
					     labels->val[p],
					     markers->val[p],
					     0);
      }
    }
    iftDestroyDataSet(&Z1);
  }
  
  iftDestroyImage(&markers);
  iftDestroyImage(&labels);
  iftDestroyAdjRel(&A);
  iftDestroyMImage(&mimg);
  iftDestroyDataSet(&Z);
  
  return(T);
}

int main(int argc, char *argv[]) {
    timer *tstart;

    /* Example: 

       iftBagOfFeaturePoints images markers 3.0 1 3 bag 

       It will define circular/spheric patches with radius 3, select 1
       feature point per background marker and 3 feature points per
       object marker. You may draw disks or strokes as markers. Each
       feature point will create one kernel at any given layer, but we
       will sort them based on some diversity criterion to obtain a
       given number of filters per layer.

    */ 
    
    if (argc!=7)
      iftError("Usage: iftBagOfFeatPoints <P1> <P2> <P3> <P4> <P5> <P6>\n"
	       "P1: image folder or csv file\n"
	       "P2: marker folder \n"
	       "P3: adj. radius for patch extraction \n"
	       "P4: number of feature points on background markers \n"
	       "P5: number of feature points on object markers \n"
	       "P6: new marker folder, representing a bag of feature points\n",
	       "main");

    tstart = iftTic();

    /* Read image filenames and get the files' extension */
    
    iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1], 0, 1);
    char ext[10];
    sprintf(ext,"%s",iftFileExt(fs->files[0]->path));
    float adj_radius = atof(argv[3]);
    int nfpts_per_bkg_marker = atoi(argv[4]);
    int nfpts_per_obj_marker = atoi(argv[5]);    
    char *out_dir = argv[6];
    iftMakeDir(out_dir);
    char *filename = iftAllocCharArray(512);
    
    /* Convert images to multichannel images and save them as layer0 */

    if (!iftDirExists("layer0")){
      iftMakeDir("layer0");
      for (int i = 0; i < fs->n; i++) {
	char *basename  = iftFilename(fs->files[i]->path, ext);
	iftImage *img   = iftReadImageByExt(fs->files[i]->path);
	iftMImage *mimg = NULL;	
	mimg = iftImageToMImage(img, color_space); /* even grayscale
						      images are
						      converted to
						      three-channel
						      images */
	
	sprintf(filename,"layer0/%s.mimg",basename);
	iftWriteMImage(mimg,filename);
	iftFree(basename);
	iftDestroyImage(&img);
	iftDestroyMImage(&mimg);
      }
    }
    
    char *path = iftDirname(fs->files[0]->path);
    iftDestroyFileSet(&fs);
    
    /* Select points from markers and save them as a bag of feature
       points */
    
    fs = iftLoadFileSetFromDirBySuffix(argv[2],"-seeds.txt", 1);  
    
    for (int i=0; i < fs->n; i++) {
      char *basename   = iftFilename(fs->files[i]->path,"-seeds.txt");
      sprintf(filename,"%s/%s%s",path,basename,ext);
      iftImage *img    = iftReadImageByExt(filename);
      iftLabeledSet *S = iftReadSeeds(img,fs->files[i]->path);
      AddMarkerLabels(img, S);      
      iftLabeledSet *T = SelectClusterCenters(img, S, adj_radius, nfpts_per_bkg_marker,nfpts_per_obj_marker);
      iftDestroyLabeledSet(&S);
      S = T;
      sprintf(filename,"%s/%s-fpts.txt",out_dir,basename);
      iftWriteLabeledSet(S,img->xsize,img->ysize,img->zsize,filename);
      iftFree(basename);
      iftDestroyImage(&img);
      iftDestroyLabeledSet(&S);
    }
    iftFree(path);
    iftFree(filename);    
    iftDestroyFileSet(&fs);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return (0);
}
