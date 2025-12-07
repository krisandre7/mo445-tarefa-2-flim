#include "ift.h"
#include "iftIterativeOPF.h"

#define InitNumberOfSeeds 2000
#define EXT3D "nii.gz"
#define EXT2D "png"

iftMImage    **ReadMImageSet(iftFileSet *fs);
void           DestroyMImageSet(iftMImage ***img, int nimages);
iftSet        *EstimateSupervoxelSeeds(iftMImage *img, iftImage *mask, int nsupervoxels, int ngroupsperimage, iftImage **supervoxels);
iftImage      *CreateMaskFromObjectNumber(iftImage *label, int object);
void           FindGroups(iftDataSet *Z, int ngroups);
iftSet        *SupervoxelRepresentatives(iftMImage *img, int ngroupsperimage, iftVoxelArray *samples);

iftSet *SupervoxelRepresentatives(iftMImage *img, int ngroupsperimage, iftVoxelArray *samples)
{
    iftDataSet *Z     = iftCreateDataSet(samples->n, img->m);
    iftSetStatus(Z,IFT_TRAIN);

    for (int s=0; s < samples->n; s++){
      if (iftMValidVoxel(img,samples->val[s])){
	int p           = iftMGetVoxelIndex(img,samples->val[s]);
	Z->sample[s].id = p;
	for (int b = 0; b < img->m; b++) {
	  Z->sample[s].feat[b] = img->val[p][b];
	}
      }
    }
    FindGroups(Z, ngroupsperimage);    
    iftSet *S = NULL;
    for (int s=0; s < Z->nsamples; s++){
      if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)){
	iftInsertSet(&S,Z->sample[s].id);
      }
    }
    iftDestroyDataSet(&Z);
    
    return(S);
}

iftSet *EstimateSupervoxelSeeds(iftMImage *img, iftImage *mask, int nsupervoxels, int ngroupsperimage, iftImage **supervoxels)
{
  iftSet *S    = NULL;    
  iftAdjRel *A = NULL;
  
  if (iftIs3DMImage(img)){
    A = iftSpheric(1.0);
  }else{
    A = iftCircular(1.0);
  }

  char local_mask=0;
  if (mask == NULL) {
    mask     = iftSelectImageDomain(img->xsize,img->ysize,img->zsize);
    local_mask = 1;
  } else {
    if ((img->xsize != mask->xsize)||(img->ysize != mask->ysize)||(img->zsize != mask->zsize))
      iftError("Input and mask images must have the same domain","EstimateSupervoxelSeeds");
  }

  int numinitseeds = InitNumberOfSeeds;
  
  if (iftIs3DMImage(img)){
    int maxnumofseeds = iftRound(pow(iftMin(iftMin(img->xsize,img->ysize),img->zsize)/5.0,3));
    numinitseeds = iftMin(numinitseeds,maxnumofseeds);
  }else{ 
    int maxnumofseeds = iftRound(pow(iftMin(img->xsize,img->ysize)/5.0,2));
    numinitseeds = iftMin(numinitseeds,maxnumofseeds);
  }
  
  if (!local_mask) {
    int masksize=0;
    for (int p=0; p < mask->n; p++)
      if (mask->val[p])
	masksize++;
    
    numinitseeds = iftMin(masksize/5.0,numinitseeds); 
  }
  
  numinitseeds = iftMax(numinitseeds, nsupervoxels);
  
  (*supervoxels) = iftDISF(img, A, numinitseeds, nsupervoxels, mask);

  iftVoxelArray *samples = iftGeometricCenterVoxelsLabelImage(*supervoxels);

  if (ngroupsperimage < nsupervoxels)
    S = SupervoxelRepresentatives(img, ngroupsperimage, samples);
  else {
    for (int i=0; i < samples->n; i++){
      if (iftValidVoxel(mask,samples->val[i])){
	int p = iftGetVoxelIndex(mask,samples->val[i]);
	iftInsertSet(&S,p);
      }
    }
  }
    
  iftDestroyVoxelArray(&samples);
  if (local_mask)
    iftDestroyImage(&mask);
  iftDestroyAdjRel(&A);

  return(S);
}

iftImage *CreateMaskFromObjectNumber(iftImage *label, int object)
{
  iftImage *mask;

  if (object == -1)
    mask = iftThreshold(label,1,IFT_INFINITY_INT,1);
  else
    mask = iftThreshold(label,object,object,1);

  return(mask);
}


/* Reads a set of multiband images */

iftMImage **ReadMImageSet(iftFileSet *fs)
{
    int first = 0;
    int last = fs->n - 1;

    iftMImage **img = (iftMImage **)calloc(fs->n, sizeof(iftMImage *));

    for (int i = first; i <= last; i++)
    {
        printf("Reading image: %d of %d\r", i + 1, (int)fs->n);
        fflush(stdout);
        img[i] = iftReadMImage(fs->files[i]->path);
    }
    printf("\n");

    return (img);
}

/* Deallocates memory for a multiband image set */

void DestroyMImageSet(iftMImage ***img, int nimages)
{
    iftMImage **aux = *img;

    for (int i = 0; i < nimages; i++)
        iftDestroyMImage(&aux[i]);
    free(aux);
    *img = NULL;
}

void FindGroups(iftDataSet *Z, int ngroups)
{

  /* iftGraph *graph    = NULL;  */
  /* graph = iftCreateGraph(Z); */
  /* graph->is_complete = true; */
  /* iftIteratedWatersheds(graph, ngroups, 100, 0, 0); */
  /* iftDestroyGraph(&graph); */

  iftKMeans(Z, ngroups, 100, 0.001, NULL, NULL);

}

int main(int argc, char *argv[])
{
    timer *tstart;

    if ((argc!=8)&&(argc!=9))
      iftError("Usage: iftSupervoxelDataSet <...>\n"
	       "[1] directory with iftMImage files \n"
	       "[2] number of supervoxels per image (e.g., 500, 1000)\n"
	       "[3] number of groups per image (e.g., 10, 50, 100)\n"
	       "[4] kernel size (e.g., 9 for 3x3, 27 for 3x3x3)\n"
	       "[5] number of groups from all images (e.g., 16, 64, 128)\n"
	       "[6] directory with the original images \n"
	       "[7] output dataset.zip to identify images and supervoxels for marker selection\n"
	       "[8] optional: object number (e.g., -1 for all, 1, 2, 3) when exists label images in [1]\n",
	       "main");

    tstart = iftTic();

    /* Load input image/activation set */
    
    iftFileSet *fs_input   = iftLoadFileSetFromDirBySuffix(argv[1],".mimg", 1);
    int nimages            = fs_input->n;   
    iftMImage **input      = ReadMImageSet(fs_input);
    int nsupervoxels       = atoi(argv[2]);
    int ngroupsperimage    = atoi(argv[3]);
    int kernelsize         = atoi(argv[4]);
    int nsamples           = ngroupsperimage*nimages;
    int ngroups            = atoi(argv[5]);
    int nclasses           = iftFileSetLabelsNumber(fs_input);
    iftAdjRel *A           = NULL;
    char *basename1,*basename2;
    basename2 = iftFilename(argv[7],".zip");
    iftMakeDir(basename2); 
    char filename[200];
    sprintf(filename,"%s.csv",basename2);
    FILE      *fp          = fopen(filename,"w");
    int patchsize          = 0;

    if (nsamples < ngroups)
      iftError("Number of groups cannot be greater than the number of samples","iftSupervoxelDataSet");
    
    if (iftIs3DMImage(input[0])){
      patchsize = cbrt(kernelsize);
      A        = iftCuboid(patchsize,patchsize,patchsize);
    }else{
      patchsize = sqrt(kernelsize);
      A = iftRectangular(patchsize,patchsize);
    }

    int nfeats             = input[0]->m*A->n;    

    /* Create supervoxel dataset */

    iftDataSet *Z     = iftCreateDataSet(nsamples, nfeats);
    iftFileSet *fsRef = iftCreateFileSet(nsamples);

    iftSetStatus(Z,IFT_TRAIN);
    Z->nclasses       = nclasses;

    int s = 0;
    for (int i=0; (i < nimages); i++) {
      basename1              = iftFilename(fs_input->files[i]->path,".mimg");
      
      printf("Processing file %s: %d of %d files\r", basename1, i + 1, nimages);
      fflush(stdout);

      iftImage *label=NULL, *mask=NULL;
      int object = 0;
      
      if (argc == 9){
	object = atoi(argv[8]);
	if (iftIs3DMImage(input[i])){
	  sprintf(filename,"%s/%s.%s",argv[1],basename1,EXT3D);
	} else {
	  sprintf(filename,"%s/%s.%s",argv[1],basename1,EXT2D);	  
	}
	label = iftReadImageByExt(filename);
	mask  = CreateMaskFromObjectNumber(label,object);
	iftDestroyImage(&label);
      }
      
      int truelabel          = fs_input->files[i]->label;
      iftImage *supervoxels  = NULL;
      iftSet *S              = EstimateSupervoxelSeeds(input[i], mask, nsupervoxels, ngroupsperimage, &supervoxels); 

      if (iftIs3DMImage(input[0]))
	sprintf(filename,"%s/%s.%s",basename2,basename1,EXT3D);
      else
	sprintf(filename,"%s/%s.%s",basename2,basename1,EXT2D);	

      iftWriteImageByExt(supervoxels,filename);
      
      if (mask != NULL) iftDestroyImage(&mask);

      int j=0;
      while ((j < nsupervoxels)&&(S != NULL)){
	int p = iftRemoveSet(&S);
	iftVoxel u = iftMGetVoxelCoord(input[i],p);
	sprintf(filename,"%s-%03d-%03d-%03d-%03d",basename1,u.x,u.y,u.z,supervoxels->val[p]);
	fprintf(fp,"%s\n",filename);	
	Z->sample[s].id        = s;
	Z->sample[s].truelabel = truelabel;
	int f = 0;
	for (int k=0; k < A->n; k++) {
	  iftVoxel v = iftGetAdjacentVoxel(A,u,k);
	  if (iftMValidVoxel(input[i],v)){
	    int q = iftMGetVoxelIndex(input[i],v);		
	    for (int b = 0; b < input[i]->m; b++) {
	      Z->sample[s].feat[f] = input[i]->val[q][b];
	      f++;
	    }
	  }
	}
	
	if (iftIs3DMImage(input[i]))
	  sprintf(filename, "%s/%s.%s", argv[6], basename1, EXT3D);
	else
	  sprintf(filename, "%s/%s.%s", argv[6], basename1, EXT2D);

	fsRef->files[s] = iftCreateFile(iftAbsPathname(filename));
	j++;
	s++;
      }

      iftDestroyImage(&supervoxels);
      iftFree(basename1);      
    }

    if (ngroups < Z->nsamples)
      FindGroups(Z, ngroups);
    else {
      for (int s=0; s < Z->nsamples; s++)
	Z->sample[s].group = s+1;
      Z->ngroups = Z->nsamples;
    }

    iftDestroyAdjRel(&A);
    fclose(fp);
    iftAddStatus(Z, IFT_SUPERVISED);
    /* if (s == nsamples)  */ /* It is making ClassifierLearning crash */
    /*   iftSetRefData(Z, fsRef, IFT_REF_DATA_FILESET); */
    iftWriteDataSet(Z, argv[7]);
    iftDestroyDataSet(&Z);
    DestroyMImageSet(&input,nimages);
    iftDestroyFileSet(&fs_input);
    iftDestroyFileSet(&fsRef);
    iftFree(basename2);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
