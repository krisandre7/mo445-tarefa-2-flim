#include "ift.h"

/* 
   Alexandre Falcão (December 15th, 2024): This code receives a patch
   dataset in which the truelabel of the samples represent distinct
   classes or objects. It computes clusters in each class, selects the
   cluster centers, computes an optimum-path forest using them as
   nodes, and selects the leaf nodes in each class as filters. It also
   outputs images in a graphics folder with the indication of the
   positions of the selected filters and the corresponding seeds in a
   marker folder.

*/

void SaveBias(char *basepath, float *bias, int number_of_kernels)
{
  char filename[200];
  FILE *fp;
  
  sprintf(filename, "%s-bias.txt", basepath);
  fp = fopen(filename, "w");
  fprintf(fp, "%d\n", number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++) {
    fprintf(fp, "%f ", bias[k]);
  }
  fclose(fp);
}

void SaveKernelWeights(char *basepath, int *truelabel, int number_of_kernels)
{
  char filename[200];
  FILE *fp;
  float pw=1, nw=-1;
  
  sprintf(filename, "%s-weights.txt", basepath);
  fp = fopen(filename, "w");
  fprintf(fp, "%d\n", number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++) {
    if (truelabel[k]==2)
      fprintf(fp, "%f ", pw);
    else
      fprintf(fp, "%f ", nw);      
  }
  fclose(fp);
}

iftDataSet *Clusters(iftDataSet *Z, int nclusters, \
		     iftKmeansDistance distance)
{  
  if (distance == NULL)
    distance = iftEuclideanDistance;
    
  /* Find the desired number of filters per class */

  iftDataSet *Zf   = iftCreateDataSet(nclusters,Z->nfeats);
  iftCSV *Zflist   = iftCreateCSV(Zf->nsamples,2);
  iftCSV *Zlist    = (iftCSV *)Z->ref_data;

  /* Compute clusters */
  iftKMeans(Z, nclusters, 100, 0.001,
	    NULL, NULL, distance);
  
  /* Create a root map */

  for (int s=0; s < Z->nsamples; s++){  
    if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)){
      Z->sample[s].label = s;
    } else {
      for (int t=0; t < Z->nsamples; t++){  
	if ((s != t)&&
	    iftHasSampleStatus(Z->sample[t], IFT_PROTOTYPE)&&
	    (Z->sample[s].group == Z->sample[t].group)
	    ){
	  Z->sample[s].label = t;	      
	}
      }
    }
  }
    
  /* Compute the size of each group and set the root sample's weight
     to the percentual of samples in its group */

  for (int s=0; s < Z->nsamples; s++){  
    Z->sample[Z->sample[s].label].weight = Z->sample[Z->sample[s].label].weight + 1.0/Z->nsamples;
    }

  /* Copy group information to Z and create Zf with the prototypes */
    
  for (int s=0, t=0; s < Z->nsamples; s++){
    if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)){
      Zf->sample[t].id        = t;
      Zf->sample[t].group     = Z->sample[s].group;
      Zf->sample[t].weight    = Z->sample[s].weight;
      strcpy(Zflist->data[t][0],Zlist->data[s][0]);
      strcpy(Zflist->data[t][1],Zlist->data[s][1]);
      for (int f=0; f < Zf->nfeats; f++)
	Zf->sample[t].feat[f] = Z->sample[s].feat[f];
      t++;
    }
  }
    
  Zf->ngroups  = Z->ngroups;
  iftSetRefData(Zf,Zflist,IFT_REF_DATA_CSV);
  iftSetStatus(Zf, IFT_TRAIN);
  
  return(Zf);
}

iftDataSet *ClustersPerClass(iftDataSet *Z, int nclusters_per_class, \
			     iftKmeansDistance distance)
{  
  if (distance == NULL)
    distance = iftEuclideanDistance;
    
  /* Find the desired number of filters per class */

  iftDataSet *Zf   = iftCreateDataSet(nclusters_per_class*Z->nclasses,Z->nfeats);
  iftCSV *Zflist   = iftCreateCSV(Zf->nsamples,2);
  iftCSV *Zlist    = (iftCSV *)Z->ref_data;

  int incr_g     = 0;
  Z->ngroups     = 0;
  for (int c  = 1; c <= Z->nclasses; c++) {
    iftDataSet *Z1  = iftExtractSamplesFromClass(Z, c);

    /* Compute clusters for class c */
    iftKMeans(Z1, nclusters_per_class, 100, 0.001,
	      NULL, NULL, distance);

    /* Create a root map */

    for (int s=0; s < Z1->nsamples; s++){  
      if (iftHasSampleStatus(Z1->sample[s], IFT_PROTOTYPE)){
	Z1->sample[s].label = s;
      } else {
	for (int t=0; t < Z1->nsamples; t++){  
	  if ((s != t)&&
	      iftHasSampleStatus(Z1->sample[t], IFT_PROTOTYPE)&&
	      (Z1->sample[s].group == Z1->sample[t].group)
	      ){
	    Z1->sample[s].label = t;	      
	  }
	}
      }
    }
    
    /* Compute the size of each group and set the root sample's weight
       to the percentual of samples in its group */

    for (int s=0; s < Z1->nsamples; s++){  
      Z1->sample[Z1->sample[s].label].weight = Z1->sample[Z1->sample[s].label].weight + 1.0/Z1->nsamples;
    }

    /* Copy group information to Z and create Zf with the prototypes */
    
    for (int i=0, j=0, k=0; i < Z->nsamples; i++){
      if (Z->sample[i].id == Z1->sample[j].id){
	Z->sample[i].group = Z1->sample[j].group + incr_g;
	if (Z->ngroups < Z->sample[i].group)
	  Z->ngroups = Z->sample[i].group;
	if (iftHasSampleStatus(Z1->sample[j], IFT_PROTOTYPE)){
	  iftAddSampleStatus(&Z->sample[i],IFT_PROTOTYPE);	  
	  Z->sample[i].weight            = Z1->sample[j].weight;
	  Zf->sample[k+incr_g].id        = k+incr_g;
	  Zf->sample[k+incr_g].label     = Z->sample[i].label;
	  Zf->sample[k+incr_g].truelabel = Z->sample[i].truelabel;
	  Zf->sample[k+incr_g].group     = Z->sample[i].group;
	  Zf->sample[k+incr_g].weight    = Z->sample[i].weight;
	  strcpy(Zflist->data[k+incr_g][0],Zlist->data[i][0]);
	  strcpy(Zflist->data[k+incr_g][1],Zlist->data[i][1]);
	  for (int f=0; f < Zf->nfeats; f++)
	    Zf->sample[k+incr_g].feat[f] = Z->sample[i].feat[f];
	  k++;
	}
	j++;
      }
    }
    iftDestroyDataSet(&Z1);
    incr_g = Z->ngroups;
  }
    
  Zf->nclasses = Z->nclasses;
  Zf->ngroups  = Z->ngroups;
  iftSetRefData(Zf,Zflist,IFT_REF_DATA_CSV);
  iftSetStatus(Zf, IFT_TRAIN);
  iftAddStatus(Zf, IFT_SUPERVISED);
  
  return(Zf);
}

/* Returns another dataset with the patches (cluster centers from
   iftClustersPerClass) that present the highest path cost to the
   prototypes (the most discriminative between classes) */

iftDataSet *iftClusterCentersByOPF(iftDataSet *Zin)
{
  iftDataSet    *Zout     = NULL;
  iftCplGraph   *graph    = iftCreateCplGraph(Zin); // Create complete graph 
  iftLabeledSet *leaf_set = NULL;
  int            nleaves  = 0;
  
  /* Compute the optimum-path forest with prototypes in each class */
  
  iftSupTrain(graph);
  
  /* Count the leaves of the forest */

  for (int u = 0; u < graph->nnodes; u++) {
      int s = graph->node[u].sample;
      bool is_leaf = true;
      for (int v = 0; (v < graph->nnodes)&&(is_leaf); v++) {
	if (u != v) {
	  if (graph->node[v].pred == u)
	    is_leaf = false;
	}
      }
      if (is_leaf){
	nleaves++;
	iftInsertLabeledSet(&leaf_set,s,Zin->sample[s].label);
      }
  }
  iftDestroyCplGraph(&graph);
  
  /* Create dataset with leaf samples only */

  Zout             = iftCreateDataSet(nleaves,Zin->nfeats);
  iftCSV *out_list = iftCreateCSV(Zout->nsamples,2);
  iftCSV *in_list  = (iftCSV *)Zin->ref_data;

  
  int t = 0;
  while (leaf_set != NULL) {
    int label;
    int s = iftRemoveLabeledSet(&leaf_set,&label);
    Zout->sample[t].id        = t;
    Zout->sample[t].truelabel = label;
    strcpy(out_list->data[t][0],in_list->data[s][0]);
    strcpy(out_list->data[t][1],in_list->data[s][1]);
    for (int f=0; f < Zout->nfeats; f++)
      Zout->sample[t].feat[f] = Zin->sample[s].feat[f];
    t++;
  }
    
  Zout->nclasses = Zin->nclasses;
  iftSetRefData(Zout,out_list,IFT_REF_DATA_CSV);
  iftSetStatus(Zout, IFT_TRAIN);
  iftAddStatus(Zout, IFT_SUPERVISED);
  
  return(Zout);  
}

char *GetBasename(char *img_path, char *ext){
  iftSList *info1  = iftSplitString(img_path,"/");
  iftSNode *node1  = info1->tail;
  iftSList *info2  = iftSplitString(node1->elem,".");
  iftSNode *node2  = info2->head;
  char *basename   = iftFilename(node2->elem,ext);
  iftDestroySList(&info1);
  iftDestroySList(&info2);
  return(basename);
}

/* Draw selected filters when the images are 2D */

void DrawPointsFromDataSet(iftDataSet *Zf, iftAdjRel *A, char *images_dir, char *model_dir, int layer)
{
  iftCSV *ilist = Zf->ref_data;
  char ext[10];
  char *basename  = NULL, *prev_basename = NULL;
  iftMImage *mimg = NULL;
  iftImage  *img  = NULL;
  char graphics_path[200];
  char filename[300];
  
  sprintf(graphics_path,"%s/graphics",model_dir);
  if (iftDirExists(graphics_path))
    iftRemoveFile(graphics_path);
  iftMakeDir(graphics_path);

  iftColorTable *ctb = NULL;
  int norm_val       = 1;
  
  for (int s=0; s < Zf->nsamples; s++) {
    int p           = atoi(ilist->data[s][0]);
    char *img_path  = ilist->data[s][1];
    
    if (basename==NULL){
      mimg = iftReadMImage(img_path);
      if (iftIs3DMImage(mimg)){
	iftDestroyMImage(&mimg);
	return;
      }else{
	sprintf(ext,"%s",".png");
      }
      prev_basename = basename = GetBasename(img_path,ext);
      sprintf(filename,"%s/%s%s",images_dir,basename,ext);
      img  = iftReadImageByExt(filename);
      if (ctb == NULL){
	norm_val = iftNormalizationValue(iftMaximumValue(img));
	ctb      = iftCreateRandomColorTable(30,norm_val);
      }
    } else {
      basename = GetBasename(img_path,ext);
      if (strcmp(basename,prev_basename)!=0){
	sprintf(filename,"%s/%s_layer%d%s",graphics_path,prev_basename,layer,ext);
	iftWriteImageByExt(img,filename);
	iftFree(prev_basename);
	iftDestroyMImage(&mimg);
	iftDestroyImage(&img);

	mimg = iftReadMImage(img_path);
	sprintf(filename,"%s/%s_layer%d%s",graphics_path,basename,layer,ext);
	if (iftFileExists(filename)){	  
	  img = iftReadImageByExt(filename);
	}else{
	  sprintf(filename,"%s/%s%s",images_dir,basename,ext);
	  img = iftReadImageByExt(filename);
	}	    
	prev_basename = basename;
      }
    }
    
    float scale[3];
    scale[0] = img->xsize/mimg->xsize;
    scale[1] = img->ysize/mimg->ysize;
    scale[2] = img->zsize/mimg->zsize;
    iftVoxel u, v;
    u        = iftMGetVoxelCoord(mimg,p);
    v.x      = u.x * scale[0];
    v.y      = u.y * scale[1];
    v.z      = u.z * scale[2];
    iftDrawPoint(img, v, ctb->color[Zf->sample[s].truelabel], A, norm_val);
  }

  if ((graphics_path!=NULL)&&(prev_basename!=NULL)){
    sprintf(filename,"%s/%s_layer%d%s",graphics_path,prev_basename,layer,ext);
    iftWriteImageByExt(img,filename);
    iftFree(prev_basename);
    iftDestroyMImage(&mimg);
    iftDestroyImage(&img);
  }

  if (ctb == NULL)
    iftDestroyColorTable(&ctb);
}

void SaveMarkersFromDataSet(iftDataSet *Zf, char *images_dir, char *model_dir, int layer)
{
  iftCSV *ilist = Zf->ref_data;
  char ext[10];
  char *basename   = NULL, *prev_basename = NULL;
  iftMImage *mimg  = NULL;
  iftImage  *img   = NULL;
  iftLabeledSet *S = NULL;
  char markers_path[200];
  char filename[300];
  
  sprintf(markers_path,"%s/markers%d",model_dir,layer);
  if (iftDirExists(markers_path))
    iftRemoveFile(markers_path);
  iftMakeDir(markers_path);
  
  for (int s=0; s < Zf->nsamples; s++) {
    int p            = atoi(ilist->data[s][0]);
    char *img_path   = ilist->data[s][1];
    
    if (basename==NULL){
      mimg = iftReadMImage(img_path);
      if (iftIs3DMImage(mimg)){
	sprintf(ext,"%s",".nii.gz");
      }else{
	sprintf(ext,"%s",".png");
      }
      prev_basename = basename = GetBasename(img_path,ext);
      sprintf(filename,"%s/%s%s",images_dir,basename,ext);
      img  = iftReadImageByExt(filename);
    } else {
      basename = GetBasename(img_path,ext);
      if (strcmp(basename,prev_basename)!=0){
	sprintf(filename,"%s/%s-seeds.txt",markers_path,prev_basename);
	iftWriteSeeds(S,img,filename);
	iftDestroyLabeledSet(&S);
	iftFree(prev_basename);
	iftDestroyImage(&img);
	iftDestroyMImage(&mimg);
	mimg = iftReadMImage(img_path);
	sprintf(filename,"%s/%s%s",images_dir,basename,ext);
	img = iftReadImageByExt(filename);
      }	    
      prev_basename = basename;
    }
    float scale[3];
    scale[0] = img->xsize/mimg->xsize;
    scale[1] = img->ysize/mimg->ysize;
    scale[2] = img->zsize/mimg->zsize;
    iftVoxel u, v;
    u        = iftMGetVoxelCoord(mimg,p);
    v.x      = u.x * scale[0];
    v.y      = u.y * scale[1];
    v.z      = u.z * scale[2];
    int q    = iftGetVoxelIndex(img,v);
    //    if (Zf->sample[s].weight > 0.02)
    iftInsertLabeledSet(&S,q,Zf->sample[s].truelabel);
  }

  if ((markers_path!=NULL)&&(prev_basename!=NULL)){
    sprintf(filename,"%s/%s-seeds.txt",markers_path,prev_basename);
    iftWriteSeeds(S,img,filename);
    iftDestroyLabeledSet(&S);
    iftFree(prev_basename);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
  }
}

/* 
   Estimate a given number of filters per class using clustering.
*/

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 6)
      iftError("Usage: iftFiltersFromPatchDataset P1 P2 P3 P4\n"
	       "P1: folder with the model, containing datasets (.zip)\n"
	       "P2: model's architecture\n" 
	       "P3: layer for filter estimation (1, 2, 3, etc)\n"
	       "P4: number of filters\n"
	       "P5: folder with the original images (.png, .nii.gz)\n",	       
	       "main");
    
    tstart = iftTic();

    char filename[300];
    char *model_dir          = argv[1];
    char *arch_file          = argv[2];
    iftFLIMArch *arch        = iftReadFLIMArch(arch_file);
    int  layer               = atoi(argv[3]);
    int  nclusters           = atoi(argv[4]);
    char *images_dir         = argv[5];
    sprintf(filename, "%s/dataset%d.zip", model_dir,layer);
    iftDataSet *Z = iftReadDataSet(filename);
    int  nclusters_per_class = 1;
    iftDataSet *Zf           = NULL;
    
    /* Update the dataset with group and prototype information,
       creating a new dataset with the prototypes only. You may use
       iftCosineDistance2 or iftEuclideanDistance for k-means
       clustering */

    if (Z->nclasses > 1){
      nclusters_per_class = iftMax(1,nclusters/Z->nclasses);
      Zf = ClustersPerClass(Z,nclusters_per_class,iftEuclideanDistance);
    } else {
      Zf = Clusters(Z,nclusters,iftEuclideanDistance);
    }
    
    /* Select filters as leaves of the optimum-path forest, when the
       above Zf is substituted by Z1 as the dataset of clusters per
       class */
    /* iftDataSet *Zf = iftClusterCentersByOPF(Z1);  */
    /* iftDestroyDataSet(&Z1);  */

    /* Normalize dataset */
    
    iftNormalizeDataSetByZScoreInPlace(Zf,NULL,arch->stdev_factor);
    sprintf(filename, "%s/filters%d.zip", model_dir,layer);
    iftWriteDataSet(Zf,filename);
    
    /* Compute filters, biases, and their truelabels. Update
       architecture with the number of filters. */
    
    iftMatrix *filters = iftCreateMatrix(Zf->nsamples, Zf->nfeats);
    int *truelabel     = iftAllocIntArray(Zf->nsamples);
      
    for (int s = 0, col=0; s < Zf->nsamples; s++) {
      iftUnitNorm(Zf->sample[s].feat, Zf->nfeats);
      truelabel[col] = Zf->sample[s].truelabel;
      for (int row = 0; row < Zf->nfeats; row++){
	iftMatrixElem(filters, col, row) = Zf->sample[s].feat[row];
      }
      col++;
    }
    
    arch->layer[layer-1].noutput_channels = filters->ncols;
    iftWriteFLIMArch(arch,arch_file);
  
    float *bias = NULL;
    bias        = iftAllocFloatArray(filters->ncols);
    for (int col=0; col < filters->ncols; col++){
      for (int row=0; row < Zf->nfeats; row++){
	iftMatrixElem(filters,col,row) =
	  iftMatrixElem(filters,col,row) / Zf->fsp.stdev[row];
	bias[col] -= (Zf->fsp.mean[row]*iftMatrixElem(filters,col,row));
      }
    }

    /* Draw points at the locations of the selected filters, whenever
       images are 2D. */

    iftAdjRel *A = iftCircular(3.0);
    DrawPointsFromDataSet(Zf,A,images_dir,model_dir,layer);
    iftDestroyAdjRel(&A);
    
    /* save filters, biases, and truelabels of the filters */
    
    sprintf(filename, "%s/conv%d-kernels.npy", model_dir, layer);
    iftWriteMatrix(filters,filename);
    sprintf(filename, "%s/conv%d", model_dir, layer);
    SaveBias(filename, bias, filters->ncols);
    SaveKernelWeights(filename, truelabel, filters->ncols);

    /* save markers in the input-image resolution */

    SaveMarkersFromDataSet(Zf, images_dir, model_dir, layer);

    /* Free memory */
    
    iftFree(bias);
    iftFree(truelabel);
    iftDestroyMatrix(&filters);
    iftDestroyDataSet(&Z);
    iftDestroyDataSet(&Zf);
    iftDestroyFLIMArch(&arch);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
