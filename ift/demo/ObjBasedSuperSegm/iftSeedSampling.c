#include <ift.h>

/**********************************************************************
                              DEFINITIONS
**********************************************************************/
#define PROG_DESC \
        "This program samples seeds in accordance of the strategy given and the number "\
        "of desired markers. It also considers the object probability map extracted via "\
        " <iftExtrObjSalMap.c> program, if given."
#define IMG_DESC \
        "Input image"
#define K_DESC \
        "Number of seed points (x > 0)"
#define METHOD_DESC \
        "Sampling method option: (0) GRID; (1) MIX; (2) OBJ-GRID; (3) OBJ-BORDER; "\
        "(4) OSMOX; (5) OSMOX on Borders"
#define OBJ_PERC_DESC \
        "Percentage of seeds within the (probable) object location"
#define MAP_THR_DESC \
        "Threshold value for the object saliency map (if needed)"
#define MASK_DESC \
        "Mask indicating the reachable pixels"
#define OBJSAL_DESC \
        "Object saliency map"
#define OUT_DESC \
        "Output image containing the seeds (!=0)"

/**********************************************************************
                                METHODS
**********************************************************************/
iftDict *iftGetArgs
(int argc, const char *argv[]);

void iftGetAndValReqArgs
(  iftDict *args, char **img_path, int *k, int *method_op, char **out_path);

void iftGetAndValOptArgs
(  iftDict *args, char **mask_path, char **objsal_path, double *obj_perc, double *map_thr);

iftMImage *iftExtendMImageByObjSalMap
(iftMImage *mimg, iftImage* objsm );

iftImage *iftSamplingByOGRIDOnBorders
(iftImage *objsm, iftImage *mask, int k, float thresh, float obj_perc);

iftImage *_iftSamplingByOSMOX
(iftImage* objsm, iftImage *mask, int num_seeds, float obj_perc);

iftImage *iftSamplingByOSMOXOnBorders
(iftImage* objsm, iftImage *mask, int num_seeds, float obj_perc);

iftSet *iftObjSalMapSamplByValueWithAreaSum
(iftImage *objsm, iftImage *mask, int num_seeds);

/**********************************************************************
																MAIN
**********************************************************************/
int main(int argc, const char *argv[]) 
{
	int k, method_op;
  double obj_perc, map_thr;
	char *img_path, *mask_path, *objsal_path, *out_path;
  iftImage  *img, *mask, *objsm, *seeds;
  iftMImage *mimg;
  iftDict* args;

  // Obtaining user inputs
  args = iftGetArgs(argc, argv);

  iftGetAndValReqArgs(args, &img_path, &k, &method_op, &out_path);
  iftGetAndValOptArgs(args, &mask_path, &objsal_path, &obj_perc, &map_thr);

  // Attrib
  img = iftReadImageByExt(img_path);
  
  if( mask_path != NULL ) mask = iftReadImageByExt(mask_path);
  else mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);

  if( objsal_path != NULL ) objsm = iftReadImageByExt(objsal_path);
  else objsm = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);

  if ( iftIsColorImage(img) ) mimg = iftImageToMImage(img,LABNorm_CSPACE);
  else mimg = iftImageToMImage(img,GRAY_CSPACE);

  // Init
  if( method_op == 0 ) seeds = iftGridSampling(mimg, mask, k);
  else if( method_op == 1 ) seeds = iftAltMixedSampling(mimg, mask, k);
  else if( method_op == 2 ) seeds = iftGrayObjMapGridSamplOnMaskByArea(objsm, mask, k, (float)map_thr, (float)obj_perc);
  else if( method_op == 3 ) seeds = iftSamplingByOGRIDOnBorders(objsm, mask, k, (float)map_thr, (float)obj_perc);
  else if( method_op == 4 ) seeds = iftSamplingByOSMOX(objsm, mask, k, (float)obj_perc, 6.0);
  else if( method_op == 5 ) seeds = iftSamplingByOSMOXOnBorders(objsm, mask, k, (float)obj_perc);

  // End
  printf("Number of seeds %d\n", iftNumberOfElements(seeds));
  iftWriteImageByExt(seeds, out_path);

  // Free
  iftFree(img_path);
  if( mask_path != NULL ) iftFree(mask_path);
  if( objsal_path != NULL ) iftFree(objsal_path);
  iftFree(out_path);
  iftDestroyImage(&img);
  iftDestroyImage(&seeds);
  iftDestroyImage(&mask);
  iftDestroyImage(&objsm);
  iftDestroyMImage(&mimg);
  iftDestroyDict(&args);

  return 0;
}

/**********************************************************************
																METHODS
**********************************************************************/
iftDict *iftGetArgs
(int argc, const char *argv[])
{
	char descr[2048] = PROG_DESC;
	int n_opts;
	iftCmdLineParser *parser;
	iftDict *args;

	iftCmdLineOpt cmd[] = {
		// Required
		{.short_name = "-i", .long_name = "--input-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
			.required = true, .help = IMG_DESC},
		{.short_name = "-k", .long_name = "--nseeds", .has_arg = true, .arg_type = IFT_LONG_TYPE,
			.required = true, .help = K_DESC},
		{.short_name = "-f", .long_name = "--method-op", .has_arg = true, .arg_type = IFT_LONG_TYPE,
			.required = true, .help = METHOD_DESC},
		{.short_name = "-o", .long_name = "--out-img", .has_arg = true, .arg_type=IFT_STR_TYPE,
			.required = true, .help = OUT_DESC},
		// Optional
		{.short_name = "-m", .long_name = "--mask-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
			.required = false, .help = MASK_DESC},
		{.short_name = "-sm", .long_name = "--obj-sm", .has_arg = true, .arg_type = IFT_STR_TYPE,
			.required = false, .help = OBJSAL_DESC},
    {.short_name = "-op", .long_name = "--obj-perc", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = OBJ_PERC_DESC},
    {.short_name = "-tm", .long_name = "--thresh-map", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = MAP_THR_DESC},
	};
	n_opts = sizeof(cmd) / sizeof (iftCmdLineOpt);

	// Parser Setup
	parser = iftCreateCmdLineParser(descr, n_opts, cmd);
	args = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

	// Free
	iftDestroyCmdLineParser(&parser);

	return args;
}

void iftGetAndValReqArgs
(  iftDict *args, char **img_path, int *k, int *method_op, char **out_path)
{
	*img_path	= iftGetStrValFromDict("--input-img", args);
	if (!iftFileExists(*img_path)) 
		iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *img_path);

	*k = iftGetLongValFromDict("--nseeds", args);
	if( *k <= 0 ) 
		iftError("Invalid number of seeds: \"%d\"", "iftGetAndValReqArgs", *k);

	*method_op = iftGetLongValFromDict("--method-op", args);
	if( *method_op < 0 || *method_op > 5 ) 
		iftError("Invalid sampling option: \"%d\"", "iftGetAndValReqArgs", *method_op);

	*out_path = iftGetStrValFromDict("--out-img", args);
	if (!iftIsImagePathnameValid(*out_path)) 
		iftError("Invalid output image: \"%s\"", "iftGetAndValReqArgs", *out_path);  
}

void iftGetAndValOptArgs
(  iftDict *args, char **mask_path, char **objsal_path, double *obj_perc, double *map_thr)
{
  if( iftDictContainKey("--mask-img", args, NULL)) {
  	*mask_path = iftGetStrValFromDict("--mask-img", args);	

  	if (!iftFileExists(*mask_path)) 
  		iftError("Non-existent image: \"%s\"", "iftGetAndValOptArgs", *mask_path);

  } else *mask_path = NULL;
  
  if( iftDictContainKey("--obj-sm", args, NULL)) {
  	*objsal_path = iftGetStrValFromDict("--obj-sm", args);	

  	if (!iftFileExists(*objsal_path)) 
  		iftError("Non-existent image: \"%s\"", "iftGetAndValOptArgs", *objsal_path);
  } else *objsal_path = NULL;

  if( iftDictContainKey("--obj-perc", args, NULL)) {
    *obj_perc = iftGetDblValFromDict("--obj-perc", args);

    if( *obj_perc < 0.0 || *obj_perc > 1.0) 
      iftError("Invalid object percentage value: \"%f\"", "iftGetAndValOptArgs", *obj_perc);
  } else *obj_perc = 0.9;

  if( iftDictContainKey("--thresh-map", args, NULL)) {
    *map_thr = iftGetDblValFromDict("--thresh-map", args);

    if( *map_thr < 0.0 || *map_thr > 1.0) 
      iftError("Invalid threshold value: \"%f\"", "iftGetAndValOptArgs", *map_thr);
  } else *map_thr = 0.5;
}

iftMImage *iftExtendMImageByObjSalMap
(iftMImage *mimg, iftImage* objsm )
{
	int p,b;
	iftMImage *emimg;

	emimg = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m+1);

  for ( p = 0; p < mimg->n; p++)  {
  	for(b = 0; b < mimg->m; b++ )  emimg->val[p][b] = mimg->val[p][b];

  	emimg->val[p][mimg->m] = ((float)objsm->val[p]);
  }

  return emimg;
}

iftImage *iftSamplingByOGRIDOnBorders
(iftImage *objsm, iftImage *mask, int k, float thresh, float obj_perc)
{
  int seedsForBkg, seedsForObj, maxVal;
  iftImage *bin, *invBin, *ero, *border, *seeds;
  iftSet *bkgSeeds, *objSeeds, *S;
  iftAdjRel *A;

  objSeeds = NULL;
  bkgSeeds = NULL;
  S = NULL;

  seedsForObj = iftRound(k * obj_perc);
  maxVal = iftMaximumValue(objsm);

  if( iftIs3DImage(objsm) ) A = iftSpheric(5.0);
  else A = iftCircular(5.0);

  seeds = iftCreateImage(objsm->xsize, objsm->ysize, objsm->zsize);
  bin = iftThreshold(objsm, thresh*maxVal, maxVal, 1);
  ero = iftErode(bin, A, NULL);
  border = iftSub(bin, ero);
  invBin = iftComplement(border);

  objSeeds = iftMultiLabelGridSamplingOnMaskByArea(border, mask, seedsForObj);
  
  seedsForBkg = abs(k - iftSetSize(objSeeds)) + 1;
  bkgSeeds = iftMultiLabelGridSamplingOnMaskByArea(invBin, mask, seedsForBkg);  

  S = objSeeds;
  while (S != NULL) { seeds->val[S->elem] = 1; S = S->next;}
  iftDestroySet(&S);

  S = bkgSeeds;
  while (S != NULL) { seeds->val[S->elem] = 1; S = S->next; }  

  // Free
  iftDestroyImage(&bin);
  iftDestroyImage(&ero);
  iftDestroyImage(&border);
  iftDestroyImage(&invBin);
  iftDestroySet(&objSeeds);
  iftDestroySet(&bkgSeeds);
  iftDestroySet(&S);
  iftDestroyAdjRel(&A);
  
  return seeds;
}

iftImage *_iftSamplingByOSMOX
(iftImage* objsm, iftImage *mask, int num_seeds, float obj_perc)
{
  if(objsm->n < num_seeds || num_seeds < 0) {
    iftError("Invalid number of seeds!", "iftSamplingByOSMOX");
  } 
  else if(obj_perc < 0.0 || obj_perc > 1.0) {
    iftError("Invalid object percentage!", "iftSamplingByOSMOX");
  }
  if(mask != NULL) iftVerifyImageDomains(objsm, mask, "iftSamplingByOSMOX");

  int obj_seeds, bkg_seeds, max_val, min_val;
  iftSet *obj_set, *bkg_set, *s;
  iftImage *seed_img, *mask_copy, *invsm;;

  obj_set = NULL;
  bkg_set = NULL;
  s = NULL;
  
  iftMinMaxValues(objsm, &min_val, &max_val);

  if( mask == NULL ) mask_copy = iftSelectImageDomain(objsm->xsize, objsm->ysize, objsm->zsize);
  else mask_copy = iftCopyImage(mask);

  seed_img = iftCreateImage(objsm->xsize, objsm->ysize, objsm->zsize);

  // Establish the number of seeds
  if(max_val == min_val)
  {
    obj_seeds = num_seeds;
    bkg_seeds = 0;
  }
  else
  {
    obj_seeds = iftRound(num_seeds * obj_perc);
    bkg_seeds = num_seeds - obj_seeds;
  }
  
  obj_set = iftObjSalMapSamplByValueWithAreaSum(objsm, mask_copy, obj_seeds);

  invsm = iftComplement(objsm);

  s = obj_set;
  while( s != NULL ) {
    mask_copy->val[s->elem] = 0;
    seed_img->val[s->elem] = 1;
    s = s->next;
  }

  bkg_set = iftObjSalMapSamplByValueWithAreaSum(invsm, mask_copy, bkg_seeds);

  s = bkg_set;
  while( s != NULL ) {
    seed_img->val[s->elem] = 1;
    s = s->next;
  }

  // Free
  iftDestroyImage(&invsm);
  iftDestroySet(&obj_set);
  iftDestroySet(&bkg_set);
  iftDestroySet(&s);
  iftDestroyImage(&mask_copy);

  return (seed_img);
}

iftImage *iftSamplingByOSMOXOnBorders
(iftImage *objsm, iftImage *mask, int num_seeds, float obj_perc)
{
  int seedsForBkg, seedsForObj, max_val, min_val;
  iftImage *dil_img, *ero_img, *inner_border, *outer_border, *seed_img, *mask_copy;
  iftSet *bkg_seeds, *obj_seeds, *S;
  iftAdjRel *A;

  obj_seeds = NULL;
  bkg_seeds = NULL;
  S = NULL;

  if( mask == NULL ) mask_copy = iftSelectImageDomain(objsm->xsize, objsm->ysize, objsm->zsize);
  else mask_copy = iftCopyImage(mask);

  seedsForObj = iftRound(num_seeds * obj_perc);
  iftMinMaxValues(objsm, &min_val, &max_val);

  if( iftIs3DImage(objsm) ) A = iftSpheric(5.0);
  else A = iftCircular(5.0);

  seed_img = iftCreateImage(objsm->xsize, objsm->ysize, objsm->zsize);

  // Establish the number of seeds
  if(max_val == min_val)
  {
    seedsForObj = num_seeds;
    seedsForBkg = 0;
  }
  else
  {
    seedsForObj = iftRound(num_seeds * obj_perc);
    seedsForBkg = num_seeds - seedsForObj;
  }

  dil_img = iftDilate(objsm, A, mask_copy);
  ero_img = iftErode(objsm, A, mask_copy);

  inner_border = iftSub(objsm, ero_img);
  outer_border = iftSub(dil_img, objsm);

  obj_seeds = iftObjSalMapSamplByValueWithAreaSum(inner_border, mask_copy, seedsForObj);

  S = obj_seeds;
  while( S != NULL ) {
    mask_copy->val[S->elem] = 0;
    seed_img->val[S->elem] = 1;
    S = S->next;
  }

  bkg_seeds = iftObjSalMapSamplByValueWithAreaSum(outer_border, mask_copy, seedsForBkg);

  S = bkg_seeds;
  while( S != NULL ) {
    mask_copy->val[S->elem] = 0;
    seed_img->val[S->elem] = 1;
    S = S->next;
  }

  // Free
  iftDestroySet(&obj_seeds);
  iftDestroySet(&bkg_seeds);
  iftDestroySet(&S);
  iftDestroyImage(&mask_copy);
  iftDestroyImage(&dil_img);
  iftDestroyImage(&ero_img);
  iftDestroyImage(&inner_border);
  iftDestroyImage(&outer_border);
  iftDestroyAdjRel(&A);
  
  return seed_img;
}

iftSet *iftObjSalMapSamplByValueWithAreaSum
(iftImage *objsm, iftImage *mask, int num_seeds)
{
  if(objsm->n < num_seeds || num_seeds < 0) {
    iftError("Invalid number of seeds!", "iftSamplingByOSMOX");
  }

  if(mask != NULL) iftVerifyImageDomains(objsm, mask, "iftObjSalMapSamplByHighestValue");

  int patch_width, seed_count, total_area;
  float stdev;
  double *pixel_val;
  iftImage *mask_copy;
  iftAdjRel *A, *B;
  iftKernel *gaussian;
  iftDHeap *heap;
  iftSet *seed;
  
  // Copy the mask values
  if( mask == NULL ) mask_copy = iftSelectImageDomain(objsm->xsize, objsm->ysize, objsm->zsize);
  else mask_copy = mask;
  
  total_area = 0;

  #pragma omp parallel for reduction(+:total_area)
  for(int p = 0; p < objsm->n; p++)
  {
    if(mask_copy->val[p] != 0)
    {
      total_area++;
    }
  }

  // Estimate the gaussian influence zone
  patch_width = iftRound(sqrtf(total_area/(float)(num_seeds)));
  stdev = patch_width/6.0;

  if(iftIs3DImage(objsm)) { A = iftSpheric(patch_width); B = iftSpheric(sqrtf(patch_width));}
  else { A = iftCircular(patch_width); B = iftCircular(sqrtf(patch_width));}
  
  // Copy of the object saliency map values
  pixel_val = (double *)calloc(objsm->n, sizeof(double));
  heap = iftCreateDHeap(objsm->n, pixel_val);

  // Gaussian penalty
  gaussian = iftCreateKernel(A);

  #pragma omp parallel for
  for(int i = 0; i < A->n; i++) {
    float dist;

    dist = A->dx[i]*A->dx[i] + A->dy[i]*A->dy[i] + A->dz[i]*A->dz[i];
    gaussian->weight[i] = exp(-dist/(2*stdev*stdev));
  }

  seed = NULL;

  // Removing the highest values
  iftSetRemovalPolicyDHeap(heap, MAXVALUE);
  #pragma omp parallel for
  for( int p = 0; p < objsm->n; p++ ) {
    if(mask_copy->val[p] != 0) {
      iftVoxel p_voxel;
      
      p_voxel = iftGetVoxelCoord(objsm, p);
      
      pixel_val[p] = objsm->val[p];
      for(int i = 1; i < B->n; i++)
      {
        iftVoxel q_voxel;
        
        q_voxel = iftGetAdjacentVoxel(B, p_voxel, i);
        
        if(iftValidVoxel(objsm, q_voxel))
        {
          int q;
          
          q = iftGetVoxelIndex(objsm, q_voxel);
          
          pixel_val[p] += objsm->val[q];  
        }
      }
    }
    else pixel_val[p] = IFT_NIL;
  }
  
  for( int p = 0; p < objsm->n; p++ ) {
    if(pixel_val[p] != IFT_NIL) iftInsertDHeap(heap, p);
  }

  seed_count = 0;
  while( seed_count < num_seeds && !iftEmptyDHeap(heap) ) {
    int p;
    iftVoxel voxel_p;

    p = iftRemoveDHeap(heap);
    voxel_p = iftGetVoxelCoord(objsm, p);

    iftInsertSet(&seed, p);

    // Mark as removed
    pixel_val[p] = IFT_NIL;

    // For every adjacent voxel in the influence zone
    for(int i = 1; i < gaussian->A->n; i++) {
      iftVoxel voxel_q;

      voxel_q = iftGetAdjacentVoxel(gaussian->A, voxel_p, i);

      if(iftValidVoxel(objsm, voxel_q)) {
        int q;

        q = iftGetVoxelIndex(objsm, voxel_q);
        
        // If it was not removed (yet)  
        if( pixel_val[q] != IFT_NIL ) {
          // Penalize
          pixel_val[q] = (1.0 - gaussian->weight[i]) * pixel_val[q];

          iftGoDownDHeap(heap, heap->pos[q]);
        }
      }
    }

    seed_count++;
  }

  // Free
  iftDestroyKernel(&gaussian);
  iftDestroyDHeap(&heap);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  if(mask == NULL) iftDestroyImage(&mask_copy);

  return (seed);
}
