#include "iftCenterSampling.h"

/**********************************************************************
																MAIN
**********************************************************************/
int main(int argc, const char *argv[]) 
{
	int k;
  float out_rad, thr_objsm, thr_seed;
	char *img_path, *mask_path, *objsal_path, *out_path;
  iftImage  *img, *mask, *objsm, *seeds;
  iftDict* args;

  // Obtaining user inputs
  args = iftGetArgs(argc, argv);

  iftGetAndValReqArgs(args, &img_path, &k, &objsal_path, &out_path);
  iftGetAndValOptArgs(args, &mask_path, &out_rad, &thr_objsm, &thr_seed);

  // // Attrib
  img = iftReadImageByExt(img_path);
  objsm = iftReadImageByExt(objsal_path);

  if( mask_path != NULL ) mask = iftReadImageByExt(mask_path);
  else mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);

  // Sampling
  seeds = iftCENTERSampling(objsm, mask, k, out_rad, thr_objsm, thr_seed);

  // // End
  printf("Number of seeds %d\n", iftNumberOfElements(seeds));
  iftWriteImageByExt(seeds, out_path);

  // Free
  iftFree(img_path);
  if( mask_path != NULL ) iftFree(mask_path);
  iftFree(objsal_path);
  iftFree(out_path);
  iftDestroyImage(&img);
  iftDestroyImage(&seeds);
  iftDestroyImage(&mask);
  iftDestroyImage(&objsm);
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
		{.short_name = "-sm", .long_name = "--obj-sm", .has_arg = true, .arg_type = IFT_STR_TYPE,
      .required = false, .help = OBJSAL_DESC},
		{.short_name = "-o", .long_name = "--out-img", .has_arg = true, .arg_type=IFT_STR_TYPE,
			.required = true, .help = OUT_DESC},
		// Optional
		{.short_name = "-m", .long_name = "--mask-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
			.required = false, .help = MASK_DESC},
    {.short_name = "-or", .long_name = "--out-rad", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = OUT_RAD_DESC},
    {.short_name = "-tm", .long_name = "--thr-objsm", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = THR_OBJSM_DESC},
    {.short_name = "-ts", .long_name = "--thr-seed", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = THR_SEED_DESC},
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
(  iftDict *args, char **img_path, int *k, char **objsal_path, char **out_path)
{
	*img_path	= iftGetStrValFromDict("--input-img", args);
	if (!iftFileExists(*img_path)) 
		iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *img_path);

	*k = iftGetLongValFromDict("--nseeds", args);
	if( *k <= 0 ) 
		iftError("Invalid number of seeds: \"%d\"", "iftGetAndValReqArgs", *k);

	*objsal_path  = iftGetStrValFromDict("--obj-sm", args);
  if (!iftFileExists(*objsal_path)) 
    iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *objsal_path);

	*out_path = iftGetStrValFromDict("--out-img", args);
	if (!iftIsImagePathnameValid(*out_path)) 
		iftError("Invalid output image: \"%s\"", "iftGetAndValReqArgs", *out_path);  
}

void iftGetAndValOptArgs
(  iftDict *args, char **mask_path, float *out_rad, float *thr_objsm, float *thr_seed)
{
  if( iftDictContainKey("--mask-img", args, NULL)) {
  	*mask_path = iftGetStrValFromDict("--mask-img", args);	

  	if (!iftFileExists(*mask_path)) 
  		iftError("Non-existent image: \"%s\"", "iftGetAndValOptArgs", *mask_path);

  } else *mask_path = NULL;
  
  if( iftDictContainKey("--out-rad", args, NULL)) {
  	*out_rad = iftGetDblValFromDict("--out-rad", args);	

  	if (*out_rad < 0.0) 
  		iftError("Negative radius: \"%f\"", "iftGetAndValOptArgs", *out_rad);

  } else *out_rad = 5.0;

  if( iftDictContainKey("--thr-objsm", args, NULL)) {
    *thr_objsm = iftGetDblValFromDict("--thr-objsm", args); 

    if (*thr_objsm < 0.0) 
      iftError("Negative threshold: \"%f\"", "iftGetAndValOptArgs", *thr_objsm);

  } else *thr_objsm = 0.2;

  if( iftDictContainKey("--thr-seed", args, NULL)) {
    *thr_seed = iftGetDblValFromDict("--thr-seed", args); 

    if (*thr_seed < 0.0) 
      iftError("Negative threshold: \"%f\"", "iftGetAndValOptArgs", *thr_seed);

  } else *thr_seed = 0.5;
}

iftImage *iftCENTERSampling
(iftImage *objsm, iftImage *mask, int k, float out_rad, float thr_objsm, float thr_seed )
{
  int maxVal;
  iftImage *bin, *invBin, *seeds, *ero, *ero_obj;
  iftSet *bkgSeeds, *objSeeds, *S;

  objSeeds = NULL;
  bkgSeeds = NULL;
  S = NULL;

  maxVal = iftMaximumValue(objsm);

  seeds = iftCreateImage(objsm->xsize, objsm->ysize, objsm->zsize);
  bin = iftThreshold(objsm, thr_objsm*maxVal, maxVal, 1);
  invBin = iftComplement(bin);
  
  ero = iftErodeLabelImage(invBin, out_rad);
  ero_obj = iftErodeLabelImage(bin, 5.0);

  bkgSeeds = iftMultiLabelGridSamplingOnMaskByArea(ero, mask, k);  
  objSeeds = iftMultiLabelCenterSamplingOnMaskByArea(ero_obj, mask, thr_seed);  

  S = objSeeds;
  while (S != NULL) { seeds->val[S->elem] = 1; S = S->next;}
  iftDestroySet(&S);

  S = bkgSeeds;
  while (S != NULL) { seeds->val[S->elem] = 1; S = S->next; }  

  // Free
  iftDestroyImage(&bin);
  iftDestroyImage(&ero);
  iftDestroyImage(&ero_obj);
  iftDestroyImage(&invBin);
  iftDestroySet(&objSeeds);
  iftDestroySet(&bkgSeeds);
  iftDestroySet(&S);
  
  return seeds;
}

iftSet *iftMultiLabelCenterSamplingOnMaskByArea
(  iftImage *label, iftImage *mask, float thresh)
{
  int totalArea, numObj;
  iftImage *newLabels;
  iftAdjRel *A;
  iftSet *seeds;
  iftIntArray *sampled;

  seeds = NULL;

  if( iftIs3DImage(label) ) A = iftSpheric(1.74);
  else A = iftCircular(1.45);

  newLabels = iftFastLabelComp(label, A);
  numObj = iftMaximumValue(newLabels);
  sampled = iftCreateIntArray(numObj);

  // Sum of component areas
  totalArea = 0;
  #pragma omp parallel for reduction(+:totalArea)
  for( int p = 0; p < newLabels->n; p++ )
  {
    if(newLabels->val[p] > 0) totalArea++;
  }
  
  // For each labeled component
  for( int i = 1; i <= numObj; i++ )
  {
    iftImage* objMask;
    int objArea;

    objArea = 0;

    objMask = iftThreshold(newLabels, i, i, 1);
    
    #pragma omp parallel for reduction(+:objArea)  
    for( int p = 0; p < objMask->n; p++ )
    {
      if(objMask->val[p] > 0) objArea++;
    }

    float objPerc = objArea / (float)totalArea;

    if( objPerc >= thresh ) {
      int center_index;
      iftVoxelArray *centers;
      
      centers = iftGeometricCentersFromLabelImage(objMask);

      center_index = iftGetVoxelIndex(objMask, centers->val[1]);

      if( mask->val[center_index] != 0 ) iftInsertSet(&seeds, center_index);

      // Free
      iftDestroyVoxelArray(&centers);
    }

    // Free
    iftDestroyImage(&objMask);
  }
  
  // Free
  iftDestroyImage(&newLabels);
  iftDestroyAdjRel(&A);
  iftDestroyIntArray(&sampled);
  
  return seeds;

}
