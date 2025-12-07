#include "iftRunOISF.h"

/**********************************************************************
																MAIN
**********************************************************************/
int main(int argc, const char *argv[]) 
{
	int iter, smooth;
  double alpha, gamma, beta;
	char *img_path, *mask_path, *seed_path, *objsal_path, *out_path;
  iftImage  *img, *mask, *objsm, *seeds, *labels;
  iftIGraph *igraph;
  iftDict* args;

  // Obtaining user inputs
  args = iftGetArgs(argc, argv);

  iftGetAndValReqArgs(args, &img_path, &seed_path, &out_path);
  iftGetAndValOptArgs(args, &mask_path, &objsal_path, &alpha, &beta, &gamma, &iter, &smooth);

  img = iftReadImageByExt(img_path);
  seeds = iftReadImageByExt(seed_path);

  if( mask_path == NULL ) mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);
  else mask = iftReadImageByExt(mask_path);

  if( objsal_path == NULL ) objsm = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);
  else objsm = iftReadImageByExt(objsal_path);
  
  igraph = iftInitOISFIGraph( img, mask, objsm );

  iftOISFSegm(igraph, seeds, alpha, beta, gamma, iter);

  if( smooth > 0 ) {
    iftIGraphSetWeightForRegionSmoothing(igraph, img);
    iftIGraphSmoothRegions(igraph, smooth);
  }

  labels = iftIGraphLabel(igraph);

  // End
  iftWriteImageByExt(labels, out_path);

  // Free
  iftFree(img_path);
  iftFree(seed_path);
  iftFree(out_path);
  if( mask_path != NULL ) iftFree(mask_path);
  if( objsal_path != NULL ) iftFree(objsal_path);
  iftDestroyImage(&img);
  iftDestroyImage(&seeds);
  iftDestroyImage(&mask);
  iftDestroyImage(&objsm);
  iftDestroyImage(&labels);
  iftDestroyIGraph(&igraph);
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
    {.short_name = "-si", .long_name = "--seed-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
      .required = true, .help = SEED_DESC},
    {.short_name = "-o", .long_name = "--out-img", .has_arg = true, .arg_type=IFT_STR_TYPE,
      .required = true, .help = OUT_DESC},
    // Optional
    {.short_name = "-sm", .long_name = "--objsm-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
      .required = false, .help = OBJSAL_DESC},
    {.short_name = "-mi", .long_name = "--mask-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
      .required = false, .help = MASK_DESC},
    {.short_name = "-a", .long_name = "--alpha", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = ALPHA_DESC},
    {.short_name = "-b", .long_name = "--beta", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = BETA_DESC},
    {.short_name = "-g", .long_name = "--gamma", .has_arg = true, .arg_type = IFT_DBL_TYPE,
      .required = false, .help = GAMMA_DESC},
    {.short_name = "-it", .long_name = "--iters", .has_arg = true, .arg_type = IFT_LONG_TYPE,
      .required = false, .help = ITER_DESC},
    {.short_name = "-is", .long_name = "--smooth", .has_arg = true, .arg_type = IFT_LONG_TYPE,
      .required = false, .help = SMOOTH_DESC},
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
(  iftDict *args, char **img_path, char **seed_path,  char **out_path)
{
  *img_path = iftGetStrValFromDict("--input-img", args);
  if (!iftFileExists(*img_path)) 
    iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *img_path);

  *seed_path = iftGetStrValFromDict("--seed-img", args);
  if (!iftFileExists(*seed_path)) 
    iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *seed_path);

  *out_path = iftGetStrValFromDict("--out-img", args);
  if (!iftIsImagePathnameValid(*out_path)) 
    iftError("Invalid output image: \"%s\"", "iftGetAndValReqArgs", *out_path);  
}

void iftGetAndValOptArgs
(  iftDict *args, char **mask_path, char **objsal_path, double *alpha, double *beta, double *gamma, int *iter, int *smooth)
{
  if( iftDictContainKey("--mask-img", args, NULL)) {
    *mask_path = iftGetStrValFromDict("--mask-img", args);  

    if (!iftFileExists(*mask_path)) 
      iftError("Non-existent image: \"%s\"", "iftGetAndValOptArgs", *mask_path);

  } else *mask_path = NULL;
  
  if( iftDictContainKey("--objsm-img", args, NULL)) {
    *objsal_path = iftGetStrValFromDict("--objsm-img", args);  

    if (!iftFileExists(*objsal_path)) 
      iftError("Non-existent image: \"%s\"", "iftGetAndValOptArgs", *objsal_path);

  } else *objsal_path = NULL;

  if( iftDictContainKey("--alpha", args, NULL)) {
    *alpha = iftGetDblValFromDict("--alpha", args);

    if( *alpha < 0.0 ) 
      iftError("Negative compactness factor: \"%f\"", "iftGetAndValOptArgs", *alpha);
  } else *alpha = 0.5;

  if( iftDictContainKey("--beta", args, NULL)) {
    *beta = iftGetDblValFromDict("--beta", args);

    if( *beta < 0.0 ) 
      iftError("Negative boundary adherence factor: \"%f\"", "iftGetAndValOptArgs", *beta);
  } else *beta = 12.0;

  if( iftDictContainKey("--gamma", args, NULL)) {
    *gamma = iftGetDblValFromDict("--gamma", args);

    if( *gamma < 0.0 ) 
      iftError("Negative object map confidence: \"%f\"", "iftGetAndValOptArgs", *gamma);
  } else *gamma = 1.5;

  if( iftDictContainKey("--iters", args, NULL)) {
    *iter = iftGetLongValFromDict("--iters", args);

    if( *iter <= 0 ) 
      iftError("Non-positive iterations for segmentation: \"%d\"", "iftGetAndValOptArgs", *iter);
  } else *iter = 1;

  if( iftDictContainKey("--smooth", args, NULL)) {
    *smooth = iftGetLongValFromDict("--smooth", args);

    if( *smooth < 0 ) 
      iftError("Negative iterations for segmentation: \"%d\"", "iftGetAndValOptArgs", *smooth);
  } else *smooth = 2;
}

iftIGraph *iftInitOISFIGraph
(iftImage *img, iftImage *mask, iftImage *objsm)
{
  iftMImage *mimg, *obj_mimg;
  iftAdjRel *A;
  iftIGraph *igraph;

  if (iftIs3DImage(img)) A = iftSpheric(1.0);
  else A = iftCircular(1.0);

  if (iftIsColorImage(img)) mimg   = iftImageToMImage(img,LABNorm_CSPACE);
  else mimg   = iftImageToMImage(img,GRAY_CSPACE);

  obj_mimg = iftExtendMImageByObjSalMap(mimg, objsm);

  igraph = iftImplicitIGraph( obj_mimg, mask, A);

  //Free
  iftDestroyMImage(&mimg);
  iftDestroyMImage(&obj_mimg);
  iftDestroyAdjRel(&A);

  return igraph;
}

iftMImage *iftExtendMImageByObjSalMap
(iftMImage *mimg, iftImage* objsm )
{
  int p,b;
  iftMImage *emimg;

  emimg = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m+1);

  for ( p = 0; p < mimg->n; p++)  {
    
    for(b = 0; b < mimg->m; b++ )  emimg->val[p][b] = mimg->val[p][b];

    emimg->band[mimg->m].val[p] = ((float)objsm->val[p]);
  }

  return emimg;
}

void iftOISFSegm
(iftIGraph *igraph, iftImage *seeds, double alpha, double beta, double gamma, int iters )
{
 double tmp;
  int r, s, t, i, p, q, it, nseeds;
  int *seed, *center;
  double color_dist, geo_dist, obj_dist;
  iftVoxel u, v;
  iftDHeap *Q;
  double *pvalue;
  iftSet *S, *new_seeds, *frontier_nodes, *trees_rm;

  nseeds = 0;

  S = NULL;
  new_seeds = NULL;
  frontier_nodes = NULL;
  trees_rm = NULL;

  pvalue = iftAllocDoubleArray(igraph->nnodes);
  Q = iftCreateDHeap(igraph->nnodes, pvalue);

  for (s=0; s < igraph->nnodes; s++) 
  {
      p               = igraph->node[s].voxel;
      pvalue[s]       = IFT_INFINITY_DBL;
      igraph->pvalue[p] = IFT_INFINITY_DBL;
      igraph->pred[p] = IFT_NIL;
      
      if (seeds->val[p]!=0)
      {
          iftInsertSet(&new_seeds,s);
          nseeds++;
      }
  }

  seed = iftAllocIntArray(nseeds);
  S = new_seeds; 
  i = 0;

  while (S != NULL) 
  {
      seed[i] = S->elem;
      p       = igraph->node[seed[i]].voxel;
      igraph->label[p] = i+1;
      i++; 
      S = S->next;
  }

  for (it=0; it < iters; it++) 
  {
    printf("Iteration: %d\n",it + 1 );

    if (trees_rm != NULL)
    { 
      frontier_nodes = iftIGraphTreeRemoval(igraph, &trees_rm, pvalue, IFT_INFINITY_DBL);
    }

    while (new_seeds != NULL) 
    {
      s = iftRemoveSet(&new_seeds);
      p = igraph->node[s].voxel;  

      if (igraph->label[p] > 0)
      { 
        pvalue[s] = 0;
        igraph->pvalue[p] = 0;
        igraph->root[p] = p;
        igraph->pred[p] = IFT_NIL;
        iftInsertDHeap(Q,s);
      }
    }

    while (frontier_nodes != NULL) 
    {
      s = iftRemoveSet(&frontier_nodes);
      
      if (Q->color[s] == IFT_WHITE) iftInsertDHeap(Q,s);
    } 

    while (!iftEmptyDHeap(Q)) 
    {
      s = iftRemoveDHeap(Q);
      p = igraph->node[s].voxel;
      r = igraph->root[p];
      igraph->pvalue[p] = pvalue[s];
      u = iftGetVoxelCoord(igraph->index,p);

      for (i=1; i < igraph->A->n; i++) 
      {
        v = iftGetAdjacentVoxel(igraph->A,u,i);
        if (iftValidVoxel(igraph->index,v))
        {
          q   = iftGetVoxelIndex(igraph->index,v);
          t   = igraph->index->val[q];
          if ((t != IFT_NIL) && (Q->color[t] != IFT_BLACK))
          {

            color_dist = (double)iftFeatDistance(igraph->feat[r], igraph->feat[q], igraph->nfeats-1);
            geo_dist = (double)iftVoxelDistance(u,v);
            obj_dist = (double)(abs((igraph->feat[r][igraph->nfeats-1] - igraph->feat[q][igraph->nfeats-1]))/255.0);

            tmp = pow( alpha*color_dist*pow(gamma, obj_dist) +gamma*obj_dist, beta);
            tmp += geo_dist;
            tmp += pvalue[s];

            if (tmp < pvalue[t])
            {
              pvalue[t]            = tmp;

              igraph->root[q]      = igraph->root[p];
              igraph->label[q]     = igraph->label[p];
              igraph->pred[q]      = p;

              if (Q->color[t] == IFT_GRAY) iftGoUpDHeap(Q, Q->pos[t]);
              else iftInsertDHeap(Q,t);
            } 
            else 
            {
              if (igraph->pred[q] == p)
              {
                if (tmp > pvalue[t]) iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
                else 
                {
                  if ((igraph->label[q] != igraph->label[p])&&(igraph->label[q]!=0))
                  {
                    iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
                  }
                }
              }
            }
          } 
        }
      }
    } 

    iftResetDHeap(Q);

    if( iters > 1 ) {
      center = iftIGraphSuperpixelCenters(igraph, seed, nseeds);
      iftIGraphEvalAndAssignNewSeeds(igraph, center, seed, nseeds, &trees_rm, &new_seeds);
    }
  }

  // Free
  iftDestroySet(&S);
  iftDestroySet(&new_seeds);
  iftDestroySet(&frontier_nodes);
  iftDestroySet(&trees_rm);
  iftDestroyDHeap(&Q);
  iftFree(pvalue);
  iftFree(seed);
}

void iftIGraphEvalAndAssignNewSeeds
(iftIGraph *igraph, int* center, int *seed, int nseeds, iftSet **trees_rm, iftSet **new_seeds )
{
  int i, p, q, s;
  iftVoxel u, v;
  float distColor, distVoxel, distColorThres, distVoxelThres;

  distColorThres = 0.0;  
  distVoxelThres = 0.0;
  for (s=0; s < igraph->nnodes; s++) {
    p     = igraph->node[s].voxel;
    i     = igraph->label[igraph->root[p]]-1;
    q     = igraph->node[seed[i]].voxel;
    u = iftGetVoxelCoord(igraph->index,p);
    v = iftGetVoxelCoord(igraph->index,q);
    distColor = iftFeatDistance(igraph->feat[p],igraph->feat[q],igraph->nfeats);
    distColorThres += distColor;
    distVoxel = iftVoxelDistance(u,v);
    distVoxelThres += distVoxel;
  }
  
  distColorThres /= igraph->nnodes;
  distVoxelThres /= igraph->nnodes;
  distColorThres = sqrtf(distColorThres);
  distVoxelThres = sqrtf(distVoxelThres);

  /* Verify if the centers can be new seeds */

  for (i=0; i < nseeds; i++) {
    p = igraph->node[seed[i]].voxel;
    q = igraph->node[center[i]].voxel;
    u = iftGetVoxelCoord(igraph->index,p);
    v = iftGetVoxelCoord(igraph->index,q);
    distColor = iftFeatDistance(igraph->feat[p],igraph->feat[q],igraph->nfeats);
    distVoxel = iftVoxelDistance(u,v);

    if ((distColor > distColorThres)||(distVoxel > distVoxelThres)){
      seed[i] = center[i];
      iftInsertSet(new_seeds,center[i]);
      iftInsertSet(trees_rm,seed[i]);
    }
  }
}