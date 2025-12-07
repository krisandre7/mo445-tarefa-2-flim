#include <ift.h>

/**********************************************************************
                              DEFINITIONS
**********************************************************************/
#define PROG_DESC \
        "Performs the Object-based Iterative Spanning Forest superpixel segmentation. "\
        "It iteratively computes an Image Foresting Transform over improved seeds "\
        "with respect of a given object saliency map, and seed image (seed != 0). If "\
        "no object map is given, a blank image will be used."
#define IMG_DESC \
        "Input image"
#define SEED_DESC \
        "Seed image (seed != 0)"
#define OBJSAL_DESC \
        "Object saliency map"
#define MASK_DESC \
        "Mask indicating the reachable pixels"
#define ALPHA_DESC \
        "Compactness factor (x >= 0)"
#define BETA_DESC \
        "Boundary adherence factor (x >= 0)"
#define GAMMA_DESC \
        "Object saliency map relevance (x > 0)"
#define ITER_DESC \
        "Number of iterations for segmentation (x > 0)"
#define SMOOTH_DESC \
        "Number of iterations for smoothing (x >= 0)"
#define PATHFUNC_DESC \
        "Path-cost function option: (0) ISMM'17; (1) MAX-GAMMA;"
#define SEEDRECOMP_DESC \
        "Seed recomputation option: (0) FEAT;"
#define OUT_DESC \
        "Output image"
/**********************************************************************
                                METHODS
**********************************************************************/

iftDict *iftGetArgs
(int argc, const char *argv[]);

void iftGetAndValReqArgs
(  iftDict *args, char **img_path, char **seed_path,  char **out_path, int *pf_op, int *rs_op);

void iftGetAndValOptArgs
(  iftDict *args, char **mask_path, char **objsal_path, double *alpha, double *beta, double *gamma, int *iter, int *smooth);

iftIGraph *iftInitOISFIGraph
(iftImage *img, iftImage *mask, iftImage *objsm);

iftMImage *iftExtendMImageByObjSalMap
(iftMImage *mimg, iftImage* objsm );

void iftOISFSegm
(iftIGraph *igraph, iftImage *seeds, double alpha, double beta, double gamma, int iters, int pf_op, int sr_op );

void iftIGraphEvalAndAssignNewSeeds
(iftIGraph *igraph, int* center, int *seed, int nseeds, iftSet **trees_rm, iftSet **new_seeds );

int *_iftIGraphSuperpixelCenters(iftIGraph *igraph, int *seed, int nseeds);

void _iftIGraphSubTreeRemoval(iftIGraph *igraph, int s, double *pvalue, double INITIAL_PATH_VALUE, iftDHeap *Q);

iftSet *_iftIGraphTreeRemoval(iftIGraph *igraph, iftSet **trees_for_removal, double *pvalue, double INITIAL_PATH_VALUE);

/**********************************************************************
																MAIN
**********************************************************************/
int main(int argc, const char *argv[]) 
{
	int pf_op, sr_op, iter, smooth;
  double alpha, gamma, beta;
	char *img_path, *mask_path, *seed_path, *objsal_path, *out_path;
  iftImage  *img, *mask, *objsm, *seeds, *labels;
  iftIGraph *igraph;
  iftDict* args;

  // Obtaining user inputs
  args = iftGetArgs(argc, argv);

  iftGetAndValReqArgs(args, &img_path, &seed_path, &out_path, &pf_op, &sr_op);
  iftGetAndValOptArgs(args, &mask_path, &objsal_path, &alpha, &beta, &gamma, &iter, &smooth);

  img = iftReadImageByExt(img_path);
  seeds = iftReadImageByExt(seed_path);

  if( mask_path == NULL ) mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);
  else mask = iftReadImageByExt(mask_path);

  if( objsal_path == NULL ) objsm = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);
  else objsm = iftReadImageByExt(objsal_path);
  
  igraph = iftInitOISFIGraph( img, mask, objsm );

  iftOISFSegm(igraph, seeds, alpha, beta, gamma, iter, pf_op, sr_op);

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
    {.short_name = "-pf", .long_name = "--path-cost", .has_arg = true, .arg_type = IFT_LONG_TYPE,
      .required = true, .help = PATHFUNC_DESC},
    {.short_name = "-sr", .long_name = "--seed-recomp", .has_arg = true, .arg_type = IFT_LONG_TYPE,
      .required = true, .help = SEEDRECOMP_DESC},
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
(  iftDict *args, char **img_path, char **seed_path,  char **out_path, int *pf_op, int *sr_op)
{
  *img_path = iftGetStrValFromDict("--input-img", args);
  if (!iftFileExists(*img_path)) 
    iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *img_path);

  *seed_path = iftGetStrValFromDict("--seed-img", args);
  if (!iftFileExists(*seed_path)) 
    iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *seed_path);

  *pf_op = iftGetLongValFromDict("--path-cost", args);
  if( *pf_op < 0 || *pf_op > 1 ) 
    iftError("Non-existent path-cost function: \"%d\"", "iftGetAndValReqArgs", *pf_op);

  *sr_op = iftGetLongValFromDict("--seed-recomp", args);
  if( *sr_op < 0 || *sr_op > 0 ) 
    iftError("Non-existent seed recomputation function: \"%d\"", "iftGetAndValReqArgs", *sr_op);

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
  } else *iter = 10;

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

  igraph = iftImplicitIGraph(obj_mimg, mask, A);

  //Free
  iftDestroyMImage(&mimg);
  iftDestroyMImage(&obj_mimg);
  iftDestroyAdjRel(&A);

  return igraph;
}

iftMImage *iftExtendMImageByObjSalMap
(iftMImage *mimg, iftImage* objsm )
{
  int p,b, min_sm_val, max_sm_val;
  float max_lab_val;
  iftMImage *emimg;

  emimg = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m+1);
  max_lab_val = iftMMaximumValue(mimg, -1);
  iftMinMaxValues(objsm, &min_sm_val, &max_sm_val);
  
  for ( p = 0; p < mimg->n; p++)  {
    
    for(b = 0; b < mimg->m; b++ )  emimg->val[p][b] = mimg->val[p][b];

    emimg->val[p][mimg->m] = max_lab_val * ((objsm->val[p] - min_sm_val)/((float)(max_sm_val - min_sm_val)));
  }

  return emimg;
}

void iftOISFSegm
(iftIGraph *igraph, iftImage *seeds, double alpha, double beta, double gamma, int iters, int pf_op, int sr_op )
{
 double tmp;
  int r, s, t, i, p, q, it, nseeds;
  int *seed, *center;
  float max_objsm_val;
  double color_dist, geo_dist, obj_dist;
  iftVoxel u, v;
  iftDHeap *Q;
  double *pvalue;
  iftSet *S, *new_seeds, *frontier_nodes, *trees_rm;

  nseeds = 0;
  max_objsm_val = iftIGraphMaximumFeatureValue(igraph, igraph->nfeats-1);
  
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
      frontier_nodes = _iftIGraphTreeRemoval(igraph, &trees_rm, pvalue, IFT_INFINITY_DBL);
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
            tmp = 0.0;
            if( pf_op == 0 ) { // ISMM'17
              color_dist = (double)iftFeatDistance(igraph->feat[r], igraph->feat[q], igraph->nfeats-1);
              geo_dist = (double)iftVoxelDistance(u,v);
              obj_dist = (double)(abs((igraph->feat[r][igraph->nfeats-1] - igraph->feat[q][igraph->nfeats-1]))/max_objsm_val);

              tmp = pow( alpha*color_dist*pow(gamma, obj_dist) +gamma*obj_dist, beta);
              tmp += geo_dist;
              tmp += pvalue[s];

            } else if( pf_op == 1 ) { // MAX-GAMMA
              color_dist = (double)iftFeatDistance(igraph->feat[r], igraph->feat[q], igraph->nfeats-1);
              obj_dist = (double)(abs((igraph->feat[r][igraph->nfeats-1] - igraph->feat[q][igraph->nfeats-1])));

              tmp = iftMax(pvalue[s], (1-gamma)*color_dist + gamma * obj_dist);
            } 
            else iftError("Nao PODE!", "iftOISFSegm");

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
                if (tmp > pvalue[t]) _iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
                else 
                {
                  if ((igraph->label[q] != igraph->label[p])&&(igraph->label[q]!=0))
                  {
                    _iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
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
      center = NULL;
      if( sr_op == 0 ) center = _iftIGraphSuperpixelCenters(igraph, seed, nseeds); // FEAT
      else iftError("Nao PODE!", "iftOISFSegm");

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

int *_iftIGraphSuperpixelCenters(iftIGraph *igraph, int *seed, int nseeds)
{
    int    i, j, p, q, s, *center;
    float  **feat,  *nelems, dist1, dist2;

    /* compute average feature vector for each superpixel */

    feat   = (float **)iftAlloc(nseeds,sizeof(float *));
    nelems = iftAllocFloatArray(nseeds);
    center = iftAllocIntArray(nseeds);
    for (i=0; i < nseeds; i++){
        feat[i]   = iftAllocFloatArray(igraph->nfeats);
        center[i] = seed[i];
    }

    for (s=0; s < igraph->nnodes; s++) {
        p = igraph->node[s].voxel;
        i = igraph->label[igraph->root[p]]-1;
        nelems[i]++;
        for (j=0; j < igraph->nfeats; j++)
            feat[i][j] += igraph->feat[p][j];
    }

    for (i=0; i < nseeds; i++) {
        for (j=0; j < igraph->nfeats; j++)
    feat[i][j] /= nelems[i];
    }

    /* compute the closest node to each superpixel center */

    for (s=0; s < igraph->nnodes; s++) {
        p     = igraph->node[s].voxel;
        i     = igraph->label[igraph->root[p]]-1;
        q     = igraph->node[center[i]].voxel;
        dist1 = iftFeatDistance(feat[i],igraph->feat[q],igraph->nfeats);
        dist2 = iftFeatDistance(feat[i],igraph->feat[p],igraph->nfeats);
        if (dist2 < dist1)
            center[i]=s;
    }

    for (i=0; i < nseeds; i++) {
        iftFree(feat[i]);
    }
    iftFree(feat);
    iftFree(nelems);

    return(center);
}

void _iftIGraphSubTreeRemoval(iftIGraph *igraph, int s, double *pvalue, double INITIAL_PATH_VALUE, iftDHeap *Q)
{
    int        i, p, q, t;
    iftVoxel   u, v;
    iftAdjRel *A = igraph->A;
    iftSet    *Frontier = NULL, *adj = NULL, *Subtree = NULL;
    iftImage  *index = igraph->index;

    switch(igraph->type) {

        case IMPLICIT:

            /* Reinitialize voxels (nodes) of the subtree of s to be
               reconquered and compute the frontier nodes (voxels) */

            iftInsertSet(&Subtree,s);

            while (Subtree != NULL){
                s = iftRemoveSet(&Subtree);
                p = igraph->node[s].voxel;
                u = iftGetVoxelCoord(index,p);

                if (Q->color[s]==IFT_GRAY)
                    iftRemoveDHeapElem(Q,s);


                igraph->pvalue[p] = pvalue[s] = INITIAL_PATH_VALUE;
                igraph->pred[p]   = IFT_NIL;

                for (i = 1; i < A->n; i++){
                    v = iftGetAdjacentVoxel(A, u, i);
                    if (iftValidVoxel(index, v)){
                        q   = iftGetVoxelIndex(index, v);
                        t   = index->val[q];
                        if (igraph->pred[q]==p)
                            iftInsertSet(&Subtree,t);
                        else{ /* consider t as a candidate to be a frontier node */
                            iftInsertSet(&Frontier,t);
                        }
                    }
                }
            }

            break;

        case EXPLICIT:


            /* Reinitialize voxels (nodes) of the subtree of s to be
           reconquered and compute the frontier nodes (voxels) */

            iftInsertSet(&Subtree,s);

            while (Subtree != NULL){
                s = iftRemoveSet(&Subtree);
                p = igraph->node[s].voxel;

                if (Q->color[s]==IFT_GRAY)
                    iftRemoveDHeapElem(Q,s);

                igraph->pvalue[p] = pvalue[s] = INITIAL_PATH_VALUE;
                igraph->pred[p]   = IFT_NIL;
                Q->color[s]       = IFT_WHITE;

                for (adj=igraph->node[s].adj; adj != NULL; adj=adj->next) {
                    t = adj->elem;
                    q = igraph->node[t].voxel;

                    if (igraph->pred[q]==p)
                        iftInsertSet(&Subtree,t);
                    else{ /* consider t as a candidate to be a frontier node */
                        iftInsertSet(&Frontier,t);
                    }
                }
            }

            break;

        case COMPLETE:

            /* Reinitialize voxels (nodes) of the subtree of s to be
           reconquered and compute the frontier nodes (voxels) */

            iftInsertSet(&Subtree,s);

            while (Subtree != NULL){
                s = iftRemoveSet(&Subtree);
                p = igraph->node[s].voxel;
                u = iftGetVoxelCoord(index,p);

                if (Q->color[s]==IFT_GRAY)
                    iftRemoveDHeapElem(Q,s);

                igraph->pvalue[p] = pvalue[s] = INITIAL_PATH_VALUE;
                igraph->pred[p]   = IFT_NIL;
                Q->color[s]       = IFT_WHITE;

                for (t=0; t < igraph->nnodes; t++) {
                    if (t != s ) {
                        q = igraph->node[t].voxel;

                        if (igraph->pred[q]==p)
                            iftInsertSet(&Subtree,t);
                        else{ /* consider t as a candidate to be a frontier node */
                            iftInsertSet(&Frontier,t);
                        }
                    }
                }
            }

            break;

        default:
            iftError("Invalid type of image graph", "_iftIGraphSubTreeRemoval");
    }

    /* Identify the real frontier nodes and insert them in Queue to
       continue the DIFT */

    while (Frontier != NULL){
        s = iftRemoveSet(&Frontier);
        p = igraph->node[s].voxel;
        if (igraph->label[p] != 0){
            if (Q->color[s] == IFT_GRAY)
                iftGoUpDHeap(Q, Q->pos[s]);
            else
                iftInsertDHeap(Q,s);
        }
    }
}

iftSet *_iftIGraphTreeRemoval(iftIGraph *igraph, iftSet **trees_for_removal, double *pvalue, double INITIAL_PATH_VALUE)
{
    int        i, p, q, r, s, t;
    iftVoxel   u, v;
    iftAdjRel *A = igraph->A;
    iftSet    *Frontier = NULL, *adj = NULL;
    iftBMap   *inFrontier = iftCreateBMap(igraph->nnodes);
    iftImage  *index = igraph->index;
    iftSet    *T1 = NULL, *T2 = NULL;

    switch(igraph->type) {

        case IMPLICIT:

            /* Remove all marked trees and find the frontier voxels
           afterwards. */

            while (*trees_for_removal != NULL){
                s = iftRemoveSet(trees_for_removal);
                p = igraph->node[s].voxel;
                r = igraph->root[p];

                if (pvalue[index->val[r]] != INITIAL_PATH_VALUE){ /* tree not marked yet */
                    igraph->pvalue[r] = pvalue[index->val[r]] = INITIAL_PATH_VALUE; /* mark removed root */
                    igraph->pred[r]   = IFT_NIL;
                    iftInsertSet(&T1, r);
                    while (T1 != NULL){
                        p = iftRemoveSet(&T1);
                        iftInsertSet(&T2, p); /* compute in T2 the union of removed trees */
                        u = iftGetVoxelCoord(index, p);
                        for (i = 1; i < A->n; i++){
                            v = iftGetAdjacentVoxel(A, u, i);
                            if (iftValidVoxel(index, v)){
                                q   = iftGetVoxelIndex(index, v);
                                t   = index->val[q];
                                if ((t != IFT_NIL) && (pvalue[t] != INITIAL_PATH_VALUE)){ /* q has not been removed */
                                    if (igraph->pred[q] == p){ /* q belongs to the tree under removal */
                                        iftInsertSet(&T1, q);
                                        pvalue[t]         = igraph->pvalue[q] = INITIAL_PATH_VALUE; /* mark removed node */
                                        igraph->pred[q]   = IFT_NIL;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Find the frontier voxels of non-removed trees */

            while (T2 != NULL){
                p = iftRemoveSet(&T2);
                u = iftGetVoxelCoord(index, p);
                for (i = 1; i < A->n; i++){
                    v = iftGetAdjacentVoxel(A, u, i);
                    if (iftValidVoxel(index, v)){
                        q   = iftGetVoxelIndex(index, v);
                        t   = index->val[q];
                        if ((t != IFT_NIL) && (pvalue[t] != INITIAL_PATH_VALUE)){ /* q is a frontier node */
                            if (iftBMapValue(inFrontier, t) == 0){ /* t has not been inserted in the frontier yet */
                                iftInsertSet(&Frontier, t);
                                iftBMapSet1(inFrontier, t);
                            }
                        }
                    }
                }
            }

            break;

        case EXPLICIT:

            /* Remove all marked trees and find the frontier voxels
           afterwards. */

            while (*trees_for_removal != NULL){
                s = iftRemoveSet(trees_for_removal);
                p = igraph->node[s].voxel;
                r = igraph->root[p];

                if (pvalue[index->val[r]] != INITIAL_PATH_VALUE){ /* tree not marked yet */
                    pvalue[index->val[r]] = igraph->pvalue[r] = INITIAL_PATH_VALUE; /* mark removed root */
                    igraph->pred[r]     = IFT_NIL;
                    iftInsertSet(&T1, r);
                    while (T1 != NULL){
                        p = iftRemoveSet(&T1);
                        s = index->val[p];
                        iftInsertSet(&T2, p); /* compute in T2 the union of removed trees */
                        for (adj=igraph->node[s].adj; adj != NULL; adj=adj->next) {
                            t   = adj->elem;
                            q   = igraph->node[t].voxel;
                            if (pvalue[t] != INITIAL_PATH_VALUE){ /* q has not been removed */
                                if (igraph->pred[q] == p){ /* q belongs to the tree under removal */
                                    iftInsertSet(&T1, q);
                                    pvalue[t]       = igraph->pvalue[q] = INITIAL_PATH_VALUE; /* mark removed node */
                                    igraph->pred[q] = IFT_NIL;
                                }
                            }
                        }
                    }
                }
            }


            /* Find the frontier voxels of non-removed trees */

            while (T2 != NULL){
                p = iftRemoveSet(&T2);
                s = igraph->index->val[p];
                for (adj=igraph->node[s].adj; adj != NULL; adj=adj->next) {
                    t   = adj->elem;
                    if (pvalue[t] != INITIAL_PATH_VALUE){ /* t is a frontier node */
                        if (iftBMapValue(inFrontier, t) == 0){  /* t has not been inserted in the frontier yet */
                            iftInsertSet(&Frontier, t);
                            iftBMapSet1(inFrontier, t);
                        }
                    }
                }
            }

            break;

        case COMPLETE:

            /* Remove all marked trees and find the frontier voxels
           afterwards. */

            while (*trees_for_removal != NULL){
                s = iftRemoveSet(trees_for_removal);
                p = igraph->node[s].voxel;
                r = igraph->root[p];

                if (pvalue[index->val[r]] != INITIAL_PATH_VALUE){ /* tree not marked yet */
                    igraph->pvalue[r] = pvalue[index->val[r]] = INITIAL_PATH_VALUE; /* mark removed root */
                    igraph->pred[r]   = IFT_NIL;
                    iftInsertSet(&T1, r);
                    while (T1 != NULL){
                        p = iftRemoveSet(&T1);
                        s = index->val[p];
                        iftInsertSet(&T2, p); /* compute in T2 the union of removed trees */

                        for (t=0; t < igraph->nnodes; t++) {
                            if (t != s ) {
                                q = igraph->node[t].voxel;
                                if (pvalue[t] != INITIAL_PATH_VALUE){ /* q has not been removed */
                                    if (igraph->pred[q] == p){ /* q belongs to the tree under removal */
                                        iftInsertSet(&T1, q);
                                        pvalue[t] = igraph->pvalue[q] = INITIAL_PATH_VALUE; /* mark removed node */
                                        igraph->pred[q]   = IFT_NIL;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Find the frontier voxels of non-removed trees */

            while (T2 != NULL){
                p = iftRemoveSet(&T2);
                s = index->val[p];
                for (t=0; t < igraph->nnodes; t++) {
                    if (t != s ) {
                        if (pvalue[t] != INITIAL_PATH_VALUE){ /* t is a frontier node */
                            if (iftBMapValue(inFrontier, t) == 0){ /* t has not been inserted in the frontier yet */
                                iftInsertSet(&Frontier, t);
                                iftBMapSet1(inFrontier, t);
                            }
                        }
                    }
                }
            }

        default:
            iftError("Invalid type of image graph", "_iftIGraphTreeRemoval");
    }

    iftDestroyBMap(&inFrontier);

    return (Frontier);
}