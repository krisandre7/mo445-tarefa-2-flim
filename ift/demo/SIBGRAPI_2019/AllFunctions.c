#include <ift.h>


iftImage *overlayBorders (iftImage *orig, iftImage *label_img, float radius, float r_perc, float g_perc, float b_perc)
{
  int normvalue;
  iftColor RGB, YCbCr;
  iftAdjRel *A;
  iftImage *copy_img, *border_img;

  normvalue = iftNormalizationValue(iftMaximumValue(orig));
  
  if(iftIs3DImage(orig)) A = iftSpheric(radius);
  else A = iftCircular(radius);

  RGB = iftRGBColor((int)normvalue * r_perc, (int)normvalue * g_perc, (int)normvalue * b_perc);
  YCbCr = iftRGBtoYCbCr(RGB, normvalue);

  copy_img = iftCopyImage(orig);
  border_img = iftBorderImage(label_img,1);

  iftDrawBorders(copy_img, border_img, A, YCbCr, A);

  // Free
  iftDestroyImage(&border_img);
  iftDestroyAdjRel(&A);

  return copy_img;
}

char *computeScores (char* image_id, iftImage *segm_img, iftImage *seed_img, iftImage *gt_img)
{
  int num_seeds, seed_min, gt_min;
  float br, ue, obj_perc;
  char *csv_str;
  iftImage *segm_border_img, *gt_border_img;

  csv_str = (char *)calloc(256, sizeof(char));

  gt_min = iftMinimumValue(gt_img);
  
  segm_border_img = iftBorderImage(segm_img,1);
  gt_border_img = iftBorderImage(gt_img,1);

  num_seeds = 0;
  obj_perc = 0.0;

  if(seed_img != NULL)
  {
    seed_min = iftMinimumValue(seed_img);
  
    for(int p = 0; p < seed_img->n; p++)
    {
      if(seed_img->val[p] > seed_min)
      {
        num_seeds++;
  
        if(gt_img->val[p] > gt_min) obj_perc++;
      }
    }
  
    obj_perc /= (float)num_seeds;
  }
  else
  { 
    num_seeds = iftMaximumValue(segm_img) - iftMinimumValue(segm_img) + 1;
    obj_perc = -1.0;
  }

  br = iftBoundaryRecall(gt_border_img, segm_border_img, 2.0);
  ue = iftUnderSegmentation(gt_img, segm_img);

  sprintf(csv_str, "%s,%f,%f,%d,%f", image_id, br,ue,num_seeds,obj_perc);

  return csv_str;
}

iftSet *_ObjSalMapSamplByValueWithAreaSum (iftImage *objsm, iftImage *mask, int num_seeds)
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

iftImage *OSMOXSampling (iftImage* objsm, iftImage *mask, int num_seeds, float obj_perc)
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
  
  obj_set = _ObjSalMapSamplByValueWithAreaSum(objsm, mask_copy, obj_seeds);

  invsm = iftComplement(objsm);

  s = obj_set;
  while( s != NULL ) {
    mask_copy->val[s->elem] = 0;
    seed_img->val[s->elem] = 1;
    s = s->next;
  }

  bkg_set = _ObjSalMapSamplByValueWithAreaSum(invsm, mask_copy, bkg_seeds);

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

void _iftIGraphISFRecomputeSeeds(iftIGraph *igraph, int *seed, int nseeds, iftSet **trees_for_removal, iftSet **new_seeds, float *new_seeds_flag)
{
    int      i, *center, p, q, s;
    iftVoxel u, v;
    float    distColor, distVoxel, distColorThres, distVoxelThres;

    /* Compute superpixel centers (i.e., the closest node to the center in the feature space) */

    center = _iftIGraphSuperpixelCenters(igraph, seed, nseeds);


    /* Estimate distColorThres */

    distColorThres = 0.0;  distVoxelThres = 0.0;
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
    //distColorThres = 0.70*distColorThres; // converges faster
    //distVoxelThres = 0.70*distVoxelThres;

    /* Verify if the centers can be new seeds */

    *new_seeds_flag = 0;
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
            iftInsertSet(trees_for_removal,seed[i]);
            *new_seeds_flag+= 1;
        }
    }
    *new_seeds_flag /= (float)nseeds;

    iftFree(center);

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
            iftError("Invalid type of image graph", "iftIGraphTreeRemoval");
    }

    iftDestroyBMap(&inFrontier);

    return (Frontier);
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
            iftError("Invalid type of image graph", "iftIGraphSubtreeRemoval");
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

int _iftIGraphISF_Root(iftIGraph *igraph, iftImage *seeds, double alpha, double beta, int niters)
{
    double      tmp;
    int        r, s, t, i, p, q, it, *seed, nseeds;
    float      new_seeds_flag;
    iftVoxel   u, v;
    iftDHeap  *Q;
    double    *pvalue = iftAllocDoubleArray(igraph->nnodes);
    iftSet    *adj=NULL, *S=NULL, *new_seeds=NULL, *frontier_nodes=NULL, *trees_for_removal=NULL;


    new_seeds_flag = 1.0;

    /* Initial seed set and trivial path initialization to infinity
       cost */

    Q      = iftCreateDHeap(igraph->nnodes, pvalue);

    nseeds = 0;
    for (s=0; s < igraph->nnodes; s++) {
        p               = igraph->node[s].voxel;
        pvalue[s]       = igraph->pvalue[p] = IFT_INFINITY_DBL;
        igraph->pred[p] = IFT_NIL;
        if (seeds->val[p]!=0){
            iftInsertSet(&new_seeds,s);
            nseeds++;
        }
    }

    seed    = iftAllocIntArray(nseeds);
    S       = new_seeds; i = 0;
    while (S != NULL) {
        seed[i] = S->elem;
        p       = igraph->node[seed[i]].voxel;
        igraph->label[p] = i+1;
        i++; S = S->next;
    }


    /* differential optimum-path forest computation */

    //  for (I=0; (I < niters)&&(new_seeds_flag >= 0.1); I++) {  // If 90% of the seed do not change much (satisfy both the criteria in LAB and XY) we stop
    for (it=0; (it < niters); it++) {

        if (trees_for_removal != NULL)
            frontier_nodes = _iftIGraphTreeRemoval(igraph, &trees_for_removal, pvalue, IFT_INFINITY_DBL);

        while (new_seeds != NULL) { /* insert seeds in the priority queue Q with cost zero */
            s = iftRemoveSet(&new_seeds);
            p = igraph->node[s].voxel;
      if (igraph->label[p] > 0){ /* we must avoid removed seeds
            from previous iteration */
        pvalue[s] = igraph->pvalue[p] = 0;
        igraph->root[p] = p;
        igraph->pred[p] = IFT_NIL;
        iftInsertDHeap(Q,s);
      }
        }

        while (frontier_nodes != NULL) { /* insert frontier nodes in Q to represent the previous seeds */
            s = iftRemoveSet(&frontier_nodes);
            if (Q->color[s] == IFT_WHITE){
                iftInsertDHeap(Q,s);
            }
        }


        switch(igraph->type) {

            case IMPLICIT:

                while (!iftEmptyDHeap(Q)) {
                    s = iftRemoveDHeap(Q);
                    p = igraph->node[s].voxel;
                    r = igraph->root[p];
                    igraph->pvalue[p] = pvalue[s];
                    u = iftGetVoxelCoord(igraph->index,p);
                    for (i=1; i < igraph->A->n; i++) {
                        v = iftGetAdjacentVoxel(igraph->A,u,i);
                        if (iftValidVoxel(igraph->index,v)){
                            q   = iftGetVoxelIndex(igraph->index,v);
                            t   = igraph->index->val[q];
                            if ((t != IFT_NIL) && (Q->color[t] != IFT_BLACK)){
                                tmp = pvalue[s] + pow(alpha*(double)iftFeatDistance(igraph->feat[r],igraph->feat[q],igraph->nfeats),beta) + (double)iftVoxelDistance(u,v);

                                if (tmp < pvalue[t]){
                                    pvalue[t]            = tmp;

                                    /* /\* This is used for the old differential version *\/ */
                                    /* if (igraph->label[p] != igraph->label[q]){ /\* voxels */
                                    /*            that */
                                    /*            have */
                                    /*            changed */
                                    /*            labels *\/ */
                                    /*   iftInsertSet(&T, q); */
                                    /* } */

                                    igraph->root[q]      = igraph->root[p];
                                    igraph->label[q]     = igraph->label[p];
                                    igraph->pred[q]      = p;

                                    if (Q->color[t] == IFT_GRAY){
                                        iftGoUpDHeap(Q, Q->pos[t]);
                                    } else {
                                        iftInsertDHeap(Q,t);
                                    }
                                } else {
                                    if (igraph->pred[q] == p){
                                        if (tmp > pvalue[t]) {
                                            _iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
                                        } else { /* tmp == pvalue[t] */
                                            if ((igraph->label[q] != igraph->label[p])&&(igraph->label[q]!=0)){
                                                _iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
                                            }
                                        }
                                    }
                                }

                            }
                        }
                    }
                }



                break;

            case EXPLICIT:

                while (!iftEmptyDHeap(Q)) {
                    s = iftRemoveDHeap(Q);
                    p = igraph->node[s].voxel;
                    r = igraph->root[p];
                    u = iftGetVoxelCoord(igraph->index,p);
                    igraph->pvalue[p] = pvalue[s];

                    for (adj=igraph->node[s].adj; adj != NULL; adj=adj->next) {
                        t   = adj->elem;
                        if (Q->color[t] != IFT_BLACK){
                            q   = igraph->node[t].voxel;
                            v   = iftGetVoxelCoord(igraph->index,q);
                            tmp = pvalue[s] + pow(alpha*(double)iftFeatDistance(igraph->feat[r],igraph->feat[q],igraph->nfeats),beta) + (double)iftVoxelDistance(u,v);
                            if (tmp < pvalue[t]) {

                                if (tmp < pvalue[t]){
                                    pvalue[t]            = tmp;

                                    /* /\* This is used for the old differential version *\/ */
                                    /* if (igraph->label[p] != igraph->label[q]){ /\* voxels */
                                    /*            that */
                                    /*            have */
                                    /*            changed */
                                    /*            labels *\/ */
                                    /*   iftInsertSet(&T, q); */
                                    /* } */

                                    igraph->root[q]      = igraph->root[p];
                                    igraph->label[q]     = igraph->label[p];
                                    igraph->pred[q]      = p;

                                    if (Q->color[t] == IFT_GRAY){
                                        iftGoUpDHeap(Q, Q->pos[t]);
                                    } else {
                                        iftInsertDHeap(Q,t);
                                    }
                                } else {
                                    if (igraph->pred[q] == p){
                                        if (tmp > pvalue[t]) {
                                            _iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
                                        } else { /* tmp == pvalue[t] */
                                            if ((igraph->label[q] != igraph->label[p])&&(igraph->label[q]!=0)){
                                                _iftIGraphSubTreeRemoval(igraph,t,pvalue,IFT_INFINITY_DBL,Q);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                break;

            case COMPLETE:
                iftError("Not implemented for complete graphs", "iftIGraphISF_Root");
        }


        iftResetDHeap(Q);
        iftIGraphISFRecomputeSeeds(igraph, seed, nseeds, &trees_for_removal, &new_seeds, &new_seeds_flag);
    }


    iftDestroySet(&adj);
    iftDestroySet(&S);
    iftDestroySet(&new_seeds);
    iftDestroySet(&frontier_nodes);
    iftDestroySet(&trees_for_removal);
    iftDestroyDHeap(&Q);
    iftFree(pvalue);
    iftFree(seed);

    return it;
}