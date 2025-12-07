/* BEGIN CUT REGION */

void iftSmoothFrontierGlobal(iftImageForest *fst, int num_smooth_iterations)
{
    iftImage  *prev_label, *next_label;
    iftFImage *prev_weight, *next_weight;
    float     *sum, max_membership;
    int        l, i, p, q, r, max_label, iter, tmp;
    iftSet    *prev_frontier = NULL, *next_frontier = NULL, *Subtree = NULL, *Processed = NULL;
    iftVoxel   u, v;
    iftAdjRel *A = fst->A;
    iftBMap *tmpFrontier = NULL;

    /* Initialization */
    prev_label  = iftCopyImage(fst->label);
    next_label  = iftCopyImage(prev_label);
    iftMaximumValue(prev_label);
    sum         = iftAllocFloatArray(prev_label->maxval + 1);
    prev_weight = iftCreateFImage(prev_label->xsize, prev_label->ysize, prev_label->zsize);
    next_weight = iftCreateFImage(next_label->xsize, next_label->ysize, next_label->zsize);

    // This is required to create an auxiliary bitmap and consider the previous iterations of the DIFT algorithm
    tmpFrontier = iftCopyBMap(fst->smooth->inFrontier);
    prev_frontier = NULL;
    for (p = 0; p < fst->smooth->inFrontier->n; p++)
        if (iftBMapValue(fst->smooth->inFrontier, p))
            iftInsertSet(&prev_frontier, p);

    for (p = 0; p < prev_label->n; p++)
        prev_weight->val[p] = next_weight->val[p] = 1.0;

    /* Smooth frontier and reset its path values */
    for (iter = 0; iter < num_smooth_iterations; iter++)
    {
        while (prev_frontier != NULL)
        {
            p = iftRemoveSet(&prev_frontier);
            iftInsertSet(&next_frontier, p);
            u   = iftGetVoxelCoord(prev_label, p);

            for (l = 0; l <= prev_label->maxval; l++)
                sum[l] = 0.0;
            for (i = 1; i < A->n; i++)
            {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(prev_label, v))
                {
                    q = iftGetVoxelIndex(prev_label, v);
                    sum[prev_label->val[q]] += prev_weight->val[q] * fst->smooth->border_weight->val[q];
                    if (iftBMapValue(tmpFrontier, q) == 0) /* expand frontier */
                    {
                        if (fst->pred->val[q] != NIL)
                        {
                            iftInsertSet(&next_frontier, q);
                            iftBMapSet1(tmpFrontier, q);
                        }
                    }
                }
            }

            for (l = 0; l <= prev_label->maxval; l++)
            {
                sum[l]  = sum[l] / fst->smooth->norm_factor->val[p];
            }


            max_membership = -INFINITY_FLT; max_label = NIL;
            for (l = 0; l <= prev_label->maxval; l++)
            {
                if (sum[l] > max_membership)
                {
                    max_membership = sum[l];
                    max_label      = l;
                }
            }
            next_label->val[p]  = max_label;
            next_weight->val[p] = sum[max_label];
        }

        prev_frontier = next_frontier;
        next_frontier = NULL;

        for (r = 0; r < prev_label->n; r++)
        {
            prev_weight->val[r] = next_weight->val[r];
            prev_label->val[r]  = next_label->val[r];
        }
    }
    free(sum);
    iftDestroyBMap(&tmpFrontier);
    iftDestroyFImage(&prev_weight);
    iftDestroyFImage(&next_weight);
    iftDestroyImage(&next_label);
    iftDestroySet(&prev_frontier);


    /* Fix the forest by first making available to be conquered all
       voxels whose label has changed and their subtrees  */
    for (p = 0; p < fst->label->n; p++)
    {
        //If it is not a seed and the label has changed
        if (fst->label->val[p] != prev_label->val[p]) //&&(fst->pred->val[p]!=NIL)
        {
            iftInsertSet(&Subtree, p);
            fst->pathval->val[p] = INFINITY_INT; //Allow the pixel to be conquered

            //All subtree needs to become available to be conquered
            while (Subtree != NULL)
            {
                r = iftRemoveSet(&Subtree);
                u = iftGetVoxelCoord(fst->pred, r);

                for (i = 1; i < A->n; i++)
                {
                    v = iftGetAdjacentVoxel(A, u, i);
                    if (iftValidVoxel(fst->pred, v))
                    {
                        q = iftGetVoxelIndex(fst->pred, v);
                        //If its neighbour predecessor pointer is pointing to r, so the neighbour also needs to be conquered.
                        if (fst->pred->val[q] == r)
                        {
                            fst->pathval->val[q] = INFINITY_INT;
                            iftInsertSet(&Subtree, q);
                        }
                    }
                }
            }
        }
    }

    /* Insert in priority queue the seed voxels, which will be the
       neighbors with the same label of the region to be conquered. */
    for (p = 0; p < fst->label->n; p++)
    {
        if (fst->pathval->val[p] == INFINITY_INT)
        {
            u = iftGetVoxelCoord(fst->pred, p);
            fst->pred->val[p] = NIL;

            for (i = 1; i < A->n; i++)
            {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(fst->pred, v))
                {
                    q = iftGetVoxelIndex(fst->pred, v);
                    if ((prev_label->val[q] == fst->label->val[q] && fst->label->val[q] == prev_label->val[p]) && (fst->pathval->val[q] != INFINITY_INT))
                    {
                        if (fst->Q->L.elem[q].color == WHITE)
                        {
                            iftInsertGQueue(&fst->Q, q);
                        }
                    }
                }
            }
        }
    }

    iftDestroyImage(&fst->label);
    fst->label = prev_label;

    /* execute the IFT to reconstruct the forest under the new labeling
       constraint. This forest is not optimum, since this is a relaxed
       IFT, but it maintains the connectivity between roots and voxels
       of the same label, respecting the filtering process. */

    while (!iftEmptyGQueue(fst->Q))
    {
        p = iftRemoveGQueue(fst->Q);
        iftInsertSet(&Processed, p);

        u = iftGetVoxelCoord(fst->img, p);

        for (i = 1; i < A->n; i++)
        {
            v  = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(fst->img, v))
            {
                q = iftGetVoxelIndex(fst->img, v);
                if (fst->Q->L.elem[q].color != BLACK)
                {
                    tmp = MAX(fst->pathval->val[p], fst->img->val[q]);
                    if (tmp < fst->pathval->val[q] || ((fst->pred->val[q] == p)))
                    {
                        if (fst->Q->L.elem[q].color == GRAY)
                            iftRemoveGQueueElem(fst->Q, q);

                        fst->root->val[q]    = fst->root->val[p];
                        fst->pred->val[q]    = p;
                        fst->label->val[q]   = fst->label->val[p];
                        fst->pathval->val[q] = tmp;
                        iftInsertGQueue(&fst->Q, q);
                    }
                }
                if (fst->label->val[p] != fst->label->val[q])
                {
                    iftBMapSet1(fst->smooth->inFrontier, p);
                    iftBMapSet1(fst->smooth->inFrontier, q);
                }
                else
                {
                    iftBMapSet0(fst->smooth->inFrontier, p);
                    iftBMapSet0(fst->smooth->inFrontier, q);
                }
            }
        }
    }
    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        fst->Q->L.elem[p].color = WHITE;
    }
    /* Verify forest consistency: This can be removed when we decide for
       the final procedure. */
    /*
      for (p=0; p < fst->label->n; p++) {
        r = p;
        while (fst->pred->val[r]!=NIL)
          {
        if (fst->label->val[r]!=fst->label->val[fst->pred->val[r]])
          {
            iftWarning("Incorrect reconstruction of the label map","iftsmoothFrontier");
            fprintf(stderr, "%d, %d:", iftGetXCoord(fst->label, r), iftGetYCoord(fst->label, r));
            fprintf(stderr, "Label: %d, Pred Label: %d\n", fst->label->val[r], fst->label->val[fst->pred->val[r]]);
            fprintf(stderr, "Cost: %d, Pred cost: %d\n", fst->pathval->val[r], fst->pathval->val[fst->pred->val[r]]);
            fprintf(stderr, "Pred of predecessor: %d\n", fst->pred->val[fst->pred->val[r]]);
          }
             r = fst->pred->val[r];
          }
      }
    */
}
void iftDiffGlobalRelaxedWatershed(iftImageForest *fst, iftLabeledSet *seed, int num_smooth_iterations, float smooth_factor)
{
    iftAdjRel *A = fst->A;
    iftGQueue *Q = fst->Q;
    iftVoxel   u, v;
    int        i, p, q, tmp, frontierPresent = 0;
    char       trees_for_removal;
    iftSet    *Frontier = NULL, *Processed = NULL;
    iftLabeledSet *S;
    iftImage  *pathval = fst->pathval, *pred = fst->pred, *label = fst->label;
    iftImage  *root = fst->root, *basins = fst->img;

    // Verify if there are trees for removal
    trees_for_removal = 0;
    S = seed;
    while (S != NULL)
    {
        if ((S->label == NIL) &&
                (label->val[S->elem] > 0)) // Removal marker
        {
            trees_for_removal = 1;
            break;
        }
        S = S->next;
    }

    // Remove marked trees
    if (trees_for_removal)
    {
        Frontier = iftCompRemoval(fst, seed);
        while (Frontier != NULL)
        {
            p = iftRemoveSet(&Frontier);
            iftInsertGQueue(&Q, p);
        }
    }

    // Trivial path initialization for new seeds
    S = seed;
    while (S != NULL)
    {
        p = S->elem;
        if (S->label != NIL)
        {
            if (Q->L.elem[p].color == GRAY)
            {
                /* p is also a frontier voxel, but the priority is it as a seed. */
                iftRemoveGQueueElem(Q, p);
            }
            label->val[p] = S->label;
            pathval->val[p] = 0;
            root->val[p] = p;
            pred->val[p] = NIL;
            iftInsertGQueue(&Q, p);
        }
        S = S->next;
    }

    // The smooth structure is only created at the first iteration
    if (fst->smooth == NULL)
        fst->smooth = iftCreateSmoothBorder(fst->img, fst->A, smooth_factor);

    /* Image Foresting Transform */
    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        iftInsertSet(&Processed, p);
        u = iftGetVoxelCoord(basins, p);
        
        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(basins, v))
            {
                q = iftGetVoxelIndex(basins, v);
                if (Q->L.elem[q].color != BLACK)
                {
                    tmp = MAX(pathval->val[p], basins->val[q]);
                    if ((tmp < pathval->val[q]) || ((pred->val[q] == p)))
                    {
                        if (Q->L.elem[q].color == GRAY)
                        {
                            iftRemoveGQueueElem(Q, q);
                        }
                        pred->val[q]  = p;
                        root->val[q]  = root->val[p];
                        label->val[q] = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
                if (label->val[p] != label->val[q])
                {
                    /* p and q must be in the Frontier set */
                    iftBMapSet1(fst->smooth->inFrontier, p);
                    iftBMapSet1(fst->smooth->inFrontier, q);
                    frontierPresent = 1;
                }
                else
                {
                    /* they were frontier once but the propagation changed that */
                    iftBMapSet0(fst->smooth->inFrontier, p);
                    iftBMapSet0(fst->smooth->inFrontier, q);
                }
            }
        }
    }

    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        Q->L.elem[p].color = WHITE;
    }

    if ((frontierPresent == 1) && (num_smooth_iterations >= 1))
    {
        iftSmoothFrontierGlobal(fst, num_smooth_iterations);
    }
    iftDestroySet(&Frontier);

}
/*
Differential Relaxed Fuzzy Connectedness.      
      float     smooth_factor=0.5;
      iftAdjRel *A           = iftSpheric(1.5);

      iftFImage *prev_weight = iftCreateFImage(img->xsize,img->ysize,img->zsize);
      iftFImage *next_weight = iftCreateFImage(img->xsize,img->ysize,img->zsize);
      iftFImage *border_weight=iftSmoothWeightImage(basins,smooth_factor);
      iftFImage *norm_factor=iftWeightNormFactor(border_weight,A);

      fst    = iftCreateImageForest(basins, A);
      iftResetImageForest(fst, FIFOBREAK, MAXVALUE);
      iftMaximumValue(fst->img);

      iftDiffRelaxedFuzzyConnectedness(fst, seed_set, border_weight, norm_factor, prev_weight, next_weight, nbr_relax_iterations);

*/

void iftDiffRelaxedFuzzyConnectedness(iftImageForest *fst, iftLabeledSet *seed, iftFImage *border_weight, iftFImage *norm_factor, iftFImage *prev_weight, iftFImage *next_weight, int num_smooth_iterations)
{
    iftAdjRel *A = fst->A;
    iftGQueue *Q = fst->Q;
    iftVoxel   u, v;
    int        i, p, q, tmp;
    float      avg = 0;
    char       trees_for_removal;
    iftSet    *Frontier = NULL, *Processed = NULL;
    iftSet    *dilated_boundary = NULL;
    iftBMap   *inFrontier = iftCreateBMap(fst->img->n);
    iftLabeledSet *S;
    iftImage  *pathval = fst->pathval, *pred = fst->pred, *label = fst->label;
    iftImage  *filtered_labels = NULL;
    iftImage  *root = fst->root, *basins = fst->img;

    // Verify if there are trees for removal
    trees_for_removal = 0;
    S = seed;
    while (S != NULL)
    {
        if ((S->label == NIL) &&
                (label->val[S->elem] > 0)) // Removal marker
        {
            trees_for_removal = 1;
            break;
        }
        S = S->next;
    }
    // Remove marked trees
    if (trees_for_removal)
    {
        Frontier = iftCompRemoval(fst, seed);
        while (Frontier != NULL)
        {
            p = iftRemoveSet(&Frontier);
            iftInsertGQueue(&Q, p);
        }
    }

    // Trivial path initialization for new seeds
    S = seed;
    while (S != NULL)
    {
        p = S->elem;
        if (S->label != NIL)
        {
            if (Q->L.elem[p].color == GRAY)
            {
                /* p is also a frontier voxel,
                   but the priority being a seed. */
                iftRemoveGQueueElem(Q, p);
            }
            label->val[p] = S->label;
            pathval->val[p] = fst->img->maxval;
            root->val[p] = p;
            pred->val[p] = NIL;
            iftInsertGQueue(&Q, p);
        }
        S = S->next;
    }

    /* Image Foresting Transform */
    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        iftInsertSet(&Processed, p);

        u = iftGetVoxelCoord(basins, p);
        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(basins, v))
            {
                q = iftGetVoxelIndex(basins, v);
                if (Q->L.elem[q].color != BLACK)
                {
                    avg = basins->maxval - basins->val[q];
                    tmp = MIN(pathval->val[p], avg);
                    if ((tmp > pathval->val[q]) || ((pred->val[q] == p)))
                    {
                        if (Q->L.elem[q].color == GRAY)
                            iftRemoveGQueueElem(Q, q);
                        pred->val[q]  = p;
                        root->val[q]  = root->val[p];
                        label->val[q] = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
                if (label->val[p] != label->val[q])
                {
                    /* p and q must be in
                      the Frontier set */
                    if (iftBMapValue(inFrontier, p) == 0)
                    {
                        iftInsertSet(&Frontier, p);
                        iftBMapSet1(inFrontier, p);
                    }
                    if (iftBMapValue(inFrontier, q) == 0)
                    {
                        iftInsertSet(&Frontier, q);
                        iftBMapSet1(inFrontier, q);
                    }
                }
            }
        }
    }
    iftSetRemovalPolicy(fst->Q, MAXVALUE);
    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        Q->L.elem[p].color = WHITE;
    }

    if ((Frontier != NULL) && (num_smooth_iterations >= 1))
    {
        filtered_labels = iftRelaxBoundaries(fst, &Frontier, inFrontier, border_weight, norm_factor, prev_weight, next_weight, &dilated_boundary, num_smooth_iterations);
        //iftSet *aux = iftSetCopy(dilated_boundary);
        iftFixConnectivity(fst, filtered_labels, dilated_boundary);

        /*while(aux != NULL)
        {
          p = iftRemoveSet(&aux);
          prev_weight->val[p] = 1.0;
          next_weight->val[p] = 1.0;
        }
        iftDestroySet(&aux);*/
    }
    iftDestroySet(&Frontier);
    //iftDestroySet(&dilated_boundary);
    iftDestroyBMap(&inFrontier);
}
void iftDiffRelaxedWatershed(iftImageForest *fst, iftLabeledSet *seed, int num_smooth_iterations, float smooth_factor)
{
  iftAdjRel *A=fst->A;
  iftGQueue *Q=fst->Q;
  iftVoxel   u,v;
  int        i,p,q,tmp;
  char       trees_for_removal;
  iftSet    *Frontier=NULL, *Processed=NULL;
  iftBMap   *inFrontier=iftCreateBMap(fst->img->n);
  iftLabeledSet *S;
  iftImage  *pathval=fst->pathval,*pred=fst->pred,*label=fst->label;
  iftImage  *root=fst->root, *basins=fst->img;
  iftFImage *border_weight=iftSmoothWeightImage(basins,smooth_factor);
  iftFImage *norm_factor=iftWeightNormFactor(border_weight,A);

  // Verify if there are trees for removal

  trees_for_removal=0;
  S = seed;
  while (S != NULL) { 
    if ((S->label == NIL)&& 
    (label->val[S->elem]>0)){ // Removal marker
      trees_for_removal=1;
      break;
    }
    S = S->next;
  }

  // Remove marked trees  
  if (trees_for_removal) {
    Frontier = iftCompRemoval(fst,seed);
    while (Frontier != NULL) {
      p = iftRemoveSet(&Frontier); 
      iftInsertGQueue(&Q,p);
    }
  }
 
  // Trivial path initialization for new seeds 

  S = seed;
  while(S != NULL){
    p=S->elem;
    if (S->label != NIL){
      if (Q->L.elem[p].color == GRAY) { /* p is also a frontier voxel,
                     but the priority is it as a seed. */
    iftRemoveGQueueElem(Q,p);
      }
      label->val[p]=S->label; 
      pathval->val[p]=0; 
      root->val[p]=p; 
      pred->val[p]=NIL; 
      iftInsertGQueue(&Q,p);
    }
    S = S->next;
  }

  /* Image Foresting Transform */

  while(!iftEmptyGQueue(Q)) {
    p=iftRemoveGQueue(Q);
    iftInsertSet(&Processed,p);

    u = iftGetVoxelCoord(basins,p);
    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(basins,v)){ 
    q = iftGetVoxelIndex(basins,v);
    if (Q->L.elem[q].color != BLACK){
      tmp = MAX(pathval->val[p],basins->val[q]);
      if ((tmp < pathval->val[q]) || ((pred->val[q]==p)))
    { 
        if (Q->L.elem[q].color == GRAY) { 
          iftRemoveGQueueElem(Q,q);
        }
        pred->val[q]  = p;
        root->val[q]  = root->val[p];
        label->val[q] = label->val[p];
        pathval->val[q]  = tmp;
        iftInsertGQueue(&Q, q);
      }
    }else{
      if (label->val[p] != label->val[q]){ /* p and q must be in
                                the Frontier set */
        if (iftBMapValue(inFrontier,p) == 0){
          iftInsertSet(&Frontier,p);
          iftBMapSet1(inFrontier,p);
        }
        if (iftBMapValue(inFrontier,q) == 0){
          iftInsertSet(&Frontier,q);
          iftBMapSet1(inFrontier,q);
        }
      }
    }
      }
    }
  }

  while(Processed != NULL){
    p = iftRemoveSet(&Processed);
    Q->L.elem[p].color = WHITE;
  }

  if ((Frontier != NULL)&&(num_smooth_iterations >= 1)){
    iftSmoothFrontier(fst,&Frontier,inFrontier,border_weight,norm_factor,num_smooth_iterations);
  }

  iftDestroySet(&Frontier);
  iftDestroyBMap(&inFrontier);
  iftDestroyFImage(&border_weight);
  iftDestroyFImage(&norm_factor);
  
}
iftImage *iftRelaxBoundaries(iftImageForest *fst, iftSet **Frontier, iftBMap *inFrontier, iftFImage *border_weight, iftFImage *norm_factor, iftFImage *prev_weight, iftFImage *next_weight, iftSet **dilated_boundary, int num_smooth_iterations)
{
    iftImage  *prev_label, *next_label, *filtered_labels;
    float     *sum, max_membership;
    int        l, i, p, q, max_label, iter;
    iftSet    *prev_frontier = *Frontier, *next_frontier = NULL;
    iftVoxel   u, v;
    iftAdjRel *A = fst->A;

    /* Initialization */
    prev_label  = iftCopyImage(fst->label);
    next_label  = iftCopyImage(fst->label);
    iftMaximumValue(prev_label);
    sum         = iftAllocFloatArray(prev_label->maxval + 1);

    /* Smooth frontier and reset its path values */
    for (iter = 0; iter < num_smooth_iterations; iter++)
    {

        while (prev_frontier != NULL)
        {
            p = iftRemoveSet(&prev_frontier);
            iftInsertSet(&next_frontier, p);
            u   = iftGetVoxelCoord(prev_label, p);


            for (l = 0; l <= prev_label->maxval; l++)
                sum[l] = 0.0;

            for (i = 1; i < A->n; i++)
            {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(prev_label, v))
                {
                    q = iftGetVoxelIndex(prev_label, v);
                    sum[prev_label->val[q]] += prev_weight->val[q] * border_weight->val[q];
                    if (iftBMapValue(inFrontier, q) == 0)
                    {
                        /* expand frontier */
                        //if (fst->pred->val[q]!=NIL){
                        iftInsertSet(&next_frontier, q);
                        iftBMapSet1(inFrontier, q);
                        //}
                    }
                }
            }

            for (l = 0; l <= prev_label->maxval; l++) //Created with maxval+1
                sum[l]  = sum[l] / norm_factor->val[p];

            max_membership = -INFINITY_FLT; max_label = NIL;
            for (l = 0; l <= prev_label->maxval; l++) //Created with maxval+1
            {
                if (sum[l] > max_membership)
                {
                    max_membership = sum[l];
                    max_label      = l;
                }
            }
            next_label->val[p]  = max_label;
            next_weight->val[p] = sum[max_label];
        }

        prev_frontier = next_frontier;
        next_frontier = NULL;

        for (int r = 0; r < prev_label->n; r++)
        {
            prev_weight->val[r] = next_weight->val[r];
            prev_label->val[r]  = next_label->val[r];
        }
    }

    filtered_labels = prev_label;
    *dilated_boundary = iftSetCopy(prev_frontier);

    free(sum);
    *Frontier  = NULL;
    iftDestroySet(&prev_frontier);
    iftDestroyImage(&next_label);

    return filtered_labels;
}

void iftFixConnectivity (iftImageForest *fst, iftImage *filtered_labels, iftSet *dilated_boundary)
{
    int i, p, q, r, tmp;
    iftVoxel   u, v;
    iftSet *union_subtrees = NULL, *curr_subtree = NULL, *Processed = NULL;
    iftAdjRel *A = fst->A;

    /* Fix the forest by first making available to be conquered all
       voxels whose label has changed and their subtrees  */
    while (dilated_boundary != NULL)
    {
        p   = iftRemoveSet(&dilated_boundary);
        u   = iftGetVoxelCoord(filtered_labels, p);
        //If the label has changed
        if (fst->label->val[p] != filtered_labels->val[p])
        {
            iftInsertSet(&union_subtrees, p);
            iftInsertSet(&curr_subtree, p);

            while (curr_subtree != NULL)
            {
                r = iftRemoveSet(&curr_subtree);
                u = iftGetVoxelCoord(fst->pred, r);
                fst->pathval->val[r] = -INFINITY_INT;
                fst->pred->val[r] = NIL;

                for (i = 1; i < A->n; i++)
                {
                    v = iftGetAdjacentVoxel(A, u, i);
                    if (iftValidVoxel(fst->pred, v))
                    {
                        q = iftGetVoxelIndex(fst->pred, v);
                        //If its neighbour predecessor pointer is pointing to r, so the neighbour also needs to be conquered.
                        if (fst->pred->val[q] == r)
                        {
                            iftInsertSet(&union_subtrees, q);
                            iftInsertSet(&curr_subtree, q);
                        }
                    }
                }
            }
        }
    }
    /* Insert in priority queue the seed voxels, which will be the
    neighbors with the same label of the region to be conquered. */
    while (union_subtrees != NULL)
        //for (p=0; p < fst->label->n; p++)
    {
        p   = iftRemoveSet(&union_subtrees);
        u   = iftGetVoxelCoord(filtered_labels, p);
        //p   = iftRemoveSet(&union_subtrees);
        if (fst->pathval->val[p] == -INFINITY_INT)
        {
            u   = iftGetVoxelCoord(filtered_labels, p);
            for (i = 1; i < A->n; i++)
            {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(fst->pred, v))
                {
                    q = iftGetVoxelIndex(fst->pred, v);
                    if ((filtered_labels->val[q] == fst->label->val[q] && fst->label->val[q] == filtered_labels->val[p]) && (fst->pathval->val[q] != -INFINITY_INT))
                    {
                        if (fst->Q->L.elem[q].color == WHITE)
                        {
                          fst->pathval->val[q] = fst->img->maxval;
                          iftInsertGQueue(&fst->Q, q);
                        }
                    }
                }
            }
        }
    }
    /* execute the IFT to reconstruct the forest under the new labeling
       constraint. This forest is not optimum, since this is a relaxed
       IFT, but it maintains the connectivity between roots and voxels
       of the same label, respecting the filtering process. */
    while (!iftEmptyGQueue(fst->Q))
    {
        p = iftRemoveGQueue(fst->Q);
        iftInsertSet(&Processed, p);
        u = iftGetVoxelCoord(fst->img, p);

        for (i = 1; i < A->n; i++)
        {
            v  = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(fst->img, v))
            {
                q = iftGetVoxelIndex(fst->img, v);
                if (fst->Q->L.elem[q].color != BLACK)
                {
                    tmp = MIN(fst->pathval->val[p], fst->img->val[q]);
                    if (tmp > fst->pathval->val[q] || ((fst->pred->val[q] == p)))
                    {
                        if (fst->Q->L.elem[q].color == GRAY)
                            iftRemoveGQueueElem(fst->Q, q);

                        fst->root->val[q]         = fst->root->val[p];
                        fst->pred->val[q]         = p;
                        filtered_labels->val[q]   = filtered_labels->val[p];
                        fst->pathval->val[q]      = tmp;
                        iftInsertGQueue(&fst->Q, q);
                    }
                }
            }
        }
    }
    iftSetRemovalPolicy(fst->Q, MAXVALUE);
    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        fst->Q->L.elem[p].color = WHITE;
    }

    iftDestroySet(&curr_subtree);
    iftDestroySet(&union_subtrees);
    iftDestroySet(&Processed);

    //This updates the forest with the filtered label map
    iftDestroyImage(&fst->label);
    fst->label = filtered_labels;

    /* Verify forest consistency: This can be removed when we decide for
       the final procedure. */
    /*for (p=0; p < fst->label->n; p++)
    {
      r = p;
      while (fst->pred->val[r]!=NIL)
      {
        if (fst->label->val[r] != fst->label->val[fst->pred->val[r]])
        {
          iftWarning("\nIncorrect reconstruction of the label map","iftRelaxBoundaries");
          fprintf(stderr, "Voxel: %d, %d. ", iftGetXCoord(fst->label, r), iftGetYCoord(fst->label, r));
          fprintf(stderr, "Label: %d, Pred Label: %d\n", fst->label->val[r], fst->label->val[fst->pred->val[r]]);
          fprintf(stderr, "Cost: %d, Pred cost: %d\n", fst->pathval->val[r], fst->pathval->val[fst->pred->val[r]]);
          fprintf(stderr, "Pred of predecessor: %d\n", fst->pred->val[fst->pred->val[r]]);
          iftError("Fix me", "iftRelaxBoundaries");
        }
        r = fst->pred->val[r];
      }
    }*/

}

/* You must call iftMaxiumValue(fst->img) before calling this function.
The iftLabeledSet is a set with the new seeds of a given iteration, not an incremental seed set from the beginning.
*/
iftBMap *iftDiffFuzzyConnectedness(iftImageForest *fst, iftLabeledSet *seed)
{
    iftAdjRel *A = fst->A;
    iftGQueue *Q = fst->Q;
    iftVoxel   u, v;
    int        i, p, q, tmp;
    float      avg = 0;
    char       trees_for_removal;
    iftSet    *Frontier = NULL, *Processed = NULL;
    iftBMap   *inFrontier = iftCreateBMap(fst->img->n);
    iftLabeledSet *S;
    iftImage  *pathval = fst->pathval, *pred = fst->pred, *label = fst->label;
    iftImage  *root = fst->root, *basins = fst->img;

    // Verify if there are trees for removal
    trees_for_removal = 0;
    S = seed;
    while (S != NULL)
    {
        if ((S->label == NIL) &&
                (label->val[S->elem] > 0)) // Removal marker
        {
            trees_for_removal = 1;
            break;
        }
        S = S->next;
    }
    // Remove marked trees
    if (trees_for_removal)
    {
        Frontier = iftCompRemoval(fst, seed);
        while (Frontier != NULL)
        {
            p = iftRemoveSet(&Frontier);
            iftInsertGQueue(&Q, p);
        }
    }

    // Trivial path initialization for new seeds
    S = seed;
    while (S != NULL)
    {
        p = S->elem;
        if (S->label != NIL)
        {
            if (Q->L.elem[p].color == GRAY)
            {
                /* p is also a frontier voxel,
                   but the priority being a seed. */
                iftRemoveGQueueElem(Q, p);
            }
            label->val[p] = S->label;
            pathval->val[p] = fst->img->maxval;
            root->val[p] = p;
            pred->val[p] = NIL;
            iftInsertGQueue(&Q, p);
        }
        S = S->next;
    }

    /* Image Foresting Transform */
    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        iftInsertSet(&Processed, p);

        u = iftGetVoxelCoord(basins, p);
        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(basins, v))
            {
                q = iftGetVoxelIndex(basins, v);
                if (Q->L.elem[q].color != BLACK)
                {
                    avg = basins->maxval - basins->val[q];
                    tmp = MIN(pathval->val[p], avg);
                    if ((tmp > pathval->val[q]) || ((pred->val[q] == p)))
                    {
                        if (Q->L.elem[q].color == GRAY)
                            iftRemoveGQueueElem(Q, q);
                        pred->val[q]  = p;
                        root->val[q]  = root->val[p];
                        label->val[q] = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
                if (label->val[p] != label->val[q])
                {
                    //p and q must be in the inFrontier bitmap 
                    iftBMapSet1(inFrontier, p);
                    iftBMapSet1(inFrontier, q);
                }

            }
        }
    }
    iftSetRemovalPolicy(fst->Q, MAXVALUE);

    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        Q->L.elem[p].color = WHITE;
    }
    return inFrontier;
}

iftImage *iftIRFC(iftImage *basins, iftAdjRel *A, iftLabeledSet *seed)
{
  iftImage  *pathval=NULL,*label=NULL;
  iftGQueue  *Q=NULL;
  int      i,p,q,tmp;
  iftVoxel    u,v;
  iftLabeledSet *S=seed;
  
  // Initialization 

  pathval  = iftCreateImage(basins->xsize,basins->ysize,basins->zsize);
  label = iftCreateImage(basins->xsize,basins->ysize,basins->zsize);
  Q     = iftCreateGQueue(iftMaximumValue(basins)+1,basins->n,pathval->val);
  iftSetRemovalPolicy(Q, MAXVALUE);

  for (p=0; p < basins->n; p++) {
    pathval->val[p]=-INFINITY_INT;
  }

  while (S != NULL){
    p = S->elem;
    label->val[p]=S->label;
    pathval->val[p] = basins->maxval;
    iftInsertGQueue(&Q,p);
    S = S->next;
  }

  // Image Foresting Transform
  int val_p=0, val_q=0;
  float avg=0;
  while(!iftEmptyGQueue(Q))
  {
    p=iftRemoveGQueue(Q);
    u = iftGetVoxelCoord(basins,p);

    for (i=1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A,u,i);

      if (iftValidVoxel(basins,v))
      {
        q = iftGetVoxelIndex(basins,v);
        if (Q->L.elem[q].color != BLACK)
        {
          val_q = basins->maxval-basins->val[q];
          val_p = basins->maxval-basins->val[p];
          avg = (val_p + val_q)/2.0;

          tmp = MIN(pathval->val[p],avg);
          if (tmp > pathval->val[q])
          {
            if (Q->L.elem[q].color == GRAY)
              iftRemoveGQueueElem(Q, q);
            label->val[q] = label->val[p];
            pathval->val[q]  = tmp;
            iftInsertGQueue(&Q, q);
          }
        }
      }
    }
  }

iftDestroyGQueue(&Q);
iftDestroyImage(&pathval);

iftCopyVoxelSize(basins,label);

return(label);
}




/* BEFORE HUGE CHANGE ON THE ALGORITHM */
/**
@brief Computes the watershed transform for a given image and seed set.
 
For a given Image Forest and Labeled Seed Set, the watershed transform is 
executed using the IFT algorithm. The labeled seed set can be generated by the 
user via a GUI or using the robots. On both cases, the first execution should 
be newly added seeds. For subsequent executions (i.e corrections), the labeled 
seed set must be only the seeds added for the correction and not the union of 
the seeds from the first iteration with seeds from the second iteration.

In this particular function, a bitmap stored in the iftImageForest 
structure, (inFrontier), is updated accordingly. 
The inFrontier bitmap contains the frontier of the last execution only. 
This is used in post-processing function, such as iftRelaxObjects.
For a frontier of the completed label map, not only the wave propagation from
the last execution, please read function iftSetLabelMapFrontier.

Complexity: O(n).

@param  fst  iftImageForest. Forest structure created with a gradient image.
@param  seed iftLabeledSet.  List of pixels with image index and seed label.

@return void. All forest maps are updated by reference.
*/

void iftDiffWatershed(iftImageForest *fst, iftLabeledSet *seed, iftBMap *propagation)
{
    iftAdjRel *A = fst->A;
    iftGQueue *Q = fst->Q;
    iftVoxel   u, v;
    int        i, p, q, tmp;
    char       trees_for_removal;
    iftSet    *Frontier = NULL, *Processed = NULL;
    iftLabeledSet *S;
    iftImage  *pathval = fst->pathval, *pred = fst->pred, *label = fst->label;
    iftImage  *root = fst->root, *basins = fst->img;

    // Verify if there are trees for removal
    trees_for_removal = 0;
    S = seed;
    while (S != NULL)
    {
        if (S->label == NIL) // Removal marker
        {
            trees_for_removal = 1;
            break;
        }
        S = S->next;
    }

    // Remove marked trees
    if (trees_for_removal)
    {
        Frontier = iftTreeRemoval(fst, seed);
        while (Frontier != NULL)
        {
            p = iftRemoveSet(&Frontier);
            iftInsertGQueue(&Q, p);
        }
    }

    // Trivial path initialization for new seeds
    S = seed;
    while (S != NULL)
    {
        p = S->elem;
        if (S->label != NIL)
        {
            if (Q->L.elem[p].color == GRAY)
            {
                /* p is also a frontier voxel, but the priority is it as a seed. */
                iftRemoveGQueueElem(Q, p);
            }
            label->val[p] = S->label;
            pathval->val[p] = 0;
            root->val[p] = p;
            pred->val[p] = NIL;
            iftInsertGQueue(&Q, p);
        }
        S = S->next;
    }

    iftFillBMap(propagation, 0);
    /* Image Foresting Transform */
    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        iftInsertSet(&Processed, p);
        u = iftGetVoxelCoord(basins, p);
        iftBMapSet1(propagation, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(basins, v))
            {
                q = iftGetVoxelIndex(basins, v);
                if (Q->L.elem[q].color != BLACK)
                {
                    tmp = MAX(pathval->val[p], basins->val[q]);
                    if ((tmp < pathval->val[q]) || ((pred->val[q] == p)))
                    {
                        if (Q->L.elem[q].color == GRAY)
                        {
                            iftRemoveGQueueElem(Q, q);
                        }
                        pred->val[q]  = p;
                        root->val[q]  = root->val[p];
                        label->val[q] = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }
    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        Q->L.elem[p].color = WHITE;
    }
}


/**
@brief Relax forest optimality in order to smooth the label map.
 
Smooth the boundaries of all objects in the label image. This function relies 
on the bitmap inFrontier that is inside the iftImageForest structure. 
These bitmaps are automatically created with the forest but are only updated
 on iftDiffWatershed, iftLabel2Forest or iftSetLabelMapFrontier functions.

Besides the object smoothing, this function will also correct any possible
inconsistency of the forest that could be caused during the relaxation. More
details of the forest inconsistency problem during post-processing filters 
are described in Nikolas Moya's master dissertation.

Usually, the inFrontier bitmap will contain the frontier of the last wave 
propagation from the iftDiffWatershed segmentation method. If you want a more
general smoothing, within the whole object boundary, invoke method 
iftSetLabelMapFrontier before invoking iftRelaxObjects.
The complete smoothing of the boundary can be useful for smoothing the delineated
object, after all user corrections, for example.

This smoothing function is independent from the segmentation method. This means
that a label map created by other functions can be smoothed by calling the
function iftLabel2Forest first and the using the forest created as a parameter
on this function.

@param  fst  iftImageForest. iftDiffWatershed or iftLabel2Forest needs to be called first.
@param  num_smooth_iterations int. Number of smooth iterations. (default 10).
@param  smooth_factor. float. Smooth factor in range [0, 1] (default 0.5).

@return void. The forest optimality is relaxed but it remains consistent.
*/
void iftRelaxObjects(iftImageForest *fst, iftBMap *propagation, iftSmoothBorder *smooth)
{
    iftImage  *prev_label, *next_label;
    iftFImage *prev_weight, *next_weight;
    float     *sum, max_membership;
    int        l, i, p, q, r, max_label, iter, tmp;
    int       frontierSize = 0;
    iftBMap   *inFrontier = NULL;
    iftSet    *prev_frontier = NULL, *next_frontier = NULL, *Subtree = NULL, *Processed = NULL;
    iftSet    *subtree_union = NULL;
    iftVoxel   u, v;
    iftAdjRel *A = fst->A;
    iftAdjRel *border = iftSpheric(sqrt(3));
    timer *t1, *t2;

    /* Initialization */
    prev_label  = iftCopyImage(fst->label);
    next_label  = iftCopyImage(fst->label);
    sum         = iftAllocFloatArray(iftMaximumValue(prev_label) + 1);
    prev_weight = iftCreateFImage(prev_label->xsize, prev_label->ysize, prev_label->zsize);
    next_weight = iftCreateFImage(next_label->xsize, next_label->ysize, next_label->zsize);
    inFrontier = iftCreateBMap(fst->label->n);

    // B_p <- B; // Converting from iftBMap to iftSet.
    prev_frontier = NULL;

    t1 = iftTic();
    printf("----------------------------------------\n");
    for (p = 0; p < propagation->n; p++)
    {
        /* M_p and M_n intially set to 1.0 */
        prev_weight->val[p] = next_weight->val[p] = 1.0; 

        u  = iftGetVoxelCoord(fst->img, p);
        if (iftBMapValue(propagation, p))
        {
            for (i = 1; i < border->n; i++)
            {
                v = iftGetAdjacentVoxel(border, u, i);
                if (iftValidVoxel(prev_label, v))
                {
                    q = iftGetVoxelIndex(fst->img, v);
                    if (fst->label->val[p] != fst->label->val[q])
                    {
                        frontierSize++;
                        if (iftBMapValue(inFrontier, p) == 0)
                        {
                            iftBMapSet1(inFrontier, p);
                            iftInsertSet(&prev_frontier, p);
                        }
                        if (iftBMapValue(inFrontier, q) == 0)
                        {
                            iftBMapSet1(inFrontier, q);
                            iftInsertSet(&prev_frontier, q);
                        }
                    }

                }
            }
        }
    }
    t2 = iftToc();
    printf("Differential border: %.2f\n", iftCompTime(t1,t2));

    if (frontierSize == 0)
    {
        iftWarning("Cannot smooth boundaries. Frontier is empty.", "iftRelaxObjects");
        return;
    }
    /* Smooth frontier and reset its path values */
    t1 = iftTic();
    for (iter = 0; iter < smooth->smooth_iterations; iter++)
    {
        while (prev_frontier != NULL)
        {
            p = iftRemoveSet(&prev_frontier);
            iftInsertSet(&next_frontier, p);
            u   = iftGetVoxelCoord(prev_label, p);

            for (l = 0; l <= prev_label->maxval; l++)
                sum[l] = 0.0;

            for (i = 1; i < A->n; i++)
            {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(prev_label, v))
                {
                    q = iftGetVoxelIndex(prev_label, v);
                    sum[prev_label->val[q]] += prev_weight->val[q] * smooth->border_weight->val[q];
                    
                    if (iftBMapValue(inFrontier, q) == 0) /* expand frontier */
                    {
                        if (fst->pred->val[q] != NIL)
                        {
                            iftInsertSet(&next_frontier, q);
                            iftBMapSet1(inFrontier, q);
                        }
                    }
                }
            }

            for (l = 0; l <= prev_label->maxval; l++)
                sum[l]  = sum[l] / smooth->norm_factor->val[p];

            max_membership = -INFINITY_FLT; max_label = NIL;
            for (l = 0; l <= prev_label->maxval; l++)
            {
                if (sum[l] > max_membership)
                {
                    max_membership = sum[l];
                    max_label      = l;
                }
            }
            next_label->val[p]  = max_label;
            next_weight->val[p] = sum[max_label];
        }

        prev_frontier = next_frontier;
        next_frontier = NULL;

        for (r = 0; r < prev_label->n; r++)
        {
            prev_weight->val[r] = next_weight->val[r];
            prev_label->val[r]  = next_label->val[r];
        }
    }
    t2 = iftToc();
    printf("Smoothing time: %.2f\n", iftCompTime(t1, t2));
    free(sum);
    iftDestroyFImage(&prev_weight);
    iftDestroyFImage(&next_weight);
    iftDestroyImage(&next_label);
    //iftDestroySet(&prev_frontier);

    // Uncomment the following three lines to avoid fixing the forest consistency
    // iftDestroyImage(&fst->label);
    // fst->label = prev_label;
    // return;

    
    /* Fix the forest by first making available to be conquered all
       voxels whose label has changed and their subtrees  */    
    //for (p = 0; p < fst->label->n; p++)
    t1 = iftTic();
    while(prev_frontier != NULL)
    {
        p = iftRemoveSet(&prev_frontier);
        /* Check for label changes */
        if (fst->label->val[p] != prev_label->val[p])
        {
            iftInsertSet(&Subtree, p);
            iftInsertSet(&subtree_union, p);
            
            //All subtree needs to become available to be conquered
            while (Subtree != NULL)
            {
                r = iftRemoveSet(&Subtree);
                u = iftGetVoxelCoord(fst->pred, r);
                
                fst->pathval->val[r] = INFINITY_INT; //Allow the pixel to be conquered
                
                for (i = 1; i < A->n; i++)
                {
                    v = iftGetAdjacentVoxel(A, u, i);
                    if (iftValidVoxel(fst->pred, v))
                    {
                        q = iftGetVoxelIndex(fst->pred, v);
                        /* q is in the subtree rooted at r. Therefore, q needs to be available to be conquered. */
                        if (fst->pred->val[q] == r)
                        {
                          iftInsertSet(&Subtree, q);
                          iftInsertSet(&subtree_union, q);
                        }
                    }
                }
            }
        }
    }
    t2 = iftToc();
    printf("Subtree traversal: %.2f\n", iftCompTime(t1,t2));

    /* Insert in priority queue the seed voxels, which will be voxels in
       the frontier of the available region whose labels are the same of
       the neighbor in the region. The initial path value of such seeds
       must be the current path value. Also insert as seeds in the queue
       the voxels in that region, whose predecessor outside the region
       maintained the old label, as long as they are true seeds (i.e.,
       they also have a neighbor outside the region with the same new
       label of them). In this case, the initial cost is the one of the
       predecessor voxel to avoid backward path propagation. */

    //for (p = 0; p < fst->label->n; p++)
    t1 = iftTic();
    while (subtree_union != NULL)
    {
         //if (fst->pathval->val[p] == INFINITY_INT)
         //{
            p = iftRemoveSet(&subtree_union);
            int pp = fst->pred->val[p]; /* predecessor of p */
            u = iftGetVoxelCoord(fst->pred, p);

            for (i = 1; i < A->n; i++)
            {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(fst->pred, v))
                {
                    q = iftGetVoxelIndex(fst->pred, v);
                    if (fst->pathval->val[q] != INFINITY_INT) /* q is outside the available region */
                    {
                        if (prev_label->val[q] == prev_label->val[p]) /* the label of q is the new label of p. Then p can be a true seed, if its predecessor is outside the available region. */
                        {
                            if (fst->Q->L.elem[q].color == WHITE && fst->pathval->val[q] == INFINITY_INT)
                            {
                                iftInsertGQueue(&fst->Q, q);
                            }
                            if (pp == NIL)
                            {
                                fst->pathval->val[p] = 0;
                                fst->pred->val[p] = NIL;
                                if (fst->Q->L.elem[p].color == WHITE)
                                    iftInsertGQueue(&fst->Q, p);
                            }
                            else if (fst->pathval->val[pp] != INFINITY_INT && prev_label->val[pp] != prev_label->val[p])   // pp is outside the available region. => its label is the old one, different from the new label of p. 
                            {
                                fst->pathval->val[p] = fst->pathval->val[pp];
                                fst->pred->val[p] = NIL;
                                if (fst->Q->L.elem[p].color == WHITE)
                                    iftInsertGQueue(&fst->Q, p);
                            }
                        }
                    }
                }
            }
        //}
    }
    t2 = iftToc();
    printf("Seed insertion: %.2f\n", iftCompTime(t1,t2));

    iftDestroyImage(&fst->label);
    fst->label = prev_label;

    /* execute the IFT to reconstruct the forest under the new labeling
       constraint. This forest is not optimum, since this is a relaxed
       IFT, but it maintains the connectivity between roots and voxels
       of the same label, respecting the filtering process. */
    t1 = iftTic();
    while (!iftEmptyGQueue(fst->Q))
    {
        p = iftRemoveGQueue(fst->Q);
        iftInsertSet(&Processed, p);

        u = iftGetVoxelCoord(fst->img, p);

        for (i = 1; i < A->n; i++)
        {
            v  = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(fst->img, v))
            {
                q = iftGetVoxelIndex(fst->img, v);
                if (fst->Q->L.elem[q].color != BLACK)
                {
                    tmp = MAX(fst->pathval->val[p], fst->img->val[q]);
                    if (tmp < fst->pathval->val[q] || ((fst->pred->val[q] == p)))
                    {
                        if (fst->Q->L.elem[q].color == GRAY)
                            iftRemoveGQueueElem(fst->Q, q);

                        fst->root->val[q]    = fst->root->val[p];
                        fst->pred->val[q]    = p;
                        fst->label->val[q]   = fst->label->val[p];
                        fst->pathval->val[q] = tmp;
                        iftInsertGQueue(&fst->Q, q);
                    }
                }
            }
        }
    }
    t2 = iftToc();
    printf("IFT execution time: %.2f\n", iftCompTime(t1,t2));
    printf("----------------------------------------\n\n");
    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        fst->Q->L.elem[p].color = WHITE;
    }
    iftDestroyBMap(&inFrontier);

    /* Verify forest consistency: This can be removed when we decide for the final procedure. */
    for (p = 0; p < fst->label->n; p++)
    {
        r = p;
        while (fst->pred->val[r] != NIL)
        {
            if (fst->label->val[r] != fst->label->val[fst->pred->val[r]])
            {
                iftWarning("Incorrect reconstruction of the label map", "iftRelaxObjects");
                // fprintf(stderr, "%d, %d:", iftGetXCoord(fst->label, r), iftGetYCoord(fst->label, r));
                // fprintf(stderr, "Label: %d, Pred Label: %d\n", fst->label->val[r], fst->label->val[fst->pred->val[r]]);
                // fprintf(stderr, "Cost: %d, Pred cost: %d\n", fst->pathval->val[r], fst->pathval->val[fst->pred->val[r]]);
                // fprintf(stderr, "Pred of predecessor: %d\n", fst->pred->val[fst->pred->val[r]]);
            }
            r = fst->pred->val[r];
        }
    }
}