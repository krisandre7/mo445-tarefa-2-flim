#include "ift.h"

void saveImage2d(iftImage *original, iftImage *seeds_image, iftImage *label_image, char *filename)
{
    int p = 0;
    iftColor yellow;
    iftColor red;
    iftColor green;

    yellow.val[0] = 255;
    yellow.val[1] = 1;
    yellow.val[2] = 148;

    red.val[0] = 76;
    red.val[1] = 85;
    red.val[2] = 255;

    green.val[0] = 137;
    green.val[1] = 95;
    green.val[2] = 79;

    //iftImage *markers_image = iftCreateColorImage(original->xsize, original->ysize, original->zsize);
    //for (p=0; p<markers_image->n; p++)
    //markers_image->val[p] = original->val[p];
    iftImage *markers_image = iftCopyImage(original);
    if (seeds_image != NULL)
    {
        for (p = 0; p < markers_image->n; p++)
        {
            //Background marker colour
            if (seeds_image->val[p] == 0)
            {
                markers_image->val[p] = red.val[0];
                markers_image->Cb[p] = red.val[1];
                markers_image->Cr[p] = red.val[2];
            }
            //Object marker colour
            else if (seeds_image->val[p] == 1)
            {
                markers_image->val[p] = yellow.val[0];
                markers_image->Cb[p] = yellow.val[1];
                markers_image->Cr[p] = yellow.val[2];
            }
            //Available to conquest
            else if (seeds_image->val[p] == 2)
            {
                // markers_image->val[p]= green.val[0];
                markers_image->Cb[p] = green.val[1];
                markers_image->Cr[p] = green.val[2];
            }
        }
    }

    //iftWriteImageP6(markers_image, "markesrimage.ppm");
    iftImage *output = iftCopyImage(markers_image);
    if (label_image != NULL)
    {
        for (p = 0; p < markers_image->n; p++)
        {
            // //Background colour
            // if (label_image->val[p] == 0)
            // {
            //     output->Cb[p] = red.val[1];
            //     output->Cr[p] = red.val[2];
            // }
            //Object colour
            if (label_image->val[p] == 1)
            {
                output->Cb[p] = yellow.val[1];
                output->Cr[p] = yellow.val[2];
            }
        }
    }
    //iftWriteImageP6(markers_image, path);
    iftWriteImageP6(output, filename);

    iftDestroyImage(&markers_image);
    iftDestroyImage(&output);
}
/*You must call iftMaxiumValue(fst->img) before calling this function.
The iftLabeledSet is a set with the new seeds of a given iteration, not an incremental seed set from the beginning.
*/
iftBMap *iftDIFT(iftImageForest *fst, iftLabeledSet *seed)
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
                    //printf("min(%d, %.0f) = %d\n", pathval->val[p], avg, tmp);
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
    iftSetRemovalPolicy(Q, MAXVALUE);
    while (Processed != NULL)
    {
        p = iftRemoveSet(&Processed);
        Q->L.elem[p].color = WHITE;
    }
    return inFrontier;
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

iftImage *iftRelax (iftImageForest *fst, iftSet **Frontier, iftBMap *inFrontier, iftFImage *border_weight,
                    iftFImage *norm_factor, iftFImage *prev_weight, iftFImage *next_weight,
                    iftSet **dilated_boundary, int num_smooth_iterations)
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
                        iftInsertSet(&next_frontier, q);
                        iftBMapSet1(inFrontier, q);
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

void iftFix(iftImageForest *fst, iftImage *filtered_labels, iftSet *dilated_boundary, iftImage *original)
{
    int i, p, q, r, tmp;
    iftVoxel   u, v;
    iftSet *union_subtrees = NULL, *curr_subtree = NULL, *Processed = NULL;
    iftAdjRel *A = fst->A;

    iftImage *seed_image = NULL;
    seed_image = iftCreateImage(fst->img->xsize, fst->img->ysize, fst->img->zsize);
    iftSetImage(seed_image, -1);


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
                seed_image->val[r] = 2; //Paint in green the available pixels

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
    {
        p   = iftRemoveSet(&union_subtrees);
        u   = iftGetVoxelCoord(filtered_labels, p);
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
                            seed_image->val[q] = filtered_labels->val[q];
                        }
                    }
                }
            }
        }
    }
    saveImage2d(original, seed_image, NULL, "available-seeds.ppm");
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
    /*for (p = 0; p < fst->label->n; p++)
    {
        r = p;
        while (fst->pred->val[r] != NIL)
        {
            if (fst->label->val[r] != fst->label->val[fst->pred->val[r]])
            {
                iftWarning("\nIncorrect reconstruction of the label map", "iftRelaxBoundaries");
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

int main(int argc, char *argv[])
{
    iftImage        *img = NULL, *basins = NULL;
    iftImage        *seed_image = NULL;
    iftImage        *filtered_labels = NULL;
    iftImageForest  *fst = NULL;
    iftLabeledSet   *seed = NULL;
    iftBMap         *bitmap = NULL;
    iftSet          *Frontier = NULL;
    iftSet          *dilated_boundary = NULL;
    iftAdjRel       *A = NULL;
    char             ext[10], *pos;
    int smooth_iterations = 0;


    if (argc != 5)
        iftError("Usage: posterimages <image.ppm> <markers> <spatial radius> <smooth_iterations> ", "main");

    pos = strrchr(argv[1], '.') + 1;
    sscanf(pos, "%s", ext);

    if (strcmp(ext, "pgm") == 0)
        img   = iftReadImageP5(argv[1]);
    else
    {
        if (strcmp(ext, "ppm") == 0)
            img   = iftReadImageP6(argv[1]);
        else
        {
            printf("Invalid image format: %s\n", ext);
            exit(-1);
        }
    }

    A      = iftCircular(atof(argv[3]));
    printf("Computing Image Basins\n");
    basins = iftImageBasins(img, A);
    fst    = iftCreateImageForest(basins, A);
    smooth_iterations = atoi(argv[4]);

    iftFImage *prev_weight = iftCreateFImage(img->xsize, img->ysize, img->zsize);
    iftFImage *next_weight = iftCreateFImage(img->xsize, img->ysize, img->zsize);

    iftFImage *border_weight = iftSmoothWeightImage(basins, 0.5);
    iftFImage *norm_factor = iftWeightNormFactor(border_weight, A);

    iftResetImageForest(fst, FIFOBREAK, MAXVALUE);
    seed = iftReadSeeds2D(argv[2], basins);
    seed_image = iftSeedImageFromLabeledSet(seed, img);

    bitmap = iftDIFT(fst, seed);
    saveImage2d(img, seed_image, fst->label, "labels-notrelaxed.pgm");
    //Convert from bitmap to set
    for (int p = 0; p < bitmap->n; p++)
    {
        prev_weight->val[p] = next_weight->val[p] = 1.0;
        if (iftBMapValue(bitmap, p) != 0)
            iftInsertSet(&Frontier, p);
    }
    filtered_labels = iftRelax(fst, &Frontier, bitmap, border_weight, norm_factor, prev_weight, next_weight, &dilated_boundary, smooth_iterations);
    saveImage2d(img, seed_image, filtered_labels, "relaxed.ppm");
    iftFix(fst, filtered_labels, dilated_boundary, img);
    saveImage2d(img, seed_image, fst->label, "fixed.ppm");

    iftDestroyImageForest(&fst);
    iftDestroyAdjRel(&A);
    iftDestroyFImage(&prev_weight);
    iftDestroyFImage(&next_weight);
    iftDestroyFImage(&border_weight);
    iftDestroyFImage(&norm_factor);
    iftDestroyImage(&img);
    iftDestroyImage(&basins);

    return (0);
}

