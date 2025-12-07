#include "ift.h"


iftAdjSet **iftImageMST(  iftImage *grad,   iftAdjRel *A)
{
    iftImage *cost = iftCreateImage(grad->xsize, grad->ysize, grad->xsize);
    iftImage *pred = iftCreateImage(grad->xsize, grad->ysize, grad->xsize);
    iftGQueue *Q = iftCreateGQueue(iftMaximumValue(grad) + 1, grad->n, cost->val);
    iftSetTieBreak(Q, LIFOBREAK);

    int argmin = -1;
    int min = IFT_INFINITY_INT;
    for (int p = 0; p < grad->n; p++) {
        if (min > grad->val[p]) {
            min = grad->val[p];
            argmin = p;
        }
    }

    for (int p = 0; p < grad->n; p++) {
        cost->val[p] = IFT_INFINITY_INT;
        pred->val[p] = IFT_NIL;
    }

    cost->val[argmin] = 0;
    iftInsertGQueue(&Q, argmin);

    while (!iftEmptyGQueue(Q))
    {
        int p = iftRemoveGQueue(Q);
        iftVoxel u = iftGetVoxelCoord(grad, p);

        for (int i = 1; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (!iftValidVoxel(grad, v))
                continue;

            int q = iftGetVoxelIndex(grad, v);

            int arc_w  = iftMax(grad->val[q], grad->val[p]);
            if (arc_w < cost->val[q])
            {
//                if (Q->L.elem[q].color == IFT_BLACK)
//                    iftWarning("I'm not sure", "iftImageMST");

                if (Q->L.elem[q].color == IFT_GRAY)
                    iftRemoveGQueueElem(Q, q);

                cost->val[q] = arc_w;
                pred->val[q] = p;

                iftInsertGQueue(&Q, q);
            }
        }
    }

    iftDestroyGQueue(&Q);
    iftDestroyImage(&cost);

    iftAdjSet **graph = iftAlloc(grad->n, sizeof *graph);

    for (int p = 0; p < grad->n; p++) {
        if (pred->val[p] != IFT_NIL) {
            int q = pred->val[p];
            int arc_w = iftMax(grad->val[p], grad->val[q]);
            iftInsertAdjSet(&graph[p], q, arc_w);
            iftInsertAdjSet(&graph[q], p, arc_w);
        }
    }

    iftDestroyImage(&pred);

    return graph;
}


iftImage *iftEvaluateLeakage(iftAdjSet **graph,   iftImage *grad,   iftImage *label)
{
    iftImage *out = iftCreateColorImage(grad->xsize, grad->ysize, grad->zsize, 8);

      int max = iftMaximumValue(grad);
    for (int p = 0; p < out->n; p++) {
        int val = 255 * grad->val[p] / max;
        iftSetRGB(out, p, val, val, val, 255);
    }

    for (int i = 0; i < label->n; i++) {
        for (  iftAdjSet *adj = graph[i]; adj; adj = adj->next)
        {
            int j = adj->node;
            if (label->val[i] != label->val[j]) {
                iftSetRGB(out, i, 255, 0, 0, 255);
                iftSetRGB(out, j, 255, 0, 0, 255);
            }
        }
    }

    return out;
}


iftImage *iftMSTLeafs(iftAdjSet **graph,   iftImage *label)
{
    iftImage *out = iftCreateColorImage(label->xsize, label->ysize, label->zsize, 8);

      int max = iftMaximumValue(label);
    for (int p = 0; p < out->n; p++) {
        int val = 255 * label->val[p] / max;
        iftSetRGB(out, p, val, val, val, 255);
    }

    for (int i = 0; i < label->n; i++) {
        int count = 0;
        for (  iftAdjSet *adj = graph[i]; adj; adj = adj->next, count++);
        if (count == 1) {
            if (label->val[i] == 1)
                iftSetRGB(out, i, 0, 0, 255, 255);
            else
                iftSetRGB(out, i, 255, 0, 0, 255);
        }
    }

    return out;
}


iftLabeledSet *iftFindLeakageAndSplit(iftAdjSet **graph,   iftImage *label)
{
    iftLabeledSet *set = NULL;

    for (int i = 0; i < label->n; i++) {
        for (iftAdjSet *adj = graph[i]; adj; )
        {
            int j = adj->node;
            int arc_w = (int) adj->arcw;
            adj = adj->next; /* must be before node removal */
            if (label->val[i] != label->val[j])
            {
                /* node might leak on more than one direction,
                 * thus the leakage barrier is the lowest value */
                iftInsertLabeledSetMarkerAndHandicap(&set, i, label->val[i], -1, arc_w);
                iftInsertLabeledSetMarkerAndHandicap(&set, j, label->val[j], -1, arc_w);

                iftRemoveAdjSetNode(&graph[i], j);
                iftRemoveAdjSetNode(&graph[j], i);
            }
        }
    }

    return set;
}


iftImage *iftFindSimpleCore(iftAdjSet **graph,   iftImage *grad,   iftLabeledSet *leaks)
{
    iftImage *cost = iftCreateImage(grad->xsize, grad->ysize, grad->zsize);
    iftImage *root = iftCreateImage(grad->xsize, grad->ysize, grad->zsize);

    iftGQueue *Q = iftCreateGQueue(iftMaximumValue(grad) + 1, grad->n, cost->val);

    for (int p = 0; p < cost->n; p++) {
        cost->val[p] = IFT_INFINITY_INT;
        root->val[p] = IFT_NIL;
    }

    for (  iftLabeledSet *s = leaks; s; s = s->next) {
        int p = s->elem;
        cost->val[p] = s->handicap;
        root->val[p] = p;
        iftInsertGQueue(&Q, p);
    }

    while (!iftEmptyGQueue(Q))
    {
        int p = iftRemoveGQueue(Q);
        for (  iftAdjSet *adj = graph[p]; adj; adj = adj->next)
        {
            int q = adj->node;

            if (cost->val[p] >= adj->arcw &&
                cost->val[q] == IFT_INFINITY_INT) /* new node */
            {
                if (Q->L.elem[q].color == IFT_BLACK)
                    iftWarning("Is this monotonic increasing", "iftFindSimpleCore");

                if (Q->L.elem[q].color == IFT_GRAY)
                    iftRemoveGQueueElem(Q, q);

                cost->val[q] = cost->val[p];
                root->val[q] = root->val[p];
                iftInsertGQueue(&Q, q);
            }
        }
    }

    iftDestroyImage(&cost);
    iftDestroyGQueue(&Q);

    iftImage *comp = iftCreateImage(grad->xsize, grad->ysize, grad->zsize);
    iftSetImage(comp, IFT_NIL);

    int count = 1;
    for (int p = 0; p < root->n; p++)
    {
        int r = root->val[p];
        if (comp->val[r] == IFT_NIL) {
            comp->val[r] = count;
            count++;
        }
    }

    for (int p = 0; p < root->n; p++)
        comp->val[p] = comp->val[root->val[p] ];

    iftDestroyImage(&root);

    return comp;
}


int main(int argc, const char *argv[])
{
    if (argc != 3)
        iftError("iftFindCore <input path> <seeds path>", "main");

    const char *img_path   = argv[1];
    const char *seeds_path = argv[2];

    iftImage *img = iftReadImageByExt(img_path);
    iftLabeledSet *seeds = iftReadSeeds(img, seeds_path);
    iftAdjRel *A = iftCircular(1.0f);

    iftImage *gradient = iftImageBasins(img, A);
    iftImage *label = iftWatershed(gradient, A, seeds, NULL);

    iftAdjSet **graph = iftImageMST(gradient, A);
    iftImage *ex = iftEvaluateLeakage(graph, gradient, label);
    iftWriteImageByExt(ex, "leakage.png");
    iftDestroyImage(&ex);
    ex = iftMSTLeafs(graph, label);
    iftWriteImageByExt(ex, "leafs.png");
    iftDestroyImage(&ex);

    iftLabeledSet *leaks = iftFindLeakageAndSplit(graph, label);
    iftImage *comp = iftFindSimpleCore(graph, gradient, leaks);

    iftImage *aux = NULL;
    aux = iftNormalize(comp, 0, 255);
    iftWriteImageByExt(aux, "components.pgm");
    iftDestroyImage(&aux);

    aux = iftNormalize(label, 0, 255);
    iftWriteImageByExt(aux, "label.pgm");
    iftDestroyImage(&aux);

    iftDestroyImage(&comp);
    iftDestroyLabeledSet(&leaks);
    iftDestroyImage(&label);
    iftDestroyImage(&gradient);
    iftDestroyAdjRel(&A);
    iftDestroyLabeledSet(&seeds);
    iftDestroyImage(&img);

    return 0;
}

