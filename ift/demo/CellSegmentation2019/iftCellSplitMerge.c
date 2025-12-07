#include "ift.h"
#include "ift/core/tools/Dialog.h"

/* OLHAR PESO DOS ARCO QUE SAEM DO MESMO NÓ E SUMIR COM O ARCO DE VALOR MUITO ABAIXO DO RESTO */

double *iftComputeCellGraph(iftImageForest *fst, iftAdjRel *A, int n_labels);
iftImage *iftCellGraphToImage(iftImageForest *fst, iftAdjRel *A, double *graph, int n_labels);
void iftSplitWeakLinks(iftImageForest *fst, iftAdjRel *A, double *graph, int n_labels, float tol);


int main(int argc, const char* argv[])
{
    const char *img_path = argv[1];
      int height = atoi(argv[2]);
      float tol = (float) atof(argv[3]);

    if (argc != 4) {
        iftError("iftCellSplitMerge <input img> <watershed height> <arc-weight tolerance (0, 1)>", "iftJoiningCells");
    }

    timer *tic = iftTic();

    iftImage *img = iftReadImageByExt(img_path);
	iftAdjRel *A = iftSpheric(1.0f);

    iftImageForest *fst = iftCreateImageForest(img, A);
    iftImage *marker = iftAddValue(img, height);

    iftWaterGrayForest(fst, marker);

    int n_labels = iftMaximumValue(fst->label) + 1;
    double *graph = iftComputeCellGraph(fst, A, n_labels);

    iftImage *border = iftCellGraphToImage(fst, A, graph, n_labels);
    iftWriteImageByExt(border, "border.nii.gz");
//    iftWriteImageByExt(fst->label, "mask.nii.gz");
    iftDestroyImage(&border);

    iftSplitWeakLinks(fst, A, graph, n_labels, tol);

    border = iftObjectBorders(fst->label, A, false, false);
    iftWriteImageByExt(border, "newborder.nii.gz");
//    iftWriteImageByExt(fst->label, "newmask.nii.gz");
    iftDestroyImage(&border);

    timer *toc = iftToc();
    char *time = iftFormattedTime(iftCompTime(tic, toc));
    printf("%s\n", time);
    iftFree(time);

    iftFree(graph);
    iftDestroyImage(&img);
    iftDestroyImage(&marker);
    iftDestroyAdjRel(&A);
    iftDestroyImageForest(&fst);
}


double *iftComputeCellGraph(iftImageForest *fst, iftAdjRel *A, int n_labels)
{
    double *arc_weight = iftAlloc(n_labels * n_labels, sizeof *arc_weight);
    int *board_length = iftAlloc(n_labels * n_labels, sizeof *board_length);

    for (int p = 0; p < fst->label->n; p++)
    {
        iftVoxel u = iftGetVoxelCoord(fst->label, p);
        int lp = fst->label->val[p];
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(fst->label, v)) {
                int q = iftGetVoxelIndex(fst->label, v);
                int lq = fst->label->val[q];
                if (lp != lq) {
                    arc_weight[lp * n_labels + lq] += fst->img->val[p] + fst->img->val[q];
                    board_length[lp * n_labels + lq] += 1;
                }
            }
        }
    }

    for (int i = 0; i < n_labels * n_labels; i++) {
        if (board_length[i] > 0) {
            arc_weight[i] /= (2 * board_length[i]);
        }
    }

    iftFree(board_length);

    return arc_weight;
}


iftImage *iftCellGraphToImage(iftImageForest *fst, iftAdjRel *A, double *graph, int n_labels)
{
    iftImage *out = iftCreateImage(fst->img->xsize, fst->img->ysize, fst->img->zsize);

    for (int p = 0; p < fst->label->n; p++)
    {
        iftVoxel u = iftGetVoxelCoord(fst->label, p);
        int lp = fst->label->val[p];
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(fst->label, v)) {
                int q = iftGetVoxelIndex(fst->label, v);
                int lq = fst->label->val[q];
                if (lp != lq) {
                    out->val[p] = graph[lp * n_labels + lq];
                }
            }
        }
    }

    return out;
}


void iftSplitWeakLinks(iftImageForest *fst, iftAdjRel *A, double *graph, int n_labels, float tol)
{
    if (tol < 0.0f && tol > 1.0f) {
        iftError("Tolerance must be between (0, 1)", "iftSplitWeakLinks");
    }

    double *node_mean = iftAlloc(n_labels, sizeof *node_mean);
    int *node_labels = iftAlloc(n_labels, sizeof *node_labels);

#pragma omp parallel for
    for (int i = 0; i < n_labels; i++) {
        int n = 0;
        for (int j = 0; j < n_labels; j++) {
            int idx = i * n_labels + j;
            if (graph[idx] > IFT_EPSILON) {
                n++;
                node_mean[i] += graph[idx];
            }
        }
        if (n == 0) {
            node_mean[i] /= n;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n_labels; i++) {
        for (int j = 0; j < n_labels; j++) {
            int idx = i * n_labels + j;
            if (graph[idx] > IFT_EPSILON && graph[idx] > node_mean[i] * tol) {
                graph[idx] = 0.0;
                graph[j * n_labels + i] = 0.0;
            }
        }
    }

    iftFree(node_mean);

    for (int i = 0; i < n_labels; i++) {
        node_labels[i] = i;
    }

    /* relabeling */
    for (int p = 0; p < fst->label->n; p++)
    {
        iftVoxel u = iftGetVoxelCoord(fst->label, p);
        int lp = fst->label->val[p];
        int new_lp = node_labels[lp];
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(fst->label, v)) {
                int q = iftGetVoxelIndex(fst->label, v);
                int lq = fst->label->val[q];
                int new_lq = node_labels[lq];
                if (new_lp != new_lq && graph[lp * n_labels + lq] < 0.001) {
                    node_labels[new_lp] = node_labels[new_lq];
                }
            }
        }
    }

    for (int p = 0; p < fst->label->n; p++) {
        int lb = fst->label->val[p];
        fst->label->val[p] = node_labels[lb];
    }

    iftImage *aux = iftRelabelRegions(fst->label, A);
    iftDestroyImage(&fst->label);
    fst->label = aux;
}