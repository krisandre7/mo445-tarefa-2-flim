#include "ift.h"
#include "iftDynamicList.h"
#include "iftDynamicForest.h"

void iftDiffDynamicIFT(iftDynamicForest *forest) {
    iftAdjRel *A     = forest->A;
    double    *cost  = forest->cost;
    iftImage  *label = forest->label;
    iftImage  *pred  = forest->pred;
    iftImage  *root  = forest->root;
    iftMImage *mimg  = forest->mimg;
    iftDHeap  *heap  = forest->heap;

    /* Image Foresting Transform */
    while (!iftEmptyDHeap(heap))
    {
        int p      = iftRemoveDHeap(heap);
        iftVoxel u = iftMGetVoxelCoord(mimg, p);

        if (cost[p] != 0) {/* it is not a seed voxel */
	  // iftUpdateDynamicTree IFT_ADD
                iftInsertDynamicList(forest, p);
        }

        for (int i = 1; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v))
            {
                int q = iftMGetVoxelIndex(mimg, v);

                if (heap->color[q] != IFT_BLACK) {
		  // iftDynamicArcWeight
                    double arc_weight = iftDistDynamicList(forest, p, q);

                    double tmp = iftMax(cost[p], arc_weight);

                    if (tmp < cost[q]) {
                        if (heap->color[q] == IFT_GRAY)
                        {
                            iftRemoveDHeapElem(heap, q);
                        }
			// iftUpdateDynamicTree IFT_SUB
                        iftRemoveDynamicList(forest, q);
                        pred->val[q]     = p;
                        root->val[q]     = root->val[p];
                        label->val[q]    = label->val[p];
                        cost[q]          = tmp;
                        iftInsertDHeap(heap, q);
                    } else if (pred->val[q] == p && label->val[p] != label->val[q]) {
                        iftDynamicSubtreeRemoval(forest, q);
                    }
                }
            }
        }
    }
    iftResetDHeap(heap);

}

void iftInsertLabeledSetElems(iftLabeledSet **s1, iftLabeledSet *s2) {
    iftLabeledSet *M = s2;
    while (M) {
        iftInsertLabeledSet(s1, M->elem, M->label);
        M = M->next;
    }
}

void iftRemoveLabeledSetElems(iftLabeledSet **s1, iftSet *s2) {
    iftSet *M = s2;
    while (M) {
        iftRemoveLabeledSetElem(s1, M->elem);
        M = M->next;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        iftError("CSV file path required.", "main");
    }

    char *csv_file = argv[1];

    iftCSV *csv = iftReadCSV(csv_file, ',');

    if (csv->ncols != 3)
        iftError("CSV file must have 3 columns.", "main");

    char *img_path   = csv->data[0][0];
    char *method_str = csv->data[0][1];
    char *method_arg = csv->data[0][2];

    iftImage   *img = iftReadImageByExt(img_path);
    iftMImage *mimg = iftImageToMImage(img, GRAY_CSPACE);
    iftAdjRel    *A = iftCircular(1.733);

    iftDynamicForest *forest = iftCreateDynamicForest(mimg, A);

    iftLabeledSet *total_seeds = NULL;

    FILE *f_output = fopen("output.csv","w");
    if (!f_output)
        iftError("Failed to open file","main");
    fprintf(f_output, "kth IFT execution, differential, from scratch\n");
    iftMakeDir("labels");
    iftMakeDir("errors");

    int k = 0;
    for (int r = 1; r < csv->nrows; r++) {
        char *command = csv->data[r][0];

        // ADDITION
        if (strcmp(command, "A") == 0) {
            char *seeds_path = csv->data[r][1];
            iftLabeledSet *seeds = iftReadSeeds(img, seeds_path);
            iftDiffDynamicSeedsAddition(forest, seeds);
            iftInsertLabeledSetElems(&total_seeds, seeds);
            iftDestroyLabeledSet(&seeds);
            // REMOVAL
        } else if (strcmp(command, "R") == 0) {
            char *seeds_path = csv->data[r][1];
            iftLabeledSet *seeds = iftReadSeeds(img, seeds_path);
            iftSet *removal_set  = iftLabeledSetElemsToSet(seeds);
            iftDiffDynamicSeedsRemoval(forest, removal_set);
            iftRemoveLabeledSetElems(&total_seeds, removal_set);
            iftDestroySet(&removal_set);
            iftDestroyLabeledSet(&seeds);
            // EXECUTION
        } else if (strcmp(command, "E") == 0) {
            k++;
            timer *t1 = iftTic(), *t2;
            iftDiffDynamicIFT(forest);
            t2 = iftToc();
            float exec1 = iftCompTime(t1, t2);
            char out_path[255];
            sprintf(out_path, "labels/label%d.png", k);
            iftWriteImageByExt(forest->label, out_path);

            iftDynamicForest *fst2 = iftCreateDynamicForest(mimg, A);
            iftDiffDynamicSeedsAddition(fst2, total_seeds);
            t1 = iftTic();
            iftDiffDynamicIFT(fst2);
            t2 = iftToc();
            float exec2 = iftCompTime(t1, t2);

            #pragma omp parallel for
            for (int i = 0; i < fst2->label->n; i++) {
                if (fst2->label->val[i] != forest->label->val[i]) {
                    fst2->label->val[i] = 255;
                } else {
                    fst2->label->val[i] = 0;
                }
            }

            sprintf(out_path, "errors/error%d.png", k);
            iftWriteImageByExt(fst2->label, out_path);
            iftDestroyDynamicForest(&fst2);

            fprintf(f_output, "Execution %d,%f,%f\n", k, exec1, exec2);
        } else {
            iftError("Invalid execution option.", "main");
        }
    }

    iftDestroyDynamicForest(&forest);
    iftDestroyAdjRel(&A);
    iftDestroyMImage(&mimg);
    iftDestroyImage(&img);
    iftDestroyCSV(&csv);
    fclose(f_output);
    return 0;
}
