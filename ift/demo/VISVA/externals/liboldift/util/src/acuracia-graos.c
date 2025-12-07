#include "ift.h"

typedef struct SubGrainInfo
{
    int lb_id;
    int area;
    struct SubGrainInfo *next;
}SubGrainInfo;


void InsertSubGrainInfo(SubGrainInfo **grain, int lb_id)
{
    if(grain == NULL) return;

    SubGrainInfo *subgrain =  (SubGrainInfo*)calloc(1,sizeof(SubGrainInfo));
    subgrain->next = NULL;
    subgrain->area = 1;
    subgrain->lb_id = lb_id;

    if(*grain != NULL)
    {
        subgrain->next = *grain;
    }

    *grain = subgrain;

}

// returns a pointer to the struct representing a given subgrain if it exists or NULL
SubGrainInfo* FindSubGrain(SubGrainInfo *grain, int lb_id)
{
    SubGrainInfo *aux = grain;

    while(aux != NULL)
    {
        if(aux->lb_id == lb_id) return aux;
        aux = aux->next;
    }

    return NULL;
}

void DestroySubGrainInfo(SubGrainInfo **grain)
{
    if(grain == NULL || *grain == NULL) return;

    SubGrainInfo *aux = *grain, *tmp;

    while(aux != NULL)
    {
        tmp = aux->next;
        free(aux);
        aux = tmp;
    }

    *grain = NULL;
}

// Creates a new label map by attributing the groundtruth labels
// according to the pixel's root
Image *RemapGroundtruthLabels(Image *gt, Image *root)
{
	int p;
	Image *remapped = CreateImage(gt->ncols, gt->nrows);

    // pixels labeled as background (root->val[p] == NIL)
    // should keep their label when remapping, since they
    // are not considered during automatic segmentation
	for(p = 0; p < gt->ncols * gt->nrows; p++)
        remapped->val[p] = (root->val[p] != NIL)? gt->val[root->val[p]] : 0;

	return remapped;
}

// Compute accuracy given a groundtruth and a remapped grain image label
float GrainAccuracy(Image *label, Image *gt)
{
	float Acc = 0.0f, **error_matrix = NULL, error = 0.0f;
	int i, *nclass = NULL, nlabels = 0;
    int max_gt_label = MaximumValue(gt);
    int n = label->ncols*label->nrows;

	error_matrix = (float **)calloc(max_gt_label+1, sizeof(float *));
	for(i = 0; i <= max_gt_label; i++)
		error_matrix[i] = (float *)calloc(2, sizeof(float));

	nclass = AllocIntArray(max_gt_label+1);

	for(i = 0; i < n; i++)
	{
		nclass[gt->val[i]]++;
	}

	for(i = 0; i < n; i++)
	{
		if(gt->val[i] != label->val[i])
		{
			error_matrix[gt->val[i]][1]++;
			error_matrix[label->val[i]][0]++;
		}
	}

	for(i = 0; i <= max_gt_label; i++)
	{
		if(nclass[i] != 0)
		{
			error_matrix[i][1] /= (float)nclass[i];
			error_matrix[i][0] /= (float)(n - nclass[i]);
			nlabels++;
		}
	}

	for(i = 0; i <= max_gt_label; i++)
	{
		if(nclass[i] != 0)
			error += (error_matrix[i][0] + error_matrix[i][1]);
	}

	Acc = 1.0 - (error / (2.0 * nlabels));

	for(i = 0; i <= max_gt_label; i++)
		free(error_matrix[i]);
	free(error_matrix);
	free(nclass);


	return(Acc);
}

SubGrainInfo** CountSubGrainAreas(Image *label, Image *gt, Image *remapped, int **gt_grain_area)
{
    int i;
    int max_gt_label = MaximumValue(gt);
    SubGrainInfo **grains = (SubGrainInfo**)calloc(max_gt_label+1,sizeof(SubGrainInfo*));
    *gt_grain_area = AllocIntArray(max_gt_label+1);

    for(i = 0; i <= max_gt_label; i++)
        grains[i] = NULL;

    for(i = 0; i < gt->ncols*gt->nrows; i++)
    {
        // if the current pixel's remapped label
        // is equal to that of the groundtruth then
        // it is part of the current grain and its
        // segmentation label should be evaluated
        // to determine if the pixel belongs to a
        // subdivision of the grain
        if(gt->val[i] == remapped->val[i])
        {
            SubGrainInfo *subgrain = FindSubGrain(grains[gt->val[i]], label->val[i]);
            if(subgrain != NULL)
            {
                subgrain->area++;
            }
            else
            {
                InsertSubGrainInfo(&(grains[gt->val[i]]), label->val[i]);
            }
        }
        (*gt_grain_area)[gt->val[i]]++;
    }

    return grains;
}


void GrainSubdivisionError(Image *label, Image *gt, Image *remapped,float *weighted_subdivision_error, float *subdivided_grains_area_perc)
{
    int i, *gt_grain_area, max_gt_label = MaximumValue(gt), subdivided_grains_area=0, grains_area=0;
    SubGrainInfo **grains = CountSubGrainAreas(label, gt, remapped, &gt_grain_area);

    *weighted_subdivision_error = 0.0;

    // only grains (not background, i.e., gt->val[i] > 0) are considered
    for(i = 1; i <= max_gt_label; i++)
    {
        SubGrainInfo *aux = grains[i];
        int max_subgrain_area = NIL;
        while(aux != NULL)
        {
            if(aux->lb_id != 0)
            {
                max_subgrain_area = MAX(aux->area, max_subgrain_area);
            }
            aux = aux->next;
        }

        if(max_subgrain_area != NIL && max_subgrain_area != gt_grain_area[i])
        {

            // Subdivision error weighted by the area of each incorrectly segmented grain.
            // The formula below is a simplification of (1.0 - (float)max_subgrain_area/(float)gt_grain_area[i])*(float)gt_grain_area[i]
            *weighted_subdivision_error += (float)gt_grain_area[i] - (float)max_subgrain_area;
            subdivided_grains_area += gt_grain_area[i];
        }
        grains_area += gt_grain_area[i];
    }

    // subdivision error weighted by the area of each incorrectly segmented grain
    *weighted_subdivision_error = *weighted_subdivision_error/subdivided_grains_area*100;
    // percentage of subdivided grain area with respect to the total occupied area by grains
    *subdivided_grains_area_perc = (float)subdivided_grains_area/(float)grains_area*100;

    for(i = 0; i <= MaximumValue(gt); i++)
        DestroySubGrainInfo(&(grains[i]));

    free(grains);
    free(gt_grain_area);

}

int main(int argc, char **argv)
{
	timer *t1 = NULL, *t2 = NULL;
	Image    *label = NULL, *gt = NULL, *root = NULL, *remapped = NULL;

	/*--------------------------------------------------------*/

	void *trash = malloc(1);
	struct mallinfo info;
	int MemDinInicial, MemDinFinal;
	free(trash);
	info = mallinfo();
	MemDinInicial = info.uordblks;

	/*--------------------------------------------------------*/

	if(argc != 4)
		Error("Usage must be: acuracia-graos <label.pgm> <gt.pgm> <root.pgm>", "main");

	t1 = Tic();

	// Read label, gt, and root map
    label = ReadImage(argv[1]);
	gt = ReadImage(argv[2]);
	root = ReadImage(argv[3]);

	remapped = RemapGroundtruthLabels(gt, root);

    printf("Acc %f\n",GrainAccuracy(remapped,gt));

    float weighted_subdivision_error, subdivided_grains_area_perc;

    GrainSubdivisionError(label, gt, remapped, &weighted_subdivision_error, &subdivided_grains_area_perc);

    printf("Weighted Grain Subdivision Error %.2f%%\n",weighted_subdivision_error);
    printf("Percentage of Grain Area %.2f%%\n",subdivided_grains_area_perc);

	t2 = Toc();

	DestroyImage(&label);
	DestroyImage(&gt);
	DestroyImage(&root);
	DestroyImage(&remapped);

	fprintf(stdout, "%s in %f ms\n", argv[0], CTime(t1, t2));

	/* ---------------------------------------------------------- */

	info = mallinfo();
	MemDinFinal = info.uordblks;
	if(MemDinInicial != MemDinFinal)
		printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
					 MemDinInicial, MemDinFinal);

	return(0);
}
