#include "ift.h"

#define MAXGRAD 1000

typedef struct _arcweights
{
	float **val;
	int     n;
} ArcWeights;

ArcWeights *CreateArcWeights(int nlabels)
{
	ArcWeights *oindex = (ArcWeights *)calloc(1, sizeof(ArcWeights));
	int i;

	oindex->val = (float **) calloc(nlabels, sizeof(float *));
	for(i = 0; i < nlabels; i++)
		oindex->val[i] = (float *)calloc(nlabels, sizeof(float));

	oindex->n = nlabels;
	return(oindex);
}

void DestroyArcWeights(ArcWeights **oindex)
{
	int i;

	if((*oindex) != NULL)
	{
		for(i = 0; i < (*oindex)->n; i++)
			free((*oindex)->val[i]);
		free((*oindex)->val);
		free((*oindex));
		*oindex = NULL;
	}
}

void PrintArcWeights(ArcWeights *oindex)
{
	int i, j;

	printf("\n");

	for(i = 0; i < oindex->n; i++)
	{
		for(j = 0; j < oindex->n; j++)
			printf("%5.2f ", oindex->val[i][j]);
		printf("\n");
	}
}

ArcWeights *OverlappingIndex(Subgraph *sg)
{
	int   i, j, k;
	float weight, tot;
	ArcWeights *oindex;

	oindex = CreateArcWeights(sg->nlabels);

	for(i = 0; i < sg->nnodes; i++)
	{
		for(j = 0; (j < sg->nnodes); j++)
		{
			k = sg->ordered_list_of_nodes[j];
			weight = opf_ArcWeight(sg->node[k].feat, sg->node[i].feat, sg->nfeats);

			if(weight <= sg->node[k].radius)
			{
				oindex->val[sg->node[i].label][sg->node[k].label]++;
			}
		}
	}

	// Normalize the overlapping index

	for(i = 0; i < sg->nlabels; i++)
	{
		tot = 0;
		for(j = 0; j < sg->nlabels; j++)
			tot += oindex->val[i][j];
		for(j = 0; j < sg->nlabels; j++)
			oindex->val[i][j] /= tot;
	}

	return(oindex);
}


int FindSubgraphRoot(Subgraph *sg, int i)
{
	if(sg->node[i].root == i)
		return(i);
	else
		return(sg->node[i].root = FindSubgraphRoot(sg, sg->node[i].root));
}

void MergeOverlapClusters(Subgraph *sg, ArcWeights *oindex)
{
	int i, j;

	printf("initial number of clusters %d\n", sg->nlabels);

	for(i = 0; i < sg->nnodes; i++)
	{
		for(j = 0; j < sg->nnodes; j++)
		{
			if(sg->node[i].label != sg->node[j].label)
				if(oindex->val[sg->node[i].label][sg->node[j].label] >= 0.07)
				{
					sg->node[i].root = FindSubgraphRoot(sg, j);
				}
		}
	}

	for(i = 0; i < sg->nnodes; i++)
		sg->node[i].root = FindSubgraphRoot(sg, i);

	j = 0;
	for(i = 0; i < sg->nnodes; i++)
		if(i == sg->node[i].root)
		{
			sg->node[sg->node[i].root].label = j;
			j++;
		}
	sg->nlabels = j;

	for(i = 0; i < sg->nnodes; i++)
		sg->node[i].label = sg->node[sg->node[i].root].label;

	printf("final number of clusters %d\n", sg->nlabels);

}

char VerifyForestIntegrity(Subgraph *sg)
{
	int p, q;

	for(p = 0; p < sg->nnodes; p++)
	{
		if(sg->node[sg->node[p].root].pred != NIL)
			return(0);
		if(sg->node[sg->node[p].root].root != sg->node[p].root)
			return(0);
		q = p;
		while(q != NIL)
		{
			if(sg->node[q].label != sg->node[sg->node[q].root].label)
				return(0);
			q = sg->node[q].pred;
		}
	}
	return(1);
}

Image *TreatOutliers(Image *label)
{
	Image *flabel = ift_CopyImage(label);
	int p, q, r, s, i, n = label->ncols * label->nrows;
	Pixel u, v;
	AdjRel *A = Circular(1.0);
	int *FIFO, first = 0, last = 0;
	int *color;
	Set *S = NULL;

	FIFO  = AllocIntArray(n);
	color = AllocIntArray(n);

	for(p = 0; p < n; p++)
	{
		if(flabel->val[p] == 0)
		{
			FIFO[last] = p;
			InsertSet(&S, p);
			last++;
			while(first != last)
			{
				q = FIFO[first];
				color[q] = BLACK;
				u.x = q % label->ncols;
				u.y = q / label->ncols;
				first++;
				for(i = 1; i < A->n; i++)
				{
					v.x = u.x + A->dx[i];
					v.y = u.y + A->dy[i];
					r = v.x + label->tbrow[v.y];
					if(ValidPixel(label, v.x, v.y))
					{
						if(color[r] == WHITE)
						{
							if(label->val[r] == 0)
							{
								FIFO[last] = r;
								color[r] = GRAY;
								InsertSet(&S, r);
								last++;
							}
							else
							{
								flabel->val[p] = label->val[r];
								first = last = 0;
								break;
							}
						}
					}
				}
			}
			while(S != NULL)
			{
				s = RemoveSet(&S);
				color[s] = WHITE;
			}
		}
	}
	DestroyAdjRel(&A);
	free(FIFO);
	free(color);

	return(flabel);
}

void DrawLabeledRegionsInPlace(CImage *cimg, Image *label)
{
	int x, y, k, p, q, u, v;
	AdjRel *A;
	Image *img = cimg->C[0];

	A = Circular(1.0);

	for(p = 0; p < img->ncols * img->nrows; p++)
	{
		x = p % img->ncols;
		y = p / img->ncols;
		for(k = 1; k < A->n; k++)
		{
			u = x + A->dx[k];
			v = y + A->dy[k];
			if(ValidPixel(img, u, v))
			{
				q = u + img->tbrow[v];
				if(label->val[p] != label->val[q])
				{
					cimg->C[0]->val[p] = 255;
					cimg->C[1]->val[p] = 0;
					cimg->C[2]->val[p] = 0;
					break;
				}
			}
		}
	}

	DestroyAdjRel(&A);
}

CImage *CGaussianFilter(CImage *cimg1)
{
	AdjRel *A = Circular(3.0);
	Kernel *K = GaussianKernel(A, 0.5);
	CImage *cimg2 = (CImage *)calloc(1, sizeof(CImage));

	cimg2->C[0] = LinearFilter2(cimg1->C[0], K);
	cimg2->C[1] = LinearFilter2(cimg1->C[1], K);
	cimg2->C[2] = LinearFilter2(cimg1->C[2], K);

	DestroyAdjRel(&A);
	DestroyKernel(&K);

	return(cimg2);
}

Image *DualWaterGrayMask(Image *img, int H, Image *mask, Image *cost, Image *pred, Image *root, AdjRel *A)
{
	Image *label = NULL;
	GQueue *Q = NULL;
	int i, p, q, tmp, n, r = 1;
	Pixel u, v;

	n     = img->ncols * img->nrows;

	label = CreateImage(img->ncols, img->nrows);
	Q     = CreateGQueue(MaximumValue(img) + 1, n, cost->val);
	SetRemovalPolicy(Q, MAXVALUE);
	Image *marker = CTVolumeOpen(img, H);
	for(p = 0; p < n; p++)
	{
		pred->val[p] = NIL;
		// only pixels within the mask
		// shall have roots
		root->val[p] = NIL;

		if(mask->val[p])
		{
		    root->val[p] = p;
			cost->val[p] = MAX(marker->val[p] - 1, 0);

			InsertGQueue(&Q, p);
		}
	}
	DestroyImage(&marker);

	while(!EmptyGQueue(Q))
	{
		p = RemoveGQueue(Q);
		if(label->val[p] == 0)
		{
			cost->val[p] = img->val[p];
			label->val[p] = r;
			r++;
		}
		u.x = p % img->ncols;
		u.y = p / img->ncols;
		for(i = 1; i < A->n; i++)
		{
			v.x = u.x + A->dx[i];
			v.y = u.y + A->dy[i];
			if(ValidPixel(img, v.x, v.y))
			{
				q = v.x + img->tbrow[v.y];
				if(mask->val[p])
				{
					if(cost->val[q] < cost->val[p])
					{
						tmp = MIN(cost->val[p], img->val[q]);
						if(tmp > cost->val[q])
						{
							UpdateGQueue(&Q, q, tmp);
							label->val[q] = label->val[p];
							pred->val[q] = p;
							root->val[q] = root->val[p];
						}
					}
				}
			}
		}
	}

	DestroyGQueue(&Q);
//  DestroyImage(&cost);

	return(label);
}


DImage *DFeatureGradient(Features *f, float radius)
{
	real    dist, gx, gy;
	int     i, j, p, q, n = f->ncols * f->nrows;
	Pixel   u, v;
	AdjRel *A = Circular(radius);
	real   *md = AllocRealArray(A->n);

//    Image* grad = CreateImage(f->ncols, f->nrows);
	DImage *dgrad = CreateDImage(f->ncols, f->nrows);

	for(i = 1; i < A->n; i++)
		md[i] = sqrt(A->dx[i] * A->dx[i] + A->dy[i] * A->dy[i]);


	for(p = 0; p < n; p++)
	{
		u.x = p % f->ncols;
		u.y = p / f->ncols;

		gx = gy = 0.0;
		float max_mag = -FLT_MAX, mag;
		for(j = 0; j < f->nfeats; j++)
		{
			gx = gy = 0.0;
			for(i = 1; i < A->n; i++)
			{
				v.x = u.x + A->dx[i];
				v.y = u.y + A->dy[i];
				if((v.x >= 0 && v.x < f->ncols) && (v.y >= 0 && v.y < f->nrows))
				{
					q    = v.x + v.y * f->ncols;
					dist = (float)(f->elem[q].feat[j] - f->elem[p].feat[j]);
					gx  += dist * A->dx[i] / md[i];
					gy  += dist * A->dy[i] / md[i];
				}
			}
			mag = sqrt(gx * gx + gy * gy);

			if(mag > max_mag)
			{
				dgrad->val[p] = (double)mag;
				max_mag = mag;
			}
		}
	}


	free(md);
	DestroyAdjRel(&A);

	return(dgrad);
}


int main(int argc, char **argv)
{
	timer *t1 = NULL, *t2 = NULL;
	Image    *img = NULL, *label = NULL, *flabel = NULL, *alabel=NULL, *dist = NULL, *cluster = NULL, *pred = NULL, *root = NULL, *cost = NULL;
	DImage *grad = NULL;
	Subgraph *sg = NULL;
	Features *f = NULL;
	CImage *cimg2 = NULL, *cimg1 = NULL;
	char ext[10], *pos;
	ArcWeights *oindex = NULL;
	AdjRel *A = NULL;
	int     *nelems, i, n, imax;
	float   *I, Imax;

	opf_ArcWeight = opf_EuclDist;

	/*--------------------------------------------------------*/

	void *trash = malloc(1);
	struct mallinfo info;
	int MemDinInicial, MemDinFinal;
	free(trash);
	info = mallinfo();
	MemDinInicial = info.uordblks;

	/*--------------------------------------------------------*/

	if(argc != 2)
		Error("Usage must be: separe-graos <image.ppm>", "main");

	pos = strrchr(argv[1], '.') + 1;
	sscanf(pos, "%s", ext);

	t1 = Tic();

	// Read image, get samples and compute image features

	if(strcmp(ext, "ppm") == 0)
	{
		cimg1 = ReadCImage(argv[1]);
		cimg2 = CImageRGBtoYCbCr(cimg1);
		//    WriteImage(cimg2->C[0],"Y.pgm");
		//  WriteImage(cimg2->C[1],"Cb.pgm");
		// WriteImage(cimg2->C[2],"Cr.pgm");
		f     = CreateFeatures(cimg2->C[0]->ncols, cimg2->C[0]->nrows, 2);
		n     = cimg2->C[0]->ncols * cimg2->C[0]->nrows;
		f->Imax = MAX(MaximumValue(cimg2->C[1]), MaximumValue(cimg2->C[2]));
		for(i = 0; i < n; i++)
		{
			f->elem[i].feat[0] = (float)cimg2->C[1]->val[i] / f->Imax;
			f->elem[i].feat[1] = (float)cimg2->C[2]->val[i] / f->Imax;
		}
		img   = ift_CopyImage(cimg2->C[0]);
		sg    = RandomSampl(img, 200);
	}
	else
	{
		printf("Invalid image format: %s\n", ext);
		exit(-1);
	}

	// Execute segmentation by clustering

	SetSubgraphFeatures(sg, f);
	opf_BestkMinCut(sg, 20, 30);
	opf_OPFClustering(sg);

	oindex = OverlappingIndex(sg);
	PrintArcWeights(oindex);
	MergeOverlapClusters(sg, oindex);

	label   = ImageClassKnnGraph(sg, f); // it propagates label+1
	cluster = TreatOutliers(label);

	// Compute average of Cr in each region

	I = AllocFloatArray(sg->nlabels);
	nelems = AllocIntArray(sg->nlabels);

	for(i = 0; i < sg->nnodes; i++)
	{
		I[sg->node[i].label] += sg->node[i].feat[1];
		nelems[sg->node[i].label]++;
	}
	for(i = 0; i < sg->nlabels; i++)
	{
		I[i] /= nelems[i];
	}

	// Select region with highest Cr

	Imax = 0;
	imax = -1;
	for(i = 0; i < sg->nlabels; i++)
	{
		if(Imax < I[i])
		{
			Imax = I[i];
			imax = i;
		}
	}
	DestroyImage(&label);
	label = Threshold(cluster, imax + 1, imax + 1); // imax+1 is due to the
	// label+1 propagation
	// in ClassifyKnnGraph

	free(nelems);
	free(I);

	// EDT

	//  flabel = CloseHoles(label);
	A      = Circular(5.0);
	flabel = CloseRec(label, A);
	DestroyAdjRel(&A);

	A      = Circular(1.5);

	alabel = FastAreaOpen(flabel,2);
	dist   = DistTrans(alabel, A, 0);

	DestroyImage(&label);

	DestroyAdjRel(&A);
	A      = Circular(5.0);

	grad = DFeatureGradient(f, 1.5);

	// Combining the distance transform with the gradient
	double max_grad = MaximumDImageValue(grad);
	int max_dist = MaximumValue(dist);
	Image *comb_dist = CreateImage(dist->ncols, dist->nrows);
	for(i = 0; i < grad->ncols * grad->nrows; i++)
	{
		if(alabel->val[i] > 0)
			comb_dist->val[i] = (int)(((double)(dist->val[i])) + (max_grad - grad->val[i]) * max_dist) / 2;
	}

	cost  = CreateImage(img->ncols, img->nrows);
	pred  = CreateImage(img->ncols, img->nrows);
	root  = CreateImage(img->ncols, img->nrows);

	label  = DualWaterGrayMask(comb_dist, 5000, alabel, cost, pred, root, A);

//	for(i = 0; i < n; i++) // remove internal holes
//	{
//		if(cluster->val[i] != (imax + 1))
//		{
//			label->val[i] = 0;
//			// updating root and predecessor accordingly
//            root->val[i] = i;
//            pred->val[i] = NIL;
//		}
//	}

	t2 = Toc();

	DrawLabeledRegionsInPlace(cimg1, label);

	WriteCImage(cimg1, "result.ppm");
	WriteImage(label, "label.pgm");
	Image *igrad = ConvertDImage2Image(grad);
	WriteImage(igrad, "grad.pgm");
	WriteImage(root, "root.pgm");
	WriteImage(comb_dist, "dist.pgm");

	DestroyAdjRel(&A);
	DestroySubgraph(&sg);
	DestroyCImage(&cimg1);
	DestroyImage(&img);
	DestroyImage(&cluster);
	DestroyImage(&label);
	DestroyImage(&pred);
	DestroyImage(&igrad);
	DestroyImage(&cost);
	DestroyImage(&root);
	DestroyImage(&comb_dist);
	DestroyDImage(&grad);
	DestroyImage(&flabel);
	DestroyImage(&alabel);
	DestroyCImage(&cimg2);
	DestroyFeatures(&f);
	DestroyImage(&dist);
	DestroyArcWeights(&oindex);

	fprintf(stdout, "separe-graos in %f ms\n", CTime(t1, t2));

	/* ---------------------------------------------------------- */

	info = mallinfo();
	MemDinFinal = info.uordblks;
	if(MemDinInicial != MemDinFinal)
		printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
					 MemDinInicial, MemDinFinal);

	return(0);
}
