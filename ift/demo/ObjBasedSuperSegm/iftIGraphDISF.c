#include "ift.h"
// PROTOTYPES
//=========================================================
void usage();
iftImage *iftDISFSegmentation(iftImage *orig_img, iftImage *mask_img, int n0, int nf);
void iftIGraphDISF(iftIGraph *igraph, iftSet *seeds, int nf);
int *iftIGraphKRelevantSeedsByContrast(iftIGraph *igraph, int k, int *seed_index, float **trees_feats, bool **trees_adj, int num_trees);

// MAIN
//=========================================================
int main(int argc, char *argv[])
{
    int n0, nf;
    iftImage *orig_img, *mask_img, *label_img;
    
    if(argc != 5 && argc != 6) usage();

    orig_img = iftReadImageByExt(argv[1]);
    n0 = atoi(argv[2]);
    nf = atoi(argv[3]);

    if(argc == 6) mask_img = iftReadImageByExt(argv[5]);
    else mask_img = NULL;

    label_img = iftDISFSegmentation(orig_img, mask_img, n0, nf);

    iftWriteImageByExt(label_img, argv[4]);

    iftDestroyImage(&orig_img);
    if(mask_img != NULL) iftDestroyImage(&mask_img);
    iftDestroyImage(&label_img);

    return EXIT_SUCCESS;
}

// DEFINITIONS
//=========================================================
void usage()
{
    printf("Usage: iftIGraphDISF [1] ... [4] {5} \n");
    printf("\t[1] - Input image (.scn,.png,.jpg,.pgm,...)\n");
    printf("\t[2] - Initial number of seeds (N0 > 1)\n");
    printf("\t[3] - Final number of superpixels (1 < Nf << N0)\n");
    printf("\t[4] - Output label image (.scn,.png,.jpg,.pgm,...)\n");
    printf("\t{5} - Mask image (.scn,.png,.jpg,.pgm,...)\n");
    iftError("Too many/few parameters!","main");
}

iftImage *iftDISFSegmentation(iftImage *orig_img, iftImage *mask_img, int n0, int nf)
{
    bool exists_mask;
    iftImage *label_img, *seed_img;
    iftMImage *orig_mimg;
    iftIGraph *igraph;
    iftSet *seeds;
    iftAdjRel *A;

    if (iftIs3DImage(orig_img)) A = iftSpheric(1.0);
    else A = iftCircular(1.0);

    if(iftIsColorImage(orig_img))
        orig_mimg = iftImageToMImage(orig_img, LABNorm_CSPACE);
    else orig_mimg = iftImageToMImage(orig_img, GRAY_CSPACE);

    if(mask_img == NULL) 
    {
        exists_mask = false;
        mask_img = iftSelectImageDomain(orig_mimg->xsize, orig_mimg->ysize, orig_mimg->zsize);
    }
    else exists_mask = true;
    
    igraph = iftExplicitIGraph(orig_mimg, mask_img, NULL, A);
    iftDestroyAdjRel(&A);
    
    seeds = NULL;
    seed_img = iftGridSampling(orig_mimg,mask_img, n0);
    iftDestroyMImage(&orig_mimg);
    if(!exists_mask) { iftDestroyImage(&mask_img); mask_img = NULL;}

    for(int i = 0; i < orig_img->n; ++i)
    {
        if(seed_img->val[i] != 0) iftInsertSet(&seeds,i);
    }
    iftDestroyImage(&seed_img);

    iftIGraphDISF(igraph, seeds, nf);

    label_img = iftIGraphLabel(igraph);
    iftDestroyIGraph(&igraph);
    iftDestroySet(&seeds);

    return label_img;
}

void iftIGraphDISF(iftIGraph *igraph, iftSet *seeds, int nf)
{
    int n0, num_trees, iter, seed_count, num_maintain;
    int *seed_index;
    float *mean_tree_tmp; // Speeding purposes
    double *pvalue;
    iftDHeap *Q;

    mean_tree_tmp = (float*)malloc(igraph->nfeats * sizeof(float));
    pvalue = (double*)malloc(igraph->nnodes * sizeof(double));
    n0 = num_trees = iftSetSize(seeds);
    seed_index = (int*)malloc(num_trees * sizeof(int));
    Q = iftCreateDHeap(igraph->nnodes, pvalue);

    seed_count = 0;
    for(iftSet *ptr = seeds; ptr != NULL; ptr = ptr->next)
        seed_index[seed_count++] = igraph->index->val[ptr->elem];

    // DISF
    //=======================================================================//
    iter = 0;
    num_maintain = num_trees;
    do
    {
        int *trees_sizes;
        bool **are_trees_adj;
        float **trees_feats;
        
        are_trees_adj = (bool**)malloc(num_trees * sizeof(bool*));
        trees_sizes = (int*)malloc(num_trees * sizeof(int));
        trees_feats = (float**)malloc(num_trees * sizeof(float*));

        // Initializing IFT auxiliary data 
        //===================================================================//
        for(int t = 0; t < igraph->nnodes; ++t)
        {
            int img_index;
            
            img_index = igraph->node[t].voxel;
            pvalue[t] = igraph->pvalue[img_index] = IFT_INFINITY_DBL;
            igraph->pred[img_index] = igraph->root[img_index] = IFT_NIL;
            igraph->label[img_index] = 0;
        }
        
        for(int i = 0; i < num_trees; ++i)
        {
            int img_index;
            are_trees_adj[i] = (bool*)malloc(num_trees * sizeof(bool));
            trees_feats[i] = (float*)malloc(igraph->nfeats * sizeof(float));

            trees_sizes[i] = 0;
            for(int j = 0; j < num_trees; ++j) are_trees_adj[i][j] = false;
            for(int j = 0; j < igraph->nfeats; ++j) trees_feats[i][j] = 0.0;
            
            img_index = igraph->node[seed_index[i]].voxel;
            pvalue[seed_index[i]] = igraph->pvalue[img_index] = 0.0;
            igraph->pred[img_index] = IFT_NIL;
            igraph->root[img_index] = img_index;
            igraph->label[img_index] = i + 1;

            iftInsertDHeap(Q, seed_index[i]);
        }
        
        // IFT
        //===================================================================//
        while(!iftEmptyDHeap(Q))
        {
            int p_node_index, p_img_index, p_label;
            iftVoxel p_voxel;

            p_node_index = iftRemoveDHeap(Q);
            p_img_index = igraph->node[p_node_index].voxel;
            p_label = igraph->label[p_img_index];
            
            trees_sizes[p_label - 1]++;
            for(int f = 0; f < igraph->nfeats; ++f)
            {
                trees_feats[p_label - 1][f] += igraph->feat[p_img_index][f];
                mean_tree_tmp[f] = trees_feats[p_label - 1][f]/(double)trees_sizes[p_label - 1];
            }
            
            switch(igraph->type)
            {
                case IMPLICIT:
                    p_voxel = iftGetVoxelCoord(igraph->index, p_img_index);

                    for(int i = 1; i < igraph->A->n; ++i)
                    {
                        iftVoxel q_voxel;

                        q_voxel = iftGetAdjacentVoxel(igraph->A, p_voxel, i);

                        if(iftValidVoxel(igraph->index, q_voxel))
                        {
                            int q_img_index, q_node_index;

                            q_img_index = iftGetVoxelIndex(igraph->index, q_voxel);
                            q_node_index = igraph->index->val[q_img_index];

                            if(q_node_index != IFT_NIL)
                            { 
                                int q_label;
                                q_label = igraph->label[q_img_index];

                                if(Q->color[q_node_index] != IFT_BLACK)
                                {
                                    double path_cost, arc_cost;

                                    arc_cost = (double)iftFeatDistance(mean_tree_tmp, igraph->feat[q_img_index], igraph->nfeats);
                                    path_cost = iftMax(pvalue[p_node_index], arc_cost);

                                    if(path_cost < pvalue[q_node_index])
                                    {
                                        pvalue[q_node_index] = path_cost;
                                        igraph->root[q_img_index] = igraph->root[p_img_index];
                                        igraph->label[q_img_index] = igraph->label[p_img_index];
                                        igraph->pred[q_img_index] = p_img_index;

                                        if(Q->color[q_node_index] == IFT_GRAY)
                                            iftGoUpDHeap(Q, Q->pos[q_node_index]);
                                        else
                                            iftInsertDHeap(Q, q_node_index);
                                    }
                                }
                                else if(q_label != p_label && !are_trees_adj[p_label - 1][q_label - 1])
                                {
                                    are_trees_adj[p_label - 1][q_label - 1] = true;
                                    are_trees_adj[q_label - 1][p_label - 1] = true;
                                }
                            }
                        }
                    }
                    break;
                case EXPLICIT:
                    for(iftSet *ptr = igraph->node[p_node_index].adj; ptr != NULL; ptr = ptr->next)
                    {
                        int q_node_index;

                        q_node_index = ptr->elem;

                        if(q_node_index != IFT_NIL)
                        { 
                            int q_img_index, q_label;

                            q_img_index = igraph->node[q_node_index].voxel;
                            q_label = igraph->label[q_img_index];

                            if(Q->color[q_node_index] != IFT_BLACK)
                            {
                                double path_cost, arc_cost;

                                arc_cost = (double)iftFeatDistance(mean_tree_tmp, igraph->feat[q_img_index], igraph->nfeats);
                                path_cost = iftMax(pvalue[p_node_index], arc_cost);

                                if(path_cost < pvalue[q_node_index])
                                {
                                    pvalue[q_node_index] = path_cost;
                                    igraph->root[q_img_index] = igraph->root[p_img_index];
                                    igraph->label[q_img_index] = igraph->label[p_img_index];
                                    igraph->pred[q_img_index] = p_img_index;

                                    if(Q->color[q_node_index] == IFT_GRAY)
                                        iftGoUpDHeap(Q, Q->pos[q_node_index]);
                                    else
                                        iftInsertDHeap(Q, q_node_index);
                                }
                            }
                            else if(q_label != p_label && !are_trees_adj[p_label - 1][q_label - 1])
                            {
                                are_trees_adj[p_label - 1][q_label - 1] = true;
                                are_trees_adj[q_label - 1][p_label - 1] = true;
                            }
                        }
                    }
                    break;
                case COMPLETE:
                    iftError("Not implemented for Complete IGraphs!", "iftIGraphDISF");
                    break;
                default:
                    iftError("Unknown IGraph type: %d!", "iftIGraphDISF", igraph->type);
                    break;
            }
        }

        // Remove irrelevants
        //===================================================================//
        int *rel_seeds_index;

        num_maintain = iftMax(n0 * exp(-(iter + 1)), nf);

        // rel_seeds_index = iftIGraphKRelevantSeedsByContrast(igraph,  num_maintain, seed_index, trees_feats, trees_adj, num_trees);
        for(int i = 0; i < num_trees; ++i)
        {
            free(are_trees_adj[i]);
            free(trees_feats[i]);
        }
        free(trees_sizes);
        free(are_trees_adj);
        free(trees_feats);
//        num_trees = num_maintain;
//        free(seed_index);
//        seed_index = rel_seeds_index;
        ++iter;

        iftResetDHeap(Q);
    } while(num_maintain > nf);
    
    free(mean_tree_tmp);
    free(pvalue);
    free(seed_index);
    iftDestroyDHeap(&Q);
}

int *iftIGraphKRelevantSeedsByContrast(iftIGraph *igraph, int k, int *seed_index, float **trees_feats, bool **trees_adj, int num_trees);

