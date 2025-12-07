#include "iftDISF.h"

//=============================================================================
// Constructors & Deconstructors
//=============================================================================

Tree *createTree(int root_index, int num_feats)
{
    Tree *tree;

    tree = (Tree*)calloc(1, sizeof(Tree));

    tree->root_index = root_index;
    tree->num_nodes = 0;
    tree->num_feats = num_feats;

    tree->sum_feat = (float*)calloc(num_feats, sizeof(float));

    return tree;
}

void freeTree(Tree **tree)
{
    if(*tree != NULL)
    {
        Tree *tmp;

        tmp = *tree;

        free(tmp->sum_feat);
        free(tmp);

        *tree = NULL;
    }
}

//=============================================================================
// Double
//=============================================================================
inline double euclDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for(int i = 0; i < num_feats; i++)
        dist += (feat1[i] - feat2[i]) * (feat1[i] - feat2[i]);
    dist = sqrtf(dist);

    return dist;
}

inline double taxicabDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for(int i = 0; i < num_feats; i++)
        dist += fabs(feat1[i] - feat2[i]);

    return dist;
}

//=============================================================================
// Float*
//=============================================================================
inline float* meanTreeFeatVector(Tree *tree)
{
    float* mean_feat;

    mean_feat = (float*)calloc(tree->num_feats, sizeof(float));

    for(int i = 0; i < tree->num_feats; i++)
        mean_feat[i] = tree->sum_feat[i]/(float)tree->num_nodes;

    return mean_feat;
}

//=============================================================================
// Image*
//=============================================================================
iftImage *runDISF(iftMImage *mimg, 
                int n_0, int n_f, 
                iftAdjRel *A,
                iftImage *mask)
{
    int num_rem_seeds, iter;
    double *cost_map;
    iftSet *seed_set;
    iftImage *label_img, *mask_img;
    iftDHeap *queue;
    int num_seeds = n_0;

    // Aux
    cost_map = (double*)calloc(mimg->n, sizeof(double));
    label_img = iftCreateImage(mimg->xsize, mimg->ysize, 1);
    queue = iftCreateDHeap(mimg->n, cost_map);

    if(mask == NULL) mask_img = iftSelectImageDomain(mimg->xsize, mimg->ysize, mimg->zsize);
    else mask_img = mask;

    // GRID SAMPLING ==============================================================
    iftImage *seed_img;

    seed_set = NULL;
    seed_img = iftGridSampling(mimg, mask_img, num_seeds);

    num_seeds = 0;
    for(int p_index = 0; p_index < mimg->n; ++p_index)
    {
        if(seed_img->val[p_index] > 0) {
            iftInsertSet(&seed_set, p_index); 
            num_seeds++;
        }
        if(mask_img->val[p_index] == 0)
          cost_map[p_index] = IFT_INFINITY_DBL_NEG;
    }
    
    iftDestroyImage(&seed_img);
    if(mask == NULL) iftDestroyImage(&mask_img);

    // DISF ========================================================================
    
    iter = 1; // At least a single iteration is performed
    do
    {
        int seed_label, num_trees, num_maintain;
        Tree **trees;
        iftSet **tree_adj;
        bool **are_trees_adj;
        
        trees = (Tree**)calloc(num_seeds, sizeof(Tree*));
        tree_adj = (iftSet**)calloc(num_seeds, sizeof(iftSet*));
        are_trees_adj = (bool**)calloc(num_seeds, sizeof(bool*));

        // Initialize values
        #pragma omp parallel for
        for(int i = 0; i < mimg->n; i++)
        {
            cost_map[i] = INFINITY;
            label_img->val[i] = -1;
        }

        seed_label = 0;
        for(iftSet *ptr = seed_set; ptr != NULL; ptr = ptr->next)
        {   
            int seed_index = ptr->elem;

            cost_map[seed_index] = 0;
            label_img->val[seed_index] = seed_label+1;

            trees[seed_label] = createTree(seed_index, mimg->m);
            tree_adj[seed_label] = NULL;
            are_trees_adj[seed_label] = (bool*)calloc(num_seeds, sizeof(bool));

            seed_label++;
            iftInsertDHeap(queue, seed_index);
        }

        // IFT algorithm
        while(!iftEmptyDHeap(queue))
        {
            int p_index, p_label;
            iftVoxel p_voxel;
            float *mean_feat_tree;

            p_index = iftRemoveDHeap(queue);
            p_voxel = iftMGetVoxelCoord(mimg, p_index);
            p_label = label_img->val[p_index]-1;
            
            // This node won't appear here ever again
            insertNodeInTree(mimg, p_index, &(trees[p_label]));

            mean_feat_tree = meanTreeFeatVector(trees[p_label]);

            for(int i = 1; i < A->n; i++)
            {
                iftVoxel q_voxel;

                q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);

                if(iftMValidVoxel(mimg, q_voxel))
                {
                    int q_index, q_label;

                    q_index = iftMGetVoxelIndex(mimg, q_voxel);
                    q_label = label_img->val[q_index]-1;

                    // If it wasn't inserted nor orderly removed from the queue
                    if(queue->color[q_index] != IFT_BLACK)
                    {
                        double arc_cost, path_cost;

                        arc_cost = euclDistance(mean_feat_tree, mimg->val[q_index], mimg->m);

                        path_cost = iftMax(cost_map[p_index], arc_cost);

                        if(path_cost < cost_map[q_index])
                        {
                            cost_map[q_index] = path_cost;
                            label_img->val[q_index] = p_label+1;

                            if(queue->color[q_index] == IFT_GRAY) iftGoUpDHeap(queue, queue->pos[q_index]);
                            else iftInsertDHeap(queue, q_index);
                        }
                    }
                    else if(p_label != q_label) // Their trees are adjacent
                    {
                        if(!are_trees_adj[p_label][q_label])
                        {
                            iftInsertSet(&(tree_adj[p_label]), q_label);
                            iftInsertSet(&(tree_adj[q_label]), p_label);
                            are_trees_adj[q_label][p_label] = true;
                            are_trees_adj[p_label][q_label] = true;
                        }
                    }
                }
            }

            free(mean_feat_tree);
        }

        num_maintain = iftMax(n_0 * exp(-iter), n_f);

        // Aux
        num_trees = num_seeds;
        iftDestroySet(&seed_set);

        seed_set = selectKMostRelevantSeeds(trees, tree_adj, mimg->n, num_trees, num_maintain);

        num_seeds = iftSetSize(seed_set);

        num_rem_seeds = num_trees - num_seeds;
        
        printf("iter:%d, num_seeds: %d, seed_set size: %d, num_trees: %d, num_maintain: %d \n", iter, num_seeds, iftSetSize(seed_set), num_trees, num_maintain);

        iter++;
        iftResetDHeap(queue);
        for(int i = 0; i < num_trees; ++i)
        {
            freeTree(&(trees[i]));
            iftDestroySet(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);
    } while(num_rem_seeds > 0);

    free(cost_map);
    iftDestroySet(&seed_set);
    iftDestroyDHeap(&queue);

    return label_img;
}

//=============================================================================
// IntList*
//=============================================================================

iftSet *selectKMostRelevantSeeds(Tree **trees, iftSet **tree_adj, int num_nodes, int num_trees, int num_maintain)
{
    double *tree_prio;
    iftSet *rel_seeds;
    iftDHeap *queue;

    tree_prio = (double*)calloc(num_trees, sizeof(double));
    rel_seeds = NULL;
    queue = iftCreateDHeap(num_trees, tree_prio);
    iftSetRemovalPolicyDHeap(queue, MAXVALUE);

    for(int i = 0; i < num_trees; i++)
    {
        double area_prio, grad_prio;
        float *mean_feat_i;

        area_prio = trees[i]->num_nodes/(float)num_nodes;

        grad_prio = INFINITY;
        mean_feat_i = meanTreeFeatVector(trees[i]);

        for(iftSet *ptr = tree_adj[i]; ptr != NULL; ptr = ptr->next)
        {
            int adj_tree_id;
            float *mean_feat_j;
            double dist;

            adj_tree_id = ptr->elem;
            mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);

            dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

            grad_prio = iftMin(grad_prio, dist);

            free(mean_feat_j);
        }

        tree_prio[i] = area_prio * grad_prio;

        iftInsertDHeap(queue, i);

        free(mean_feat_i);
    }

    for(int i = 0; i < num_maintain && !iftEmptyDHeap(queue); i++)
    {
        int tree_id, root_index;

        tree_id = iftRemoveDHeap(queue);
        root_index = trees[tree_id]->root_index;

        iftInsertSet(&rel_seeds, root_index);
    }

    iftDestroyDHeap(&queue); // The remaining are discarded
    free(tree_prio);

    return rel_seeds;
}

//=============================================================================
// Void
//=============================================================================
void insertNodeInTree(iftMImage *mimg, int index, Tree **tree)
{
    (*tree)->num_nodes++;

    for(int i = 0; i < mimg->m; i++)
        (*tree)->sum_feat[i] += mimg->val[index][i];
}
