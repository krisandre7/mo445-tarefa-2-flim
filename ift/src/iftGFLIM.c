#include "iftFLIM.cuh"
#include "iftFLIM.h"
#include "iftGFLIM.h"

#include "iftMImage.h"
#include "iftImage.h"
#include "ift/core/dtypes/DHeap.h"
#include "ift/core/dtypes/Set.h"
#include "iftAdjacency.h"


///////////////////////////////////////
//              PROTOTYPES
///////////////////////////////////////

/* --------------------- FLIMGraph construction ----------------------------- */
// create / insert
iftFLIMGraph *iftCreateEmptyFLIMGraph(int num_feats, int num_nodes);
void iftInsertFLIMGraphNeighborNode(unsigned long node_index, 
                                    unsigned long neighbor_index, 
                                    iftFLIMGraph *graph);
iftFLIMGraph *getAdjacencyGraphFromSuperpixels(unsigned long num_superpixels,
                                            iftAdjRel *A,
                                            iftMImage *oiginal_img, 
                                            iftImage *labeled_img);
// modify
void iftFLIMGraphChangeNodesFeatures(iftFLIMGraph *graph,
                                     int *node_indexes,
                                     double **features,
                                     int num_nodes);
void iftDeleteVertexFLIMGraph(iftFLIMGraph *graph, int node_index);
void NormalizeGraphByZScore(iftFLIMGraph *graph, float *mean, float *stdev);

// copy
iftFLIMGraph *iftCopyFLIMGraph(int num_feats, iftFLIMGraph *graph_orig);

// read from / write to disk
iftFLIMGraph *iftReadFLIMGraph(char *filename);
void iftWriteFLIMGraph(iftFLIMGraph *graph, char *filename);

// validate
bool iftIsValidFLIMGraph(iftFLIMGraph *graph);

// destroy
void iftDestroyFLIMGraph(iftFLIMGraph **graph);

/* --------------------- FLIMGraph conversion ----------------------------- */
// image to graph
void iftImageToFLIMGraph(char *image_dir, 
                    char *labels_dir, 
                    char *graph_dir, 
                    int num_init_seeds, 
                    int num_superpixels);
void iftImageToFLIMGraphs(char *image_dir, char *labels_dir, 
                    char *graph_dir, int num_init_seeds, 
                    int num_superpixels);

// graph to iftImage / iftMImage 
iftImage *iftGraphToImage(iftFLIMGraph *graph, iftImage *labels, int Imax, int band);

// graph from / to MIMG 
iftMImage *iftGraphToMImage(iftFLIMGraph *graph, iftImage *labels);
iftFLIMGraph* iftMImageToFLIMGraph(iftMImage *mimg, 
                                iftFLIMGraph *graph_ref);
//(write in disk)
void iftGraphToMIMG(char *graph_dir, char *labels_dir, char *output_dir);


// graph from/to iftMatrix
iftMatrix *iftGraphToFeatureMatrix(iftFLIMGraph *graph,
                                int num_adj,
                                float *fill_band_with,
                                bool most_similar);
iftFLIMGraph *iftMatrixToFLIMGraph(iftMatrix *matrix, 
                                iftFLIMGraph *graph_ref);



// graph to image activations (one grayscale image per node feature)
iftImage **iftGetActivations(iftFLIMGraph *graph, iftImage *labels);

/* --------------------- FLIMGraph training ----------------------------- */

//training 
void iftFLIMGraphLearnLayer(char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftFLIMArch *arch, char *output_dir);
void FLIMGraphTrain(int layer, char *param_dir, char *orig_dir, char *seeds_dir);

// extract features
void FLIMGraphExtractFeaturesFromLayerPerBatch(iftFileSet *fs_graphs, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                               char *param_dir, int layer_index, char *feat_dir, char *object_dir);
void FLIMGraphExtractFeaturesFromLayerPerImage(iftFileSet *fs_graphs, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                               char *param_dir, int layer_index, char *feat_dir, char *object_dir);
void FLIMGraphExtractFeaturesPerImage(iftFileSet *fs_graphs, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                               char *param_dir, char *feat_dir, char *object_dir);

void ExtractFeaturesFromLayerForTraining(char *param_dir, char *orig_dir, int layer, char *seeds_dir, char *labels_dir, bool write_img);
void FLIMGraphExtractFeaturesFromLayer(int layer, char *param_dir, char *orig_dir, char *seeds_dir, char *labels_dir, bool write_img);
void iftFLIMGraphExtractFeaturesFromLayer(char *orig_dir, char *graph_list, iftFLIMArch *arch, char *param_dir, int layer_index,
                                          char *feat_dir, char *object_dir, int device);

/* --------------------- FLIMGraph seeds (user drawn markers) ----------------------------- */

// iftLabeledSet *iftReadSeedsGraph(char *filename); in iftSeeds.c
//void iftWriteSeedsGraph(iftLabeledSet* seed, const char *_filename, ...); in iftSeeds.c

// Convert image seeds file to graph seeds file
void seedsPixelToGraph(int num_nodes, char *labels_file, char *seeds_file, char *output_file);
void seedsPixelToGraphDir(char *graph_dir, char *labels_dir, char *seeds_dir, char *output_dir);

// label seeds as their connected components
iftLabeledSet *LabelMarkersGraph(iftFLIMGraph *graph, iftLabeledSet *S);

// get the mean and standard deviation from the seeds files
void StatisticsFromAllSeedsGraph(iftFileSet *fs_seeds, char *dir, float *mean, float *stdev, float stdev_factor);

/* --------------------- FLIMGraph pooling functions ----------------------------- */

iftMImage* iftFLIMGraphAtrousAveragePooling(iftMImage *conv, iftFLIMGraph *graph,
                                      int width, int height, int depth,
                                      int atrous_factor, int stride,
                                      bool most_similar);
void iftFLIMGraphAtrousAveragePooling_fromMatrix(iftMatrix *conv, iftFLIMGraph *graph,
                                      int width, int height, int depth,
                                      int atrous_factor, int stride,
                                      bool most_similar);

iftMImage* iftFLIMGraphAtrousMaxPooling(iftMImage *conv,
                                  iftFLIMGraph *graph,
                                  int width,
                                  int height,
                                  int depth,
                                  int atrous_factor,
                                  int stride,
                                  bool most_similar);

void iftFLIMGraphAtrousMaxPooling_fromMatrix(iftMatrix *conv,
                                  iftFLIMGraph *graph,
                                  int width,
                                  int height,
                                  int depth,
                                  int atrous_factor,
                                  int stride,
                                  bool most_similar);

/* --------------------- Others ----------------------------- */
void writeActivationsImg(char *source, char *destination, char *ext, char *labels_dir);
void writeCSVFiles(char *csv_file, char *files_dir);
void selectKernelsManual(char *param_dir, char *orig_dir, int layer, char *seeds_dir, char *string_kernels, char *labels_dir, bool write_img);
void getNeighborsFLIMGraph(iftFLIMGraph *graph, iftSet *roots,
                           int k, double *dissimilarity,
                           double *feature_reference,
                           iftDHeap *heap,
                           bool most_similar);
void getKNeigborsFLIMGraph(int node_index, iftFLIMGraph *graph, int k,
                           double *dissimilarity, iftSet *roots,
                           iftDHeap *auxiliaryHeap, iftDHeap *heap,
                           int dilation,
                           bool most_similar);
iftDataSet *ComputeSeedDataSetGraph(iftFLIMGraph *graph,
                                    iftLabeledSet *S,
                                    int *kernel_size,
                                    int nsamples,
                                    bool most_similar);
iftMatrix *LearnKernelBankGraph(iftFLIMGraph *graph, iftLabeledSet *M,
                                int *kernel_size, int nsamples,
                                int nkernels_per_image, int nkernels_per_marker,
                                bool most_similar);

iftFLIMGraph **
FLIMGraphConvolutionalLayer(iftFLIMGraph **graph, int ngraphs,
                            iftImage **mask, iftFLIMArch *arch,
                            int layer, int layer_index,
                            int atrous_factor, char *param_dir);

iftMatrix *ConsensusKernelbankGraph(iftFileSet *fs_seeds, char *inputdata_dir, int noutput_channels, float stdev_factor);
void FLIMExtractFeaturesFromLayer(int layer, char *param_dir, char *orig_dir, char *seeds_dir, bool write_img);
void iftFLIMTrain(int layer, char *param_dir, char *orig_dir, char *seeds_dir);

///////////////////////////////////////


/* --------------------- DISF functions ----------------------------- */

typedef struct
{
    int root_index, num_nodes, num_feats;
    float *sum_feat;
} Tree;

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

void insertNodeInTree(iftMImage *mimg, int index, Tree **tree)
{
    (*tree)->num_nodes++;

    for(int i = 0; i < mimg->m; i++)
        (*tree)->sum_feat[i] += mimg->val[index][i];
}

inline double euclDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for(int i = 0; i < num_feats; i++)
        dist += (feat1[i] - feat2[i]) * (feat1[i] - feat2[i]);
    dist = sqrtf(dist);

    return dist;
}

inline float* meanTreeFeatVector(Tree *tree)
{
    float* mean_feat;

    mean_feat = (float*)calloc(tree->num_feats, sizeof(float));

    for(int i = 0; i < tree->num_feats; i++)
        mean_feat[i] = tree->sum_feat[i]/(float)tree->num_nodes;

    return mean_feat;
}

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


/* --------------------- FLIMGraph construction ----------------------------- */

/**
 * @brief Create a new graph with num_nodes nodes and num_feats features, but no edges.
 * @param num_feats Number of features
 * @param num_nodes Number of nodes
 * @return iftFLIMGraph*
 * @note num_feats and num_nodes must be greater than 0
 * @note The graph has no edges. To add edges, use iftInsertFLIMGraphNeighborNode
 * @note The graph have features zero. To change the features' values, use iftFLIMGraphChangeNodesFeatures
 * @author Isabela B. Barcelos
 * @date fev, 2024
 */
iftFLIMGraph *iftCreateEmptyFLIMGraph(int num_feats, int num_nodes)
{

    assert(num_feats > 0);
    assert(num_nodes > 0);

    iftFLIMGraph *graph = (iftFLIMGraph *)calloc(1, sizeof(iftFLIMGraph));
    graph->num_nodes = num_nodes;
    graph->num_feats = num_feats;

    graph->nodes = (iftFLIMGraphNode *)calloc(graph->num_nodes, sizeof(iftFLIMGraphNode));

    for (unsigned long spx_index = 0; spx_index < graph->num_nodes; spx_index++)
    {
        graph->nodes[spx_index].numNeighbors = 0;
        graph->nodes[spx_index].neighbors_list_head = NULL;
        graph->nodes[spx_index].neighbors_list_tail = NULL;
        graph->nodes[spx_index].numPixels = 0;
        graph->nodes[spx_index].index = spx_index;
        graph->nodes[spx_index].feats = (double *)malloc(num_feats * sizeof(double));
        for (unsigned long feat_index = 0; feat_index < num_feats; feat_index++)
            graph->nodes[spx_index].feats[feat_index] = 0.0;
    }
    return graph;
}

/**
 * @brief Insert an edge in the graph linking node_index with neighbor_index
 * @param node_index Node index
 * @param neighbor_index Neighbor index
 * @param graph Graph
 * @note node_index and neighbor_index must be valid indexes for the graph's nodes
 * @note For undirected edge, also call iftInsertFLIMGraphNeighborNode switching node_index and neighbor_index
 * @warning The graph must be valid. Use iftIsValidFLIMGraph to check if the graph is valid
 * @warning This function does not check if the edge already exists in the graph
 * @author Isabela B. Barcelos
 * @date fev, 2024
 */
void iftInsertFLIMGraphNeighborNode(unsigned long node_index, unsigned long neighbor_index, iftFLIMGraph *graph)
{
    assert(node_index >= 0);
    assert(neighbor_index >= 0);

    iftFLIMGraphNodeList *tmp = (iftFLIMGraphNodeList *)malloc(sizeof(iftFLIMGraphNodeList));
    tmp->next = NULL;
    tmp->node = &(graph->nodes[neighbor_index]);
    if (graph->nodes[node_index].numNeighbors == 0)
    {
        graph->nodes[node_index].neighbors_list_head = tmp;
        graph->nodes[node_index].neighbors_list_tail = tmp;
        graph->nodes[node_index].neighbors_list_tail->next = NULL;
        graph->nodes[node_index].numNeighbors++;
        return;
    }

    graph->nodes[node_index].neighbors_list_tail->next = tmp;
    graph->nodes[node_index].neighbors_list_tail = tmp;
    graph->nodes[node_index].neighbors_list_tail->next = NULL;
    graph->nodes[node_index].numNeighbors++;
}

/**
 * @brief Calculate a FLIMGraph from the original image and its superpixel labels map.
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param num_superpixels   The number of superpixels in 'labeled_img'.
 * @param A                 An adjacency relation between pixels.
 * @param oiginal_img       The original image.
 * @param labeled_img       A map of superpixel labels.
 * @return The FLIMGraph.
 */
iftFLIMGraph *getAdjacencyGraphFromSuperpixels(
    unsigned long num_superpixels,
    iftAdjRel *A,
    iftMImage *oiginal_img, iftImage *labeled_img)
{

    iftFLIMGraph *graph = (iftFLIMGraph *)calloc(1, sizeof(iftFLIMGraph));
    graph->num_nodes = num_superpixels;
    graph->num_feats = oiginal_img->m;

    graph->nodes = (iftFLIMGraphNode *)calloc(num_superpixels, sizeof(iftFLIMGraphNode));
    bool **is_adj = (bool **)calloc(num_superpixels, sizeof(bool *));

    for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
    {
        graph->nodes[spx_index].feats = (double *)calloc(graph->num_feats, sizeof(double));
        graph->nodes[spx_index].numPixels = 0;
        graph->nodes[spx_index].neighbors_list_head = NULL;
        graph->nodes[spx_index].neighbors_list_tail = NULL;
        graph->nodes[spx_index].numNeighbors = 0;
        graph->nodes[spx_index].index = spx_index;
        is_adj[spx_index] = (bool *)calloc(spx_index + 1, sizeof(bool)); // maior índice armazena os vizinhos (evita preencher toda a tabela)
    }

    // for each pixel
    for (int p_index = 0; p_index < labeled_img->n; p_index++)
    {
        unsigned long label = (unsigned long)(labeled_img->val[p_index] - 1); // get its label
        graph->nodes[label].numPixels++;

        // increase feature
        for (unsigned long band_index = 0; band_index < oiginal_img->m; band_index++)
            graph->nodes[label].feats[band_index] += oiginal_img->val[p_index][band_index];
        
        // look at neighbors
        iftVoxel p_voxel = iftMGetVoxelCoord(oiginal_img, p_index);
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);
            if (iftMValidVoxel(oiginal_img, q_voxel))
            {
                unsigned long q_label = (unsigned long)(labeled_img->val[iftMGetVoxelIndex(oiginal_img, q_voxel)] - 1);
                if (label > q_label && !is_adj[label][q_label])
                {
                    is_adj[label][q_label] = true;

                    iftInsertFLIMGraphNeighborNode(label, q_label, graph);
                    iftInsertFLIMGraphNeighborNode(q_label, label, graph);
                }
            }
        }
    }

    // for each node, compute its mean color
    for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
    {
        for (unsigned long band_index = 0; band_index < graph->num_feats; band_index++)
            graph->nodes[spx_index].feats[band_index] /= (double)(graph->nodes[spx_index].numPixels);
        
        free(is_adj[spx_index]);
    }

    free(is_adj);
    return graph;
}

/**
 * @brief Change the nodes' weights in the graph
 * @param graph Graph
 * @param node_indexes Nodes indexes
 * @param features The new nodes' weights for the graph's nodes
 * @param num_nodes Number of nodes in node_indexes and number of rows in features
 * @note The number of features in features must be equal to the number of features in the graph
 * @note num_nodes must be lower of equal the number of graph nodes.
 * @note The nodes' indexes in node_indexes must be valid indexes for the graph's nodes
 * @author Isabela B. Barcelos
 * @date fev, 2024
 */
void iftFLIMGraphChangeNodesFeatures(iftFLIMGraph *graph,
                                     int *node_indexes,
                                     double **features,
                                     int num_nodes)
{
    assert(graph->num_nodes >= num_nodes);
    for (int i = 0; i < num_nodes; i++)
    {
        int vertex = node_indexes[i];

        assert(vertex < graph->num_nodes && vertex >= 0);
        for (int j = 0; j < graph->num_feats; j++)
        {
            graph->nodes[vertex].feats[j] = features[i][j];
        }
    }
}

/**
 * @brief Delete a node graph along with its edges
 * @param graph Graph
 * @param node_index Node index
 * @note The node_index must be a valid index for the graph's nodes
*/
void iftDeleteVertexFLIMGraph(iftFLIMGraph *graph, int node_index)
{
    assert(node_index >= 0 && node_index < graph->num_nodes);
    assert(iftIsValidFLIMGraph(graph));

    iftFLIMGraphNode *node = &graph->nodes[node_index];

    iftFLIMGraphNodeList *node_list = graph->nodes[node_index].neighbors_list_head;
    while (node_list != NULL)
    {
        // delete the node from the neighbor's list
        iftFLIMGraphNodeList *tmp = node_list->node->neighbors_list_head;
        iftFLIMGraphNodeList *prev = NULL;
        graph->nodes[node_list->node->index].numNeighbors--;
        while (tmp != NULL)
        {
            if (tmp->node->index == node_index)
            {
                if (prev == NULL)
                    tmp->node->neighbors_list_head = tmp->next;
                else
                    prev->next = tmp->next;
                tmp->next = NULL;
                tmp->node = NULL;
                iftFree(tmp);
                break;
            }
            prev = tmp;
            tmp = tmp->next;
        }

        // delete its neigbors list
        tmp = node_list;
        node_list = node_list->next;
        tmp->next = NULL;
        tmp->node = NULL;
        iftFree(tmp);
    }

    node->neighbors_list_head = NULL;
    node->neighbors_list_tail = NULL;
    iftFree(node->feats);
    node->feats = NULL;

    for (unsigned long i = node_index; i < graph->num_nodes - 1; i++)
    {
        graph->nodes[i] = graph->nodes[i + 1];
        // graph->nodes[i].index--;
    }

    graph->num_nodes--;
}

/**
 * @brief Normalize the node features based on mean and standard deviation values
 * @param graph Graph
 * @param mean Mean values
 * @param stdev Standard deviation values
 * @note The number of features in mean and stdev must be equal to the number of features in the graph
*/
void NormalizeGraphByZScore(iftFLIMGraph *graph, float *mean, float *stdev)
{
    int num_feats = graph->num_feats;

    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < num_feats; b++)
        {
            graph->nodes[p].feats[b] = ((float)graph->nodes[p].feats[b] - mean[b]) / stdev[b];
        }
    }
}

/**
 * @brief Create a new empty graph (graph_orig == NULL) or a copy of graph_orig
 * @param num_feats Number of features
 * @param graph_orig Original graph
 * @return iftFLIMGraph*
 * @note graph_orig must be NULL or a valid graph
 * @note If graph_orig is not NULL, return a copy of graph_orig. Oterwise, return a graph with no nodes and num_feats features.
 * @note num_feats must be greater than 0
 * @author Isabela B. Barcelos
 * @date abr, 2023
 */
iftFLIMGraph *iftCopyFLIMGraph(int num_feats, iftFLIMGraph *graph_orig)
{
    iftFLIMGraph *graph;

    assert(num_feats > 0);

    if (graph_orig == NULL)
    {
        graph = (iftFLIMGraph*)malloc(sizeof(iftFLIMGraph));
        graph->num_feats = num_feats;
        graph->num_nodes = 0;
        return graph;
    }

    assert(iftIsValidFLIMGraph(graph_orig));

    graph = iftCreateEmptyFLIMGraph(num_feats, graph_orig->num_nodes);

    for (unsigned long node_index = 0; node_index < graph->num_nodes; node_index++)
    {
        for (unsigned long b = 0; b < graph_orig->num_feats; b++)
            graph->nodes[node_index].feats[b] = graph_orig->nodes[node_index].feats[b];
        for (unsigned long b = graph_orig->num_feats; b < graph->num_feats; b++)
            graph->nodes[node_index].feats[b] = 0;

        graph->nodes[node_index].numPixels = graph_orig->nodes[node_index].numPixels;
        graph->nodes[node_index].index = node_index;

        iftFLIMGraphNodeList *neighbors = graph_orig->nodes[node_index].neighbors_list_head;
        while (neighbors != NULL)
        {
            iftInsertFLIMGraphNeighborNode((unsigned long)node_index, neighbors->node->index, graph);
            iftInsertFLIMGraphNeighborNode(neighbors->node->index, (unsigned long)node_index, graph);
            neighbors = neighbors->next;
        }
    }
    return graph;
}


/**
 * @brief Read a graph from a json file
 * @param filename File name
 * @return iftFLIMGraph*
 * @note The file must be a valid json file
 * @author Isabela B. Barcelos
 * @date abr, 2023
 */
iftFLIMGraph *iftReadFLIMGraph(char *filename)
{
    iftDict *graph_dict = iftReadJson(filename);

    unsigned long num_nodes = strtoul(iftGetStrValFromDict("num_nodes", graph_dict), NULL, 0);
    unsigned long num_feats = strtoul(iftGetStrValFromDict("num_feats", graph_dict), NULL, 0);
    iftFLIMGraph *graph = (iftFLIMGraph *)calloc(1, sizeof(iftFLIMGraph));
    graph->num_nodes = num_nodes;
    graph->num_feats = num_feats;
    graph->nodes = (iftFLIMGraphNode *)calloc(graph->num_nodes, sizeof(iftFLIMGraphNode));
    for (unsigned long spx_index = 0; spx_index < graph->num_nodes; spx_index++)
    {
        graph->nodes[spx_index].numNeighbors = 0;
        graph->nodes[spx_index].neighbors_list_head = NULL;
        graph->nodes[spx_index].neighbors_list_tail = NULL;
        graph->nodes[spx_index].numPixels = 0;
        graph->nodes[spx_index].index = spx_index;
        graph->nodes[spx_index].feats = (double *)malloc(num_feats * sizeof(double));
        for (unsigned long feat_index = 0; feat_index < num_feats; feat_index++)
            graph->nodes[spx_index].feats[feat_index] = 0.0;
    }
    for (unsigned long l = 0; l < graph->num_nodes; l++)
    {
        char name[50];
        sprintf(name, "node%lu", l);
        iftDict *node_dict = iftGetDictFromDict(name, graph_dict);

        unsigned long node_index = strtoul(iftGetStrValFromDict("node_index", node_dict), NULL, 0);
        unsigned long num_pixels = strtoul(iftGetStrValFromDict("num_pixels", node_dict), NULL, 0);
        unsigned long num_neighbors = strtoul(iftGetStrValFromDict("num_neighbors", node_dict), NULL, 0);

        graph->nodes[node_index].numPixels = num_pixels;

        iftDblArray *feats = iftGetDblArrayFromDict("feats", node_dict);
        for (unsigned long feat_index = 0; feat_index < graph->num_feats; feat_index++)
        {
            graph->nodes[node_index].feats[feat_index] = feats->val[feat_index];
        }

        iftStrArray *input = iftGetStrArrayFromDict("neighbors", node_dict);

        unsigned long added_neighb = 0;
        for (unsigned long neighb = 0; neighb < num_neighbors; neighb++)
        {
            unsigned long neighb_val = strtoul(input->val[neighb], NULL, 0);
            if (node_index > neighb_val)
            {
                added_neighb++;
                iftInsertFLIMGraphNeighborNode(node_index, neighb_val, graph);
                iftInsertFLIMGraphNeighborNode(neighb_val, node_index, graph);
            }
        }

        // iftDestroyStrArray(&input);
        // iftDestroyDict(&node_dict);
    }
    iftDestroyDict(&graph_dict);
    iftIsValidFLIMGraph(graph);
    return (graph);
}

/**
 * @brief Write a graph to a json file
 * @param graph Graph
 * @param filename File path
*/
void iftWriteFLIMGraph(iftFLIMGraph *graph, char *filename)
{
    assert(iftIsValidFLIMGraph(graph));

    FILE *fp = fopen(filename, "wb");

    if (fp == NULL)
        iftError("Cannot open file: \"%s\"", "iftWriteFLIMGraph", filename);

    fprintf(fp, "{\n");

    for (int l = 0; l < graph->num_nodes; l++)
    {
        for (int i = 0; i < 4; i++)
            fprintf(fp, " ");
        fprintf(fp, "\"node%d\": {\n", l);

        for (int i = 0; i < 8; i++)
            fprintf(fp, " ");
        fprintf(fp, "\"feats\": [\n");
        for (int i = 0; i < graph->num_feats; i++)
        {
            for (int j = 0; j < 12; j++)
                fprintf(fp, " ");
            fprintf(fp, "%lf", graph->nodes[l].feats[i]);
            if (i != graph->num_feats - 1)
                fprintf(fp, ",");
            fprintf(fp, "\n");
        }
        for (int i = 0; i < 8; i++)
            fprintf(fp, " ");
        fprintf(fp, "],\n");

        for (int i = 0; i < 8; i++)
            fprintf(fp, " ");
        fprintf(fp, "\"neighbors\": [\n");
        iftFLIMGraphNodeList *node_list_ptr = graph->nodes[l].neighbors_list_head;
        for (unsigned long neighb = 0; neighb < graph->nodes[l].numNeighbors; neighb++)
        {
            for (int i = 0; i < 12; i++)
                fprintf(fp, " ");
            fprintf(fp, "\"%lu\"", node_list_ptr->node->index);
            node_list_ptr = node_list_ptr->next;
            if (neighb != graph->nodes[l].numNeighbors - 1)
                fprintf(fp, ",");
            fprintf(fp, "\n");
        }
        for (int i = 0; i < 8; i++)
            fprintf(fp, " ");
        fprintf(fp, "],\n");

        for (int i = 0; i < 8; i++)
            fprintf(fp, " ");
        fprintf(fp, "\"node_index\": \"%lu\",\n", graph->nodes[l].index);
        for (int i = 0; i < 8; i++)
            fprintf(fp, " ");
        fprintf(fp, "\"num_neighbors\": \"%lu\",\n", graph->nodes[l].numNeighbors);
        for (int i = 0; i < 8; i++)
            fprintf(fp, " ");
        fprintf(fp, "\"num_pixels\": \"%lu\"\n", graph->nodes[l].numPixels);

        for (int i = 0; i < 4; i++)
            fprintf(fp, " ");
        fprintf(fp, "},\n");
    }
    for (int i = 0; i < 4; i++)
        fprintf(fp, " ");
    fprintf(fp, "\"num_feats\": \"%lu\",\n", graph->num_feats);
    for (int i = 0; i < 4; i++)
        fprintf(fp, " ");
    fprintf(fp, "\"num_nodes\": \"%lu\"\n", graph->num_nodes);
    fprintf(fp, "}");

    fclose(fp);
}

/**
 * @brief Check whether a graph is valid
 */
bool iftIsValidFLIMGraph(iftFLIMGraph *graph)
{
    if (graph == NULL)
    {
        iftError("Graph is NULL", "iftIsValidFLIMGraph");
    }
    if (graph->num_nodes == 0)
    {
        iftError("Graph has no nodes", "iftIsValidFLIMGraph");
    }
    if (graph->nodes == NULL)
    {
        iftError("Graph has no nodes", "iftIsValidFLIMGraph");
    }
    return true;
}

/**
 * @brief Destroy a graph
 */
void iftDestroyFLIMGraph(iftFLIMGraph **graph)
{
    iftFLIMGraph *aux = *graph;

    if (aux != NULL)
    {
        for (int i = 0; i < aux->num_nodes; i++)
        {
            iftFLIMGraphNodeList *node_list = aux->nodes[i].neighbors_list_head;
            aux->nodes[i].neighbors_list_head = NULL;
            aux->nodes[i].neighbors_list_tail = NULL;

            while (node_list != NULL)
            {
                iftFLIMGraphNodeList *tmp = node_list;
                node_list = node_list->next;
                tmp->next = NULL;
                tmp->node = NULL;
                iftFree(tmp);
            }
            iftFree(aux->nodes[i].feats);
        }
        free(aux->nodes);
        iftFree(aux);
        *graph = NULL;
    }
}


/* --------------------- FLIMGraph conversion ----------------------------- */

/**
 * @brief Convert an image to a FLIMGraph
 * @param image_dir image file path
 * @param labels_dir superpixel images file path
 * @param graph_dir graph file path
 * @param num_init_seeds Number of initial seeds (used in DISF)
 * @param num_superpixels Number of superpixels / nodes in the graph
 * @note The image must be in .pgm or .ppm format
*/
void iftImageToFLIMGraph(char *image_dir, 
                    char *labels_dir, 
                    char *graph_dir, 
                    int num_init_seeds, 
                    int num_superpixels)
{
    iftImage *img, *label_img;
    iftImage *mask = NULL;
    iftMImage *mimg;
    iftAdjRel *A;

    if(num_superpixels >= num_init_seeds) iftError("Number of superpixels >= initial seeds", "ImageToFLIMGraph");
    
    //-------------------------
    // compute iDISF
    img = iftReadImageByExt(image_dir);

    // Init other inputs
    if (iftIsColorImage(img)) mimg = iftImageToMImage(img, LABNorm_CSPACE);
    else mimg = iftImageToMImage(img, GRAYNorm_CSPACE);
    iftDestroyImage(&img);

    if (iftIs3DMImage(mimg)) A = iftSpheric(sqrtf(3.0));
    else A = iftCircular(sqrtf(2.0));

    label_img = runDISF(mimg, num_init_seeds, num_superpixels, A, mask);

    //--------------------
    if (labels_dir != NULL)
        iftWriteImageByExt(label_img, labels_dir);
    
    //-------------------------
    // Compute graph
    iftFLIMGraph *graph = getAdjacencyGraphFromSuperpixels(num_superpixels, A, 
                                                            mimg, label_img);
    iftDestroyAdjRel(&A);
    iftDestroyMImage(&mimg);
    iftDestroyImage(&label_img);

    if (graph_dir != NULL) iftWriteFLIMGraph(graph, graph_dir);

    iftDestroyFLIMGraph(&graph);
    //-------------------------
}

/**
 * @brief Compute graphs of the images in "image_dir" using DISF,
 * save their pixel-superpixel maps in "labels_dir",
 * and save them in "graph_dir"
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param image_dir         Directory of the original image.
 * @param graph_dir         Directory of the graph.
 * @param num_init_seeds    Initial number of seeds in DISF.
 * @param num_superpixels Number of superpixels / nodes in the graph
 * @note The image must be in .pgm or .ppm format
 * @note The labels_dir and graph_dir must be valid directories
*/
void iftImageToFLIMGraphs(char *image_dir, 
                    char *labels_dir, 
                    char *graph_dir, 
                    int num_init_seeds, 
                    int num_superpixels)
{
    iftFileSet *fs_imgs = iftLoadFileSetFromDir(image_dir, 1);

    for (int i = 0; i < fs_imgs->n; i++)
    {
        char *basename = NULL;
        char graph_file[200], ext[10], label_file[200];

        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        basename = iftFilename(fs_imgs->files[i]->path, ext);
        sprintf(label_file, "%s/%s.pgm", labels_dir, basename);
        sprintf(graph_file, "%s/%s.json", graph_dir, basename);

        iftImageToFLIMGraph(fs_imgs->files[i]->path, 
                    label_file, 
                    graph_file, 
                    num_init_seeds, 
                    num_superpixels);
    }
}

/**
 * @brief Convert a graph to a iftImage
 * @param graph Graph
 * @param labels Superpixel image
 * @param Imax Maximum intensity value
 * @param band node feature to be converted to image
*/
iftImage *iftGraphToImage(iftFLIMGraph *graph, iftImage *labels, int Imax, int band)
{
    int p, b = band;

    iftImage *img = iftCreateImage(labels->xsize, labels->ysize, labels->zsize);
    double min = IFT_INFINITY_FLT, max = IFT_INFINITY_FLT_NEG;

    if ((band < 0) || (band >= (int)graph->num_feats))
        iftError("Invalid band", "iftGraphToImage");

    for (p = 0; p < graph->num_nodes; p++)
    {
        if (graph->nodes[p].feats[b] < min)
            min = graph->nodes[p].feats[b];

        if (graph->nodes[p].feats[b] > max)
            max = graph->nodes[p].feats[b];
    }

    double final_max = 0;
    double final_min = 255;

    if (max > min)
    {
        for (p = 0; p < img->n; p++)
        {
            img->val[p] = (int)((double)(Imax) * ((graph->nodes[labels->val[p] - 1].feats[b] - min) / (max - min)));

            if (img->val[p] < final_min)
                final_min = (double)(img->val[p]);
            if (img->val[p] > final_max)
                final_max = (double)(img->val[p]);
        }
    }

    img->dx = 1;
    img->dy = 1;
    img->dz = 1;

    return img;
}

/**
 * @brief Convert a graph to a iftMImage
 * @param graph Graph
 * @param labels Superpixel image
 * @return iftMImage*
 * @note The number of bands in the mimg is equal to the number of features in the graph
*/
iftMImage *iftGraphToMImage(iftFLIMGraph *graph, iftImage *labels)
{
    int p, b;

    iftMImage *img = iftCreateMImage(labels->xsize, labels->ysize, labels->zsize, graph->num_feats);

    for (b = 0; b < img->m; b++)
    {
        for (p = 0; p < img->n; p++)
        {
            img->val[p][b] = graph->nodes[labels->val[p] - 1].feats[b];
        }
    }

    img->dx = 1;
    img->dy = 1;
    img->dz = 1;

    return img;
}


 /**
 * @brief Convert a graph (in json format) to a .mimg file using the superpixel labels map as auxiliar.
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param graph_dir    Directory with graph files (in json format).
 * @param labels_dir   Directory with the superpixel labels map.
 * @param output_dir   Directory to write the .mimg file.
 * @note The labels must be in .pgm or .ppm format
 * @note The graph must be in .json format
 */
void iftGraphToMIMG(char *graph_dir, char *labels_dir, char *output_dir)
{
    char ext[10];
    iftFileSet *fs_graphs = iftLoadFileSetFromDir(graph_dir, 1);
    iftFileSet *fs_labels = iftLoadFileSetFromDir(labels_dir, 1);

    if (fs_graphs->n > fs_labels->n)
    {
        printf("Error: number of graphs is greater than number of labels\n");
        exit(1);
    }

    sprintf(ext, "%s", iftFileExt(fs_labels->files[0]->path));

    for (int i = 0; i < fs_graphs->n; i++)
    {
        char *basename = NULL;
        char filename[200], label_file[200];

        basename = iftFilename(fs_graphs->files[i]->path, ".json");

        sprintf(label_file, "%s/%s%s", labels_dir, basename, ext);

        iftFLIMGraph *graph = iftReadFLIMGraph(fs_graphs->files[i]->path);
        iftImage *label_img = iftReadImageByExt(label_file);
        iftMImage *features = iftGraphToMImage(graph, label_img);

        sprintf(filename, "%s/%s.mimg", output_dir, basename);
        iftWriteMImage(features, filename);
        iftDestroyFLIMGraph(&graph);
        iftDestroyImage(&label_img);
    }
}


iftFLIMGraph* iftMImageToFLIMGraph(iftMImage *mimg, 
                                iftFLIMGraph *graph_ref)
{
    iftFLIMGraph *graph;

    assert(mimg->n == graph_ref->num_nodes);
    assert(iftIsValidFLIMGraph(graph_ref));

    graph = iftCreateEmptyFLIMGraph(mimg->m, graph_ref->num_nodes);

    for(int i = 0; i < mimg->n; i++)
    {
        for(int j = 0; j < mimg->m; j++)
            graph->nodes[i].feats[j] = mimg->val[i][j];
        
        graph->nodes[i].numPixels = graph_ref->nodes[i].numPixels;
        graph->nodes[i].index = i;

        iftFLIMGraphNodeList *neighbors = graph_ref->nodes[i].neighbors_list_head;
        while(neighbors != NULL)
        {
            iftInsertFLIMGraphNeighborNode((unsigned long)i, neighbors->node->index, graph);
            iftInsertFLIMGraphNeighborNode(neighbors->node->index, (unsigned long)i, graph);
            neighbors = neighbors->next;
        }
    }

    return graph;
}

/**
 * @brief Create a matrix from a graph
 * @param graph Graph
 * @param num_adj Number of neighbors to be included in the matrix
 * @param fill_band_with Fill the matrix with this value when the node has less than num_adj neighbors
 * @param most_similar If true, the most similar neighbors are included in the matrix. Otherwise, the less similar neighbors are included.
 * @return iftMatrix*
 * @note The matrix size is [num nodes][num_adj * num features]
 * @note Each matrix row contains the features of a node and its [num_adj] most similar neigbors
 * @note The similarity measure to find neighbors is euclidean distance in feature space.
*/
iftMatrix *iftGraphToFeatureMatrix(iftFLIMGraph *graph,
                                   int num_adj,
                                   float *fill_band_with,
                                   bool most_similar)
{
    // cria a matriz de saída
    iftMatrix *matrix = iftCreateMatrix(num_adj * graph->num_feats, graph->num_nodes);

    double *dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));

    // auxiliaryHeap é usado para armazenar até k vizinhos mais similares.
    // Se a quantidade de vizinhos for maior que k, remove o menos similar
    // do heap (que está no topo) usando "pop" e verifica quem é o mais similar
    // para nova inserção.
    iftDHeap *auxiliaryHeap = NULL;
    auxiliaryHeap = iftCreateDHeap(graph->num_nodes, dissimilarity);
    if (most_similar)
        auxiliaryHeap->removal_policy = MAXVALUE;

    // MinHeap armazena todos os vizinhos que serão incluídos na matriz de saída.
    // O minHeap fornece a ordem de inclusão na matriz de saída
    // (do mais similar para o menos similar).
    // A raíz do mineap deve sempre ser o vértice inicial,
    // pois possui maior similaridade consigo mesmo.
    iftDHeap *heap;
    heap = iftCreateDHeap((int)graph->num_nodes, dissimilarity);
    if (!most_similar)
        heap->removal_policy = MAXVALUE;

    iftSet *roots = NULL;

    for (int p = 0; p < graph->num_nodes; p++)
    {

        getKNeigborsFLIMGraph(p, graph, num_adj,
                              dissimilarity, roots,
                              auxiliaryHeap, heap, 1,
                              most_similar);

        // move as features dos vértices incluídos no heap para a matriz de saída
        int i = 0;

        while (!iftEmptyDHeap(heap))
        {
            int node_index = iftRemoveDHeap(heap);
            for (int b = 0; b < graph->num_feats; b++)
            {
                iftMatrixElem(matrix, b + i * graph->num_feats, p) = (float)(graph->nodes[node_index].feats[b]);
            }
            i++;
        }
        iftResetDHeap(heap);

        // caso não tenha vizinhos suficientes,
        // completa o kernel com zeros ou com o valor de fill_band_with
        float tmp[graph->num_feats];
        if (fill_band_with != NULL)
            memcpy(tmp, fill_band_with, graph->num_feats * sizeof(float));
        else
            memset(tmp, 0, graph->num_feats * sizeof(float));

        while (i < num_adj)
        {
            for (int b = 0; b < graph->num_feats; b++)
            {
                iftMatrixElem(matrix, b + i * graph->num_feats, p) = (float)(tmp[b]);
            }
            i++;
        }
    }
    iftDestroyDHeap(&auxiliaryHeap);
    iftDestroyDHeap(&heap);
    iftFree(dissimilarity);
    iftDestroySet(&roots);

    return matrix;
}

/*
    Create a graph with n vertices, each one with m features.
    The graph edges come from a reference graph
    and the node features come from a matrix with size [n][m].
*/
iftFLIMGraph *iftMatrixToFLIMGraph(iftMatrix *matrix, iftFLIMGraph *graph_ref)
{
    iftFLIMGraph *graph;

    assert(matrix->nrows == graph_ref->num_nodes);

    graph = (iftFLIMGraph *)malloc(sizeof(iftFLIMGraph));
    graph->num_feats = matrix->ncols;

    graph->num_nodes = graph_ref->num_nodes;
    graph->nodes = (iftFLIMGraphNode *)calloc(graph->num_nodes, sizeof(iftFLIMGraphNode));

    for (int node_index = 0; node_index < graph->num_nodes; node_index++)
    {
        graph->nodes[node_index].numNeighbors = 0;
        graph->nodes[node_index].numPixels = graph_ref->nodes[node_index].numPixels;
        graph->nodes[node_index].index = node_index;

        graph->nodes[node_index].neighbors_list_head = NULL;
        graph->nodes[node_index].neighbors_list_tail = NULL;

        graph->nodes[node_index].feats = (double *)malloc(matrix->ncols * sizeof(double));
        for (int b = 0; b < matrix->ncols; b++)
            graph->nodes[node_index].feats[b] = iftMatrixElem(matrix, b, node_index);

        iftFLIMGraphNodeList *neighbors = graph_ref->nodes[node_index].neighbors_list_head;
        while (neighbors != NULL)
        {
            // allocate a new neigbor for node graph->nodes[node_index]
            iftFLIMGraphNodeList *tmp = (iftFLIMGraphNodeList *)malloc(sizeof(iftFLIMGraphNodeList));

            tmp->next = NULL;
            tmp->node = &(graph->nodes[neighbors->node->index]);
            if (graph->nodes[node_index].numNeighbors == 0)
            {
                graph->nodes[node_index].neighbors_list_head = tmp;
                graph->nodes[node_index].neighbors_list_tail = tmp;
                graph->nodes[node_index].numNeighbors++;
            }
            else
            {
                graph->nodes[node_index].neighbors_list_tail->next = tmp;
                graph->nodes[node_index].neighbors_list_tail = tmp;
                graph->nodes[node_index].numNeighbors++;
            }
            neighbors = neighbors->next;
        }
    }
    iftIsValidFLIMGraph(graph);
    return graph;
}

/**
 * @brief Convert a graph to a vector of iftImage
 * @param graph Graph
 * @param labels Superpixel image
 * @return iftImage**
 * @note The number of images is equal to the number of features in the graph
*/
iftImage **iftGetActivations(iftFLIMGraph *graph, iftImage *labels)
{
    iftImage **activations = (iftImage **)malloc(graph->num_feats * sizeof(iftImage *));

    /* ranged from 0 to 1*/
    for (int i = 0; i < (int)(graph->num_feats); i++)
    {
        iftImage *band = iftGraphToImage(graph, labels, 255, i);

        if (iftIs3DImage(band))
        {
            activations[i] = iftGetXYSlice(band, band->zsize / 2);
        }
        else
        {
            activations[i] = iftCopyImage(band);
        }
        iftDestroyImage(&band);
    }

    return activations;
}



/* --------------------- FLIMGraph training ----------------------------- */

/**
 * @brief Train a FLIMGraph model.
 * @param layer   Layer to train.
 * @param param_dir       Directory to write the FLIMGraph model.
 * @param orig_dir        Directory with the input of layer 1 (a graph in a .json file).
 * @param seeds_dir       Directory with the seeds files.
 */
void FLIMGraphTrain(int layer, char *param_dir, char *orig_dir, char *seeds_dir)
{
    char arch_file[255], layer_output[255], layer_input[255], arch_layer_file[255];

    if (layer == 1)
    {
        sprintf(layer_input, "%s", orig_dir);
    }
    else
    {
        char file_kernels[255], file_mean[255], file_stdev[255];

        sprintf(file_kernels, "%s/conv%d-kernels.npy", param_dir, layer);
        sprintf(file_mean, "%s/conv%d-mean.txt", param_dir, layer);
        sprintf(file_stdev, "%s/conv%d-stdev.txt", param_dir, layer);

        if (!iftFileExists(file_kernels) && !iftFileExists(file_mean) && !iftFileExists(file_stdev))
            FLIMGraphTrain(layer - 1, param_dir, orig_dir, seeds_dir);

        sprintf(layer_input, "%s/layer%d/", param_dir, layer - 1);
    }

    sprintf(layer_output, "%s/layer%d/", param_dir, layer);

    if (iftDirExists(layer_output))
    {
        iftRemoveDir(layer_output);
        iftMakeDir(layer_output);
    }
    else
        iftMakeDir(layer_output);

    // Reading architecture
    sprintf(arch_file, "%s/arch.json", param_dir);
    sprintf(arch_layer_file, "%s/arch_layer%d.json", param_dir, layer);

    // save arch on arch_layer%d.json
    iftFLIMArch *arch = iftReadFLIMArch(arch_file);

    iftFLIMArch *save_arch = (iftFLIMArch *)calloc(1, sizeof(iftFLIMArch));
    save_arch->nlayers = 1;
    save_arch->stdev_factor = arch->stdev_factor;
    save_arch->apply_intrinsic_atrous = arch->apply_intrinsic_atrous;
    save_arch->layer = (iftFLIMLayer *)calloc(1, sizeof(iftFLIMLayer));

    for (int i = 0; i < 3; i++)
    {
        save_arch->layer[0].kernel_size[i] = arch->layer[layer - 1].kernel_size[i];
        save_arch->layer[0].dilation_rate[i] = arch->layer[layer - 1].dilation_rate[i];
        save_arch->layer[0].pool_size[i] = arch->layer[layer - 1].pool_size[i];
    }
    save_arch->layer[0].nkernels_per_image = arch->layer[layer - 1].nkernels_per_image;
    save_arch->layer[0].nkernels_per_marker = arch->layer[layer - 1].nkernels_per_marker;
    save_arch->layer[0].noutput_channels = arch->layer[layer - 1].noutput_channels;
    save_arch->layer[0].relu = arch->layer[layer - 1].relu;
    
    save_arch->layer[0].pool_type = (char *)malloc((strlen(arch->layer[layer - 1].pool_type) + 1) * sizeof(char));
    strcpy(save_arch->layer[0].pool_type, arch->layer[layer - 1].pool_type);
    save_arch->layer[0].pool_stride = arch->layer[layer - 1].pool_stride;

    iftDestroyFLIMArch(&arch);
    iftWriteFLIMArch(save_arch, arch_layer_file);

    iftFLIMGraphLearnLayer(layer_input, seeds_dir, param_dir, layer, save_arch, layer_output);

    /* Writting architecture in case it changed */
    arch = iftReadFLIMArch(arch_file);

    arch->layer[layer - 1].noutput_channels = save_arch->layer[0].noutput_channels;
    iftWriteFLIMArch(arch, arch_file);

    iftDestroyFLIMArch(&arch);
    iftDestroyFLIMArch(&save_arch);
}

/**
 * @brief Learn a layer
 * @param activ_dir directory with activation files
 * @param markers_dir directory with seeds files
 * @param param_dir directory with the model files
 * @param layer_index layer index
 * @param arch architecture
 * @param output_dir directory to write the output files
 * @note The activation files must be graphs in .json format
 * @note The seeds files must be in .txt format
 * @note The output_dir must be a valid directory
 * @note The architecture must be in .json format
*/
void iftFLIMGraphLearnLayer(char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftFLIMArch *arch, char *output_dir)
{
    /* Set input parameters */
    iftMakeDir("tmp");

    iftFileSet *fs_activ = iftLoadFileSetFromDirOrCSV(activ_dir, 1, 1); // original images (for layer 1)
    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds_graph.txt", 1);
    iftFLIMGraph **output   = NULL;
    int ninput_channels     = 0;
    int ngraphs             = fs_seeds->n;
    int atrous_factor       = 1;
    char *basename          = NULL;
    char filename[200], ext[10];

    sprintf(ext, "%s", iftFileExt(fs_activ->files[0]->path));

    /* Generate input layer */

    // read activation files and write them in the temp directory
    for (int i = 0; i < ngraphs; i++)
    {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");
        sprintf(filename, "%s/%s%s", activ_dir, basename, ext);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);
        iftIsValidFLIMGraph(graph);

        sprintf(filename, "tmp/%s.json", basename);
        iftWriteFLIMGraph(graph, filename);
        if (i == 0)
        {
            ninput_channels = graph->num_feats;
        }
        iftFree(basename);
        iftDestroyFLIMGraph(&graph);
    }
    
    // compute mean and std of the graph node seeds
    float *mean = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);

    StatisticsFromAllSeedsGraph(fs_seeds, "tmp", mean, stdev, arch->stdev_factor);
    
    printf("\nLayer %d\n", layer_index);
    fflush(stdout);

    /* For a specific layer do */
    /* Learn and save marker-based normalization parameters and kernels from each training image */
    /*
    Para cada arquivo (grafo):
        - lê o grafo
        - lê o arquivo de sementes como iftLabeledSet: S
        - Rotula conjuntos conexos de sementes (conforme o grafo)
        - Normaliza todas as sementes conforme sua média e desvio padrão (de todas as sementes do grafo)
        - Escreve o grafo normalizado em "tmp/%s-norm.json"
        - calcula os kernels
        - escreve o kernel bank em "tmp/%s-kernels.npy"
    */
    
    for (int i = 0; i < ngraphs; i++)
    {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");
        sprintf(filename, "tmp/%s.json", basename);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);
        iftIsValidFLIMGraph(graph);

        printf("Processing file %s: %d of %d files\r", basename, i + 1, ngraphs);
        fflush(stdout);

        /* Learn mean and standard deviation */

        iftLabeledSet *S = iftReadSeedsGraph(fs_seeds->files[i]->path);
        iftLabeledSet *M = LabelMarkersGraph(graph, S);
        iftDestroyLabeledSet(&S);

        /* Apply marker-based image normalization */
        NormalizeGraphByZScore(graph, mean, stdev);
        sprintf(filename, "tmp/%s-norm.json", basename);
        iftWriteFLIMGraph(graph, filename);

        /* Learn and save kernel bank */

        int nsamples            = iftLabeledSetSize(M); // número de conjuntos conexos de sementes
        iftMatrix *kernelbank   = LearnKernelBankGraph(graph, M,
                                          arch->layer[0].kernel_size,
                                          nsamples,
                                          arch->layer[0].nkernels_per_image,
                                          arch->layer[0].nkernels_per_marker, true);

        iftDestroyFLIMGraph(&graph);
        iftDestroyLabeledSet(&M);
        sprintf(filename, "tmp/%s-kernels.npy", basename);
        iftWriteMatrix(kernelbank, filename);
        iftDestroyMatrix(&kernelbank);
        iftFree(basename);
    }

    /* Create a consensus layer (i.e., merge kernels) and save final kernel bank for layer l */
    iftMatrix *kernelbank = ConsensusKernelbankGraph(fs_seeds, "tmp", arch->layer[0].noutput_channels, arch->stdev_factor);
    
    /* BIAS: read bias array */
    char *use_bias = getenv("USE_BIAS");
    float *bias = NULL;

    /************ graph nodes as seeds ***************/
    if (use_bias != NULL)
    {
        bias = iftAllocFloatArray(kernelbank->ncols); 
        for (int col = 0; col < kernelbank->ncols; col++){
            for (int row = 0; row < kernelbank->nrows;){
                for (int ch = 0; ch < ninput_channels; ch++){
                    iftMatrixElem(kernelbank, col, row) =
                        iftMatrixElem(kernelbank, col, row) / stdev[ch];
                    bias[col] -= (mean[ch] * iftMatrixElem(kernelbank, col, row));
                    row++;
                }
            }
        }

        sprintf(filename, "%s/conv%d", param_dir, layer_index);
        iftWriteBias(filename, bias, kernelbank->ncols);
        iftFree(bias);
    }

    // write the final kernel bank
    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftWriteMatrix(kernelbank, filename);
    iftDestroyMatrix(&kernelbank);

    // write the mean and std of the labeled nodes
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
    iftFree(mean);
    iftFree(stdev);

    int pool_stride;

    /* Apply convolutional layer using the consensus kernel bank and the statistics from all markers */
    pool_stride = arch->layer[0].pool_stride;
    arch->layer[0].pool_stride = 1;
    for (int i = 0; i < ngraphs; i++)
    {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");
        sprintf(filename, "tmp/%s.json", basename);

        iftFLIMGraph *graph = iftReadFLIMGraph(filename);
        iftIsValidFLIMGraph(graph);

        output = FLIMGraphConvolutionalLayer(&graph, 1, NULL, arch, 0, layer_index, atrous_factor, param_dir);
        iftIsValidFLIMGraph(output[0]);
        iftDestroyFLIMGraph(&graph);

        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteFLIMGraph(output[0], filename);
        iftDestroyFLIMGraph(&(output[0]));
        iftFree(output);
        iftFree(basename);
    }
    arch->layer[0].pool_stride = pool_stride;

    // updating atrous factor with pooling of current layer
    if (arch->apply_intrinsic_atrous)
    {
        atrous_factor *= arch->layer[0].pool_stride;
        printf("Updating atrous factor for next layer\n");
        fflush(stdout);
    }
    // writing atrous factor on file
    sprintf(filename, "%s/intrinsic_atrous.txt", param_dir);
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d", atrous_factor);
    fclose(fp);

    iftRemoveDir("tmp");
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFileSet(&fs_activ);
}


/*
 * Extract features
 */
void FLIMGraphExtractFeaturesFromLayerPerBatch(iftFileSet *fs_graphs, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                               char *param_dir, int layer_index, char *feat_dir, char *object_dir)
{
    iftFLIMGraph **input = NULL, **output = NULL;
    iftImage **mask = NULL;
    char filename[200];
    int ngraphs = last - first + 1;
    char ext[20];
    sprintf(ext, "%s", iftFileExt(fs_graphs->files[0]->path));
    char **basename = (char **)calloc(ngraphs, sizeof(char *));
    input = (iftFLIMGraph **)calloc(ngraphs, sizeof(iftFLIMGraph *));
    output = NULL;

    if (object_dir != NULL)
        iftWarning("Mask not implemented for graphs", "FLIMGraphExtractFeaturesFromLayerPerBatch");

    /* Load batch */

    for (int i = first, k = 0; i <= last; i++, k++)
    {
        sprintf(filename, "%s/%s", orig_dir, fs_graphs->files[i]->path);
        input[k] = iftReadFLIMGraph(filename);
        basename[k] = iftFilename(fs_graphs->files[i]->path, ext);
    }

    /* For each layer do */

    output = FLIMGraphConvolutionalLayer(input, ngraphs, mask, arch, 0, layer_index, 1, param_dir);

    for (int k = 0; k < ngraphs; k++)
    {
        iftDestroyFLIMGraph(&input[k]);
        input[k] = output[k];
        output[k] = NULL;
    }
    iftFree(output);

    for (int k = 0; k < ngraphs; k++)
    {
        sprintf(filename, "%s/%s.json", feat_dir, basename[k]);
        iftFree(basename[k]);
        iftWriteFLIMGraph(input[k], filename);
        iftDestroyFLIMGraph(&input[k]);
    }

    iftFree(input);
    iftFree(basename);
}

void FLIMGraphExtractFeaturesFromLayerPerImage(iftFileSet *fs_graphs, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                               char *param_dir, int layer_index, char *feat_dir, char *object_dir)
{
    iftFLIMGraph **input = NULL, **output = NULL;
    iftImage **mask = NULL;
    char filename[200];

    printf("\n");

    for (int i = first; i <= last; i++)
    {
        input = (iftFLIMGraph **)calloc(1, sizeof(iftFLIMGraph *));

        char ext[20];
        sprintf(ext, "%s", iftFileExt(fs_graphs->files[i]->path));
        sprintf(filename, "%s/%s", orig_dir, fs_graphs->files[i]->path);
        
        input[0] = iftReadFLIMGraph(filename);

        char *basename = iftFilename(fs_graphs->files[i]->path, ext);
        if (object_dir != NULL)
        {
            iftWarning("Object mask not supported for graphs", "FLIMGraphExtractFeaturesFromLayerPerImage");
        }

        printf("Processing file %s: %d of %d files\r", basename, i + 1, last - first + 1);
        fflush(stdout);

        output = FLIMGraphConvolutionalLayer(input, 1, mask, arch, 0, layer_index, 1, param_dir);
        iftDestroyFLIMGraph(&input[0]);
        input[0] = output[0];
        output[0] = NULL;
        iftFree(output);

        sprintf(filename, "%s/%s.json", feat_dir, basename);
        iftFree(basename);
        iftWriteFLIMGraph(input[0], filename);
        iftDestroyFLIMGraph(&input[0]);
        iftFree(input);
    }
}

void FLIMGraphExtractFeaturesPerImage(iftFileSet *fs_graphs, int first, int last, char *orig_dir, iftFLIMArch *arch,
                                               char *param_dir, char *feat_dir, char *object_dir)
{
    iftFLIMGraph **input = NULL, **output = NULL;
    iftImage **mask = NULL;
    char filename[200];

    printf("\n");

    for (int i = first; i <= last; i++)
    {
        input = (iftFLIMGraph **)calloc(1, sizeof(iftFLIMGraph *));

        char ext[20];
        sprintf(ext, "%s", iftFileExt(fs_graphs->files[i]->path));
        sprintf(filename, "%s/%s", orig_dir, fs_graphs->files[i]->path);
        
        input[0] = iftReadFLIMGraph(filename);

        char *basename = iftFilename(fs_graphs->files[i]->path, ext);
        if (object_dir != NULL)
        {
            iftWarning("Object mask not supported for graphs", "FLIMGraphExtractFeaturesPerImage");
        }

        printf("Processing file %s: %d of %d files\r", basename, i + 1, last - first + 1);
        fflush(stdout);

        for (int l = 0; l < arch->nlayers; l++) {
            output = FLIMGraphConvolutionalLayer(input, 1, mask, arch, l, l+1, 1, param_dir);
            iftDestroyFLIMGraph(&input[0]);
            input[0] = output[0];
            output[0] = NULL;
            iftFree(output);
        }

        sprintf(filename, "%s/%s.json", feat_dir, basename);
        iftFree(basename);
        iftWriteFLIMGraph(input[0], filename);
        iftDestroyFLIMGraph(&input[0]);
        iftFree(input);
    }
}


void ExtractFeaturesFromLayerForTraining(char *param_dir, char *orig_dir, int layer, char *seeds_dir, char *labels_dir, bool write_img)
{
    char file_kernels[255], file_mean[255], file_stdev[255];
    char arch_file[255], layer_input[255], graph_list[255], activation_dir[255];

    sprintf(file_kernels, "%s/conv%d-kernels.npy", param_dir, layer);
    sprintf(file_mean, "%s/conv%d-mean.txt", param_dir, layer);
    sprintf(file_stdev, "%s/conv%d-stdev.txt", param_dir, layer);

    if (!iftFileExists(file_kernels) && !iftFileExists(file_mean) && !iftFileExists(file_stdev))
    {
        iftError("Arquivos de kernels, mean e stdev não existem", "ExtractFeaturesFromLayerForTraining");
    }

    sprintf(activation_dir, "%s/layer%d/", param_dir, layer);

    if (layer == 1)
    {
        sprintf(layer_input, "%s", orig_dir);
        sprintf(graph_list, "%s/input.csv", param_dir);
    }
    else
    {
        sprintf(layer_input, "%s/layer%d/", param_dir, layer - 1);
        sprintf(graph_list, "%s/layer%d.csv", param_dir, layer - 1);
    }

    iftFileSet *list = iftLoadFileSetFromDir(layer_input, 1);
    iftFileSet *list_orig = iftLoadFileSetFromDir(orig_dir, 1);

    if (list->n < list_orig->n)
    {
        if (labels_dir != NULL)
            FLIMGraphExtractFeaturesFromLayer(layer - 1, param_dir, orig_dir, seeds_dir, labels_dir, write_img);
        else
            FLIMExtractFeaturesFromLayer(layer - 1, param_dir, orig_dir, seeds_dir, write_img);
    }

    // Creating list of graphs
    writeCSVFiles(graph_list, layer_input);

    // Creating output dir for layer
    if (iftDirExists(activation_dir))
    {
        iftRemoveDir(activation_dir);
        iftMakeDir(activation_dir);
    }
    else
    {
        iftMakeDir(activation_dir);
    }

    // Reading architecture
    sprintf(arch_file, "%s/arch_layer%d.json", param_dir, layer);
    iftFLIMArch *tmp_arch = iftReadFLIMArch(arch_file);

    // Changing pooling stride to 1
    tmp_arch->layer[0].pool_stride = 1;

    if (labels_dir != NULL)
        iftFLIMGraphExtractFeaturesFromLayer(layer_input, graph_list, tmp_arch, param_dir, layer,
                                             activation_dir, NULL, -1);
    else
        iftFLIMExtractFeaturesFromLayer(layer_input, graph_list, tmp_arch, param_dir, layer,
                                        activation_dir, NULL, -1);

    iftDestroyFLIMArch(&tmp_arch);

    if (labels_dir != NULL)
        FLIMGraphExtractFeaturesFromLayer(layer, param_dir, orig_dir, seeds_dir, labels_dir, write_img);
    else
        FLIMExtractFeaturesFromLayer(layer, param_dir, orig_dir, seeds_dir, write_img);
}

void FLIMGraphExtractFeaturesFromLayer(int layer, char *param_dir, char *orig_dir, char *seeds_dir, char *labels_dir, bool write_img)
{
    char arch_file[255], layer_input[255], graph_list[255], activation_dir[255], kernels_img_dir[255];
    char file_kernels[255], file_mean[255], file_stdev[255];

    sprintf(file_kernels, "%s/conv%d-kernels.npy", param_dir, layer);
    sprintf(file_mean, "%s/conv%d-mean.txt", param_dir, layer);
    sprintf(file_stdev, "%s/conv%d-stdev.txt", param_dir, layer);

    if (!iftFileExists(file_kernels) && !iftFileExists(file_mean) && !iftFileExists(file_stdev))
        iftFLIMTrain(layer, param_dir, orig_dir, seeds_dir);

    sprintf(activation_dir, "%s/layer%d/", param_dir, layer);

    if (layer == 1)
    {
        sprintf(layer_input, "%s", orig_dir);
        sprintf(graph_list, "%s/input.csv", param_dir);
    }
    else
    {
        sprintf(layer_input, "%s/layer%d/", param_dir, layer - 1);
        sprintf(graph_list, "%s/layer%d.csv", param_dir, layer - 1);
    }

    iftFileSet *list = iftLoadFileSetFromDir(layer_input, 1);
    iftFileSet *list_orig = iftLoadFileSetFromDir(orig_dir, 1);

    if (list->n < list_orig->n)
    {
        FLIMGraphExtractFeaturesFromLayer(layer - 1, param_dir, orig_dir, seeds_dir, labels_dir, write_img);
    }

    // Creating list of graphs
    writeCSVFiles(graph_list, layer_input);

    // Creating output dir for layer
    if (iftDirExists(activation_dir))
    {
        iftRemoveDir(activation_dir);
        iftMakeDir(activation_dir);
    }
    else
    {
        iftMakeDir(activation_dir);
    }

    // Reading architecture
    sprintf(arch_file, "%s/arch_layer%d.json", param_dir, layer);
    iftFLIMArch *tmp_arch = iftReadFLIMArch(arch_file);

    iftFLIMGraphExtractFeaturesFromLayer(layer_input, graph_list, tmp_arch, param_dir, layer,
                                         activation_dir, NULL, -1);

    iftDestroyFLIMArch(&tmp_arch);

    // visualization

    /* Verify if layer<n> exists, which is necessary to visualize activations */
    /*char layer_output[255];
    sprintf(layer_output, "%s/layer%d/", param_dir, layer);
    if (!iftDirExists(layer_output)){
      iftMakeDir(layer_output);
    }*/

    if (write_img)
    {
        char ext[10];
        sprintf(ext, "pgm");
        sprintf(kernels_img_dir, "%s/img_activations/layer%d/", param_dir, layer);
        iftRemoveDir(kernels_img_dir);
        iftMakeDir(kernels_img_dir);
        writeActivationsImg(activation_dir, kernels_img_dir, ext, labels_dir);
    }
}

void iftFLIMGraphExtractFeaturesFromLayer(char *orig_dir, char *graph_list, iftFLIMArch *arch, char *param_dir, int layer_index,
                                          char *feat_dir, char *object_dir, int device)
{
    
    iftFileSet *fs_graphs = iftLoadFileSetFromDirOrCSV(graph_list, 1, 1);
    int ngraphs = fs_graphs->n;
    int batchsize = 0;
    int nbatches = 0;
    int nremaining_graphs = ngraphs;
    iftFLIMGraph *input = NULL;
    char filename[200], ext[10];
    bool batch_process = true;
    int ninput_channels = 0;
    int input_graph_size = 0;

    sprintf(ext, "%s", iftFileExt(fs_graphs->files[0]->path));

    /* Verify if all graphs have the same dimension for batch processing */

    sprintf(filename, "%s/%s", orig_dir, fs_graphs->files[0]->path);
    input = iftReadFLIMGraph(filename);

    ninput_channels = input->num_feats;
    input_graph_size = input->num_nodes;
    iftDestroyFLIMGraph(&input);

    for (int i = 1; i < ngraphs; i++)
    {
        sprintf(filename, "%s/%s", orig_dir, fs_graphs->files[i]->path);
        input = iftReadFLIMGraph(filename);

        if ((input_graph_size != input->num_nodes))
        {
            batch_process = false;
            iftDestroyFLIMGraph(&input);
            break;
        }
    }

    /* Select batch size for GPU/CPU processing */

if (batch_process) {  /* process in batch */
#ifdef IFT_GPU
    int ndevices = iftNumberOfDevices();
    if (ndevices > 0 && iftStartDevice(device)){
        batchsize         = iftFLIMBatchSizeGPU(arch,input_graph_size,ninput_channels,device);
        batchsize         = iftMin(batchsize,ngraphs);
        nbatches          = ngraphs/batchsize;
        nremaining_graphs = nremaining_graphs - nbatches*batchsize;

        for (int batch=0; batch < nbatches; batch++){
            int first = batch*batchsize, last  = first + batchsize - 1;
            printf("Processing batch %d of %d batches\n", batch+1, nbatches);
            fflush(stdout);
            FLIMGraphExtractFeaturesFromLayerPerBatch(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
        }
        int first = ngraphs - nremaining_graphs, last  = ngraphs - 1;
        if (first <= last){
            printf("Processing remaining %d images\n", last-first+1);
            fflush(stdout);
            FLIMGraphExtractFeaturesFromLayerPerBatch(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
        }
        iftStopDevice(device);
    } else {
        int first = 0, last  = ngraphs - 1;
        FLIMGraphExtractFeaturesFromLayerPerImage(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
    }
#else
    batchsize = iftFLIMBatchSizeCPU(arch, input_graph_size, ninput_channels);
    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;
    for (int batch = 0; batch < nbatches; batch++) {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\n", batch + 1, nbatches);
        fflush(stdout);
        FLIMGraphExtractFeaturesFromLayerPerBatch(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
    }
    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last) {
        printf("Processing remaining %d images\n", last - first + 1);
        fflush(stdout);
        FLIMGraphExtractFeaturesFromLayerPerBatch(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
    }
#endif
    } else { /* process per image */
        int first = 0, last = ngraphs - 1;
        FLIMGraphExtractFeaturesFromLayerPerImage(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, feat_dir, object_dir);
    }

    iftDestroyFileSet(&fs_graphs);
}


/* --------------------- FLIMGraph seeds (user drawn markers) ----------------------------- */

/**
 * @brief Convert image seeds file to graph seeds file.
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param num_nodes     Number of graph nodes.
 * @param labels_file   File with the superpixel labels map.
 * @param seeds_file    File with the seeds file.
 * @param output_file   File with the graph seeds file.
 */
void seedsPixelToGraph(int num_nodes, char *labels_file, char *seeds_file, char *output_file)
{
    iftImage *label_img = iftReadImageByExt(labels_file);
    iftLabeledSet *S = iftReadSeeds(label_img, seeds_file);
    iftLabeledSet *graph_seeds = NULL;

    int *foreground_count = (int *)calloc(num_nodes, sizeof(int));
    int *background_count = (int *)calloc(num_nodes, sizeof(int));

    while (S != NULL)
    {
        int l;
        int p = iftRemoveLabeledSet(&S, &l);

        if (l == 1)
            foreground_count[label_img->val[p] - 1]++;
        else
            background_count[label_img->val[p] - 1]++;
    }

    iftDestroyImage(&label_img);
    iftDestroyLabeledSet(&S);

    for (int p = 0; p < num_nodes; p++)
    {
        if (foreground_count[p] > 0 || background_count[p] > 0)
        {
            if (foreground_count[p] > background_count[p])
                iftInsertLabeledSet(&graph_seeds, p, 1);
            else
                iftInsertLabeledSet(&graph_seeds, p, 0);
        }
    }

    iftWriteSeedsGraph(graph_seeds, output_file);
    iftDestroyLabeledSet(&graph_seeds);
}

/**
 * @brief Convert image seeds files in 'seeds_dir' to graph seeds files in 'output_dir'.
 * Iterate over the files in 'seeds_dir' and call 'seedsPixelToGraph'.
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param num_nodes   Number of graph nodes.
 * @param labels_dir  Directory of the superpixel labels map.
 * @param seeds_dir   Directory of the seeds file.
 * @param output_dir  Directory of the graph seeds file.
 */
void seedsPixelToGraphDir(char *graph_dir, char *labels_dir, char *seeds_dir, char *output_dir)
{
    iftFileSet *fs_labels = iftLoadFileSetFromDir(labels_dir, 1);
    iftFileSet *fs_seeds = iftLoadFileSetFromDir(seeds_dir, 1);

    char graph_file[255], ext[10];
    char *basename = NULL;

    basename = iftFilename(fs_seeds->files[0]->path, "-seeds.txt");

    sprintf(graph_file, "%s/%s.json", graph_dir, basename);
    iftFLIMGraph *graph = iftReadFLIMGraph(graph_file);
    int num_nodes = graph->num_nodes;

    iftDestroyFLIMGraph(&graph);
    iftFree(basename);

    for (int i = 0; i < fs_seeds->n; i++)
    {
        char seeds_file[255], output_file[255], labels_file[255];

        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(ext, "%s", iftFileExt(fs_labels->files[i]->path));

        sprintf(seeds_file, "%s/%s-seeds.txt", seeds_dir, basename);
        sprintf(output_file, "%s/%s-seeds_graph.txt", output_dir, basename);
        sprintf(labels_file, "%s/%s%s", labels_dir, basename, ext);

        seedsPixelToGraph(num_nodes, labels_file, seeds_file, output_file);
        iftFree(basename);
    }
    iftDestroyFileSet(&fs_labels);
    iftDestroyFileSet(&fs_seeds);
}

/*
Dado um conjunto S de sementes em um grafo graph,
retorna o mesmo conjunto com rótulos iguais para sementes vizinhas.
*/
iftLabeledSet *LabelMarkersGraph(iftFLIMGraph *graph, iftLabeledSet *S)
{
    iftLabeledSet *M = NULL, *seed = S;
    iftDHeap *queue;

    double *markers = (double *)malloc((int)(graph->num_nodes) * sizeof(double));
    double max_label = (double)(graph->num_nodes) + 1.0;
    queue = iftCreateDHeap((int)(graph->num_nodes), markers);

    if (markers == NULL)
    {
        iftError("Error to allocate memory for markers", "LabelMarkersGraph");
    }

    for (int i = 0; i < graph->num_nodes; i++)
        markers[i] = max_label + 1;

    while (seed != NULL)
    {
        int p = seed->elem;
        markers[p] = max_label;
        iftInsertDHeap(queue, p);
        seed = seed->next;
    }

    double curr_label = 1.0;
    while (!iftEmptyDHeap(queue))
    {
        int node_index = iftRemoveDHeap(queue);

        if (markers[node_index] == max_label)
        { // unconnected seed
            markers[node_index] = curr_label;
            curr_label++;
        }
        double node_label = markers[node_index];

        iftFLIMGraphNodeList *node_list_ptr = graph->nodes[node_index].neighbors_list_head;

        while (node_list_ptr != NULL)
        {
            int adj_index = (int)(node_list_ptr->node->index);

            if (markers[adj_index] == max_label && queue->color[adj_index] == IFT_GRAY)
            {
                markers[adj_index] = node_label;
                iftGoUpDHeap(queue, queue->pos[adj_index]);
            }
            node_list_ptr = node_list_ptr->next;
        }
    }

    /*
    Constroi um conjunto de sementes rotuladas
    */
    seed = S;
    while (seed != NULL)
    {
        int p = seed->elem;
        iftInsertLabeledSet(&M, p, (int)(markers[p]));
        seed = seed->next;
    }

    iftFree(markers);
    iftDestroyDHeap(&queue);

    return (M);
}

/**
 * @brief Get the mean and standard deviation from seeds files
 * @param fs_seeds FileSet with seeds files
 * @param dir Directory where the graph files are
 * @param mean Mean of the seeds (output)
 * @param stdev Standard deviation of the seeds (output)
 * @param stdev_factor Factor to add to the standard deviation
*/
void StatisticsFromAllSeedsGraph(iftFileSet *fs_seeds, char *dir, float *mean, float *stdev, float stdev_factor)
{
    int nseeds = 0, num_feats = 0;
    char *basename = NULL;
    char filename[200];
    iftFLIMGraph *graph = NULL;

    /*
    Para cada arquivo (imagem):
        - lê o grafo (para operar a média e desvio sobre as features)
        - lê o arquivo de sementes como iftLabeledSet
        - itera sobre as sementes, contando seu número e o somatório de suas features (cor)
    */
    for (int i = 0; i < fs_seeds->n; i++)
    {
        // lê as sementes
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");

        // le o grafo
        sprintf(filename, "%s/%s.json", dir, basename);
        graph = iftReadFLIMGraph(filename);
        iftIsValidFLIMGraph(graph);
        iftFree(basename);

        num_feats = graph->num_feats;
        iftLabeledSet *S = iftReadSeedsGraph(fs_seeds->files[i]->path);
        while (S != NULL){
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            nseeds += 1;
            for (int b = 0; b < num_feats; b++){
                mean[b] += graph->nodes[p].feats[b];
            }
        }
        iftDestroyFLIMGraph(&graph);
    }

    // termina de calcular a média
    for (int b = 0; b < num_feats; b++){
        mean[b] = mean[b] / nseeds;
    }

    /*
    Para cada arquivo (imagem):
        - lê o grafo (para operar a média e desvio sobre as features)
        - lê o arquivo de sementes como iftLabeledSet
        - itera sobre as sementes, contando seu número e o desvio padrão de suas features (cor)
    */
    for (int i = 0; i < fs_seeds->n; i++)
    {
        // lê as sementes
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");

        // le o grafo
        sprintf(filename, "%s/%s.json", dir, basename);
        graph = iftReadFLIMGraph(filename);

        iftFree(basename);
        num_feats = graph->num_feats;
        iftLabeledSet *S = iftReadSeedsGraph(fs_seeds->files[i]->path);
        while (S != NULL){
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            for (int b = 0; b < num_feats; b++){
                stdev[b] += (graph->nodes[p].feats[b] - mean[b]) * (graph->nodes[p].feats[b] - mean[b]);
            }
        }
        iftDestroyFLIMGraph(&graph);
    }

    // termina de calcualr o desvio padrão
    for (int b = 0; b < num_feats; b++){
        stdev[b] = sqrtf(stdev[b] / nseeds) + stdev_factor;
    }
}


/* --------------------- FLIMGraph pooling functions ----------------------------- */

/**
 * @brief Compute the average pooling from a FLIMGraph
 * @param conv Matrix to store the output
 * @param graph FLIMGraph
 * @param width Width of the pooling
 * @param height Height of the pooling
 * @param depth Depth of the pooling
 * @param stride Stride of the pooling
 * @param most_similar If true, the most similar nodes are selected
 * @note To perform pooling, the average of the features of the neighbors of a node is calculated
 */
iftMImage* iftFLIMGraphAtrousAveragePooling(iftMImage *conv, iftFLIMGraph *graph,
                                      int width, int height, int depth,
                                      int atrous_factor, int stride,
                                      bool most_similar)
{
    // a indexação dos vértices pode alterar o resultado devido ao stride

    // stride: inicia no vértice v de indice 0.
    //         O próximo vértice escolhido depende de stride.
    //         Para stride > 1, pula os vizinhos de v.
    //         A quantidade de pulos é dada por stride-1.

    // pooling: "width" e "height" contêm o tamanho do pooling.
    //          No grafo, para um dado vértice v, seleciona os width*height vizinhos de v
    //          (da mesma forma que se encontra os vizinhos para criar os kernels)
    //          e atualiza as features de v para a média das features dos vizinhos (incluindo v).

    iftMImage *pool = iftCreateMImage(conv->xsize, conv->ysize, conv->zsize, conv->m);

#pragma omp parallel for
    for (int p = 0; p < graph->num_nodes; p++)
    {

        double *dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));

        // auxiliaryHeap para armazenar até k vizinhos mais similares.
        // Se a quantidade de vizinhos for maior que k, remove o menos similar
        // do heap (que está no topo) usando "pop" e verifica quem é o mais similar
        // para nova inserção.
        iftDHeap *auxiliaryHeap = NULL;
        auxiliaryHeap = iftCreateDHeap(graph->num_nodes, dissimilarity);
        if (most_similar)
            auxiliaryHeap->removal_policy = MAXVALUE;

        // MinHeap armazena todos os vizinhos que serão incluídos na matriz de saída.
        // O minHeap fornece a ordem de inclusão na matriz de saída
        // (do mais similar para o menos similar).
        // A raíz do mineap deve sempre ser o vértice inicial,
        // pois possui maior similaridade consigo mesmo.
        iftDHeap *heap;
        heap = iftCreateDHeap((int)graph->num_nodes, dissimilarity);
        if (!most_similar)
            heap->removal_policy = MAXVALUE;

        iftSet *roots = NULL;

        getKNeigborsFLIMGraph(p, graph,
                              width * height,
                              dissimilarity,
                              roots,
                              auxiliaryHeap,
                              heap,
                              atrous_factor,
                              most_similar);

        // move as features dos vértices incluídos no heap para a matriz de saída
        int i = 0;
        iftRemoveDHeap(heap); // remove p from heap
        while (!iftEmptyDHeap(heap))
        {
            int node_index = iftRemoveDHeap(heap);
            for (int b = 0; b < graph->num_feats; b++)
            {
                pool->val[p][b] = conv->val[p][b] +
                                (float)(graph->nodes[node_index].feats[b]);
            }
            i++;
        }
        
        for (int b = 0; b < graph->num_feats; b++)
            pool->val[p][b] = conv->val[p][b] / 
                            (float)(width * height);

        iftDestroyDHeap(&auxiliaryHeap);
        iftDestroyDHeap(&heap);
        iftFree(dissimilarity);
        iftDestroySet(&roots);
    }

    if (stride > 1)
    {
        iftWarning("Stride > 1 not implemented yet", "iftFLIMGraphAtrousAveragePooling");
    }

    return pool;
}

void iftFLIMGraphAtrousAveragePooling_fromMatrix(iftMatrix *conv, iftFLIMGraph *graph,
                                      int width, int height, int depth,
                                      int atrous_factor, int stride,
                                      bool most_similar)
{
    // a indexação dos vértices pode alterar o resultado devido ao stride

    // stride: inicia no vértice v de indice 0.
    //         O próximo vértice escolhido depende de stride.
    //         Para stride > 1, pula os vizinhos de v.
    //         A quantidade de pulos é dada por stride-1.

    // pooling: "width" e "height" contêm o tamanho do pooling.
    //          No grafo, para um dado vértice v, seleciona os width*height vizinhos de v
    //          (da mesma forma que se encontra os vizinhos para criar os kernels)
    //          e atualiza as features de v para a média das features dos vizinhos (incluindo v).

    double *dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));

    // auxiliaryHeap para armazenar até k vizinhos mais similares.
    // Se a quantidade de vizinhos for maior que k, remove o menos similar
    // do heap (que está no topo) usando "pop" e verifica quem é o mais similar
    // para nova inserção.
    iftDHeap *auxiliaryHeap = NULL;
    auxiliaryHeap = iftCreateDHeap(graph->num_nodes, dissimilarity);
    if (most_similar)
        auxiliaryHeap->removal_policy = MAXVALUE;

    // MinHeap armazena todos os vizinhos que serão incluídos na matriz de saída.
    // O minHeap fornece a ordem de inclusão na matriz de saída
    // (do mais similar para o menos similar).
    // A raíz do mineap deve sempre ser o vértice inicial,
    // pois possui maior similaridade consigo mesmo.
    iftDHeap *heap;
    heap = iftCreateDHeap((int)graph->num_nodes, dissimilarity);
    if (!most_similar)
        heap->removal_policy = MAXVALUE;

    iftSet *roots = NULL;

    for (int p = 0; p < graph->num_nodes; p++)
    {
        getKNeigborsFLIMGraph(p, graph,
                              width * height,
                              dissimilarity,
                              roots,
                              auxiliaryHeap,
                              heap,
                              atrous_factor,
                              most_similar);

        // move as features dos vértices incluídos no heap para a matriz de saída
        int i = 0;
        iftRemoveDHeap(heap); // remove p from heap
        while (!iftEmptyDHeap(heap))
        {
            int node_index = iftRemoveDHeap(heap);
            for (int b = 0; b < graph->num_feats; b++)
            {
                iftMatrixElem(conv, b, p) += (float)(graph->nodes[node_index].feats[b]);
            }
            i++;
        }
        iftResetDHeap(heap);
        for (int b = 0; b < graph->num_feats; b++)
            iftMatrixElem(conv, b, p) /= (float)(width * height);
    }

    iftDestroyDHeap(&auxiliaryHeap);
    iftDestroyDHeap(&heap);
    iftFree(dissimilarity);
    iftDestroySet(&roots);

    if (stride > 1)
    {
        iftWarning("Stride > 1 not implemented yet", "iftFLIMGraphAtrousAveragePooling");
    }
}

iftMImage* iftFLIMGraphAtrousMaxPooling(iftMImage *conv,
                                  iftFLIMGraph *graph,
                                  int width,
                                  int height,
                                  int depth,
                                  int atrous_factor,
                                  int stride,
                                  bool most_similar)
{

    iftMImage *pool = iftCreateMImage(conv->xsize, conv->ysize, conv->zsize, conv->m);
    iftSetMImage(pool, IFT_INFINITY_FLT_NEG);
    
    for (int p = 0; p < graph->num_nodes; p++)
    {

        double *dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));

        // MaxHeap para armazenar até k vizinhos mais similares.
        // Se a quantidade de vizinhos for maior que k, remove o menos similar
        // do heap (que está no topo) usando "pop" e verifica quem é o mais similar
        // para nova inserção.
        iftDHeap *auxiliaryHeap = iftCreateDHeap(graph->num_nodes, dissimilarity);
        if (most_similar)
            auxiliaryHeap->removal_policy = MAXVALUE;

        // MinHeap armazena todos os vizinhos que serão incluídos na matriz de saída.
        // O minHeap fornece a ordem de inclusão na matriz de saída
        // (do mais similar para o menos similar).
        // A raíz do mineap deve sempre ser o vértice inicial,
        // pois possui maior similaridade consigo mesmo.
        iftDHeap *heap;
        heap = iftCreateDHeap((int)graph->num_nodes, dissimilarity);
        if (!most_similar)
            heap->removal_policy = MAXVALUE;

        iftSet *roots = NULL;
        
        getKNeigborsFLIMGraph(p, graph, width * height,
                              dissimilarity, roots,
                              auxiliaryHeap, heap, atrous_factor,
                              most_similar);

        // move as features dos vértices incluídos no heap para a matriz de saída
        iftRemoveDHeap(heap); // remove p from heap
        while (!iftEmptyDHeap(heap))
        {
            int node_index = iftRemoveDHeap(heap);
            for (int b = 0; b < graph->num_feats; b++)
            {
                if ((float)(graph->nodes[node_index].feats[b]) > pool->val[p][b])
                    pool->val[p][b] = (float)(graph->nodes[node_index].feats[b]);
            }
        }
        iftDestroyDHeap(&auxiliaryHeap);
        iftDestroyDHeap(&heap);
        iftFree(dissimilarity);
        iftDestroySet(&roots);
    }
    
    if (stride > 1)
    {
        iftWarning("Stride > 1 not implemented yet", "iftFLIMGraphMaxPooling");
    }

    return pool;
}

void iftFLIMGraphAtrousMaxPooling_fromMatrix(iftMatrix *conv,
                                  iftFLIMGraph *graph,
                                  int width,
                                  int height,
                                  int depth,
                                  int atrous_factor,
                                  int stride,
                                  bool most_similar)
{

    iftSetMatrix(conv, IFT_INFINITY_FLT_NEG);

    double *dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));

    // MaxHeap para armazenar até k vizinhos mais similares.
    // Se a quantidade de vizinhos for maior que k, remove o menos similar
    // do heap (que está no topo) usando "pop" e verifica quem é o mais similar
    // para nova inserção.
    iftDHeap *auxiliaryHeap = iftCreateDHeap(graph->num_nodes, dissimilarity);
    if (most_similar)
        auxiliaryHeap->removal_policy = MAXVALUE;

    // MinHeap armazena todos os vizinhos que serão incluídos na matriz de saída.
    // O minHeap fornece a ordem de inclusão na matriz de saída
    // (do mais similar para o menos similar).
    // A raíz do mineap deve sempre ser o vértice inicial,
    // pois possui maior similaridade consigo mesmo.
    iftDHeap *heap;
    heap = iftCreateDHeap((int)graph->num_nodes, dissimilarity);
    if (!most_similar)
        heap->removal_policy = MAXVALUE;

    iftSet *roots = NULL;

    for (int p = 0; p < graph->num_nodes; p++)
    {

        getKNeigborsFLIMGraph(p, graph, width * height,
                              dissimilarity, roots,
                              auxiliaryHeap, heap, atrous_factor,
                              most_similar);

        // move as features dos vértices incluídos no heap para a matriz de saída
        iftRemoveDHeap(heap); // remove p from heap
        while (!iftEmptyDHeap(heap))
        {
            int node_index = iftRemoveDHeap(heap);
            for (int b = 0; b < graph->num_feats; b++)
            {
                if (iftMatrixElem(conv, b, p) < (float)(graph->nodes[node_index].feats[b]))
                    iftMatrixElem(conv, b, p) = (float)(graph->nodes[node_index].feats[b]);
            }
        }
        iftResetDHeap(heap);
    }
    iftDestroyDHeap(&auxiliaryHeap);
    iftDestroyDHeap(&heap);
    iftFree(dissimilarity);
    iftDestroySet(&roots);

    if (stride > 1)
    {
        iftWarning("Stride > 1 not implemented yet", "iftFLIMGraphMaxPooling");
    }
}



/* --------------------- Others ----------------------------- */


/**
 * @brief Given an activation directory (source) with files in .mimg or .json format,
 * write the activations' images in destination directory with extension 'ext' (pgm or png)
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param source        File with features. Can be in .mimg or .json format.
 * @param destination   Directory to write the images.
 * @param ext           Extension of the output images (pgm or png).
 * @param labels_dir    Directory with the superpixel labels map (pgm).
 */
void writeActivationsImg(char *source, char *destination, char *ext, char *labels_dir)
{
    char filename[200];
    char ext_input[10];

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(source, 1);
    for (int i = 0; i < fs_imgs->n; i++)
    {
        sprintf(ext_input, "%s", iftFileExt(fs_imgs->files[i]->path));
        char *basename = iftFilename(fs_imgs->files[i]->path, ext_input);
        if (labels_dir != NULL)
        {
            iftFLIMGraph *graph = iftReadFLIMGraph(fs_imgs->files[i]->path);
            sprintf(filename, "%s/%s.pgm", labels_dir, basename);
            iftImage *img = iftReadImageByExt(filename);

            iftImage **activations = iftGetActivations(graph, img);
            iftDestroyImage(&img);

            int num_activations = (int)graph->num_feats;

            sprintf(filename, "%s/%s", destination, basename);
            iftMakeDir(filename);

#pragma omp parallel for shared(activations, num_activations, destination, basename, ext) private(filename)
            for (int j = 0; j < num_activations; j++)
            {
                sprintf(filename, "%s/%s/kernel-%03d.%s", destination, basename, j + 1, ext);
                iftWriteImageByExt(activations[j], filename);
            }

            for (int j = 0; j < num_activations; j++)
                iftDestroyImage(&(activations[j]));

            iftFree(activations);
            iftDestroyFLIMGraph(&graph);
        }
        else
        {

            iftMImage *mimg = iftReadMImage(fs_imgs->files[i]->path);

            for (int j = 0; j < mimg->m; j++)
            {
                iftImage *band = iftMImageToImage(mimg, 255, j);
                iftImage *band_resize = NULL;
                if (iftIs3DImage(band))
                {
                    band_resize = iftGetXYSlice(band, band->zsize / 2);
                }
                else
                {
                    band_resize = iftCopyImage(band);
                }
                iftDestroyImage(&band);

                sprintf(filename, "%s/%s", destination, basename);
                iftMakeDir(filename);
                sprintf(filename, "%s/%s/kernel-%03d.%s", destination, basename, j + 1, ext);
                iftWriteImageByExt(band_resize, filename);
                iftDestroyImage(&band_resize);
            }

            iftDestroyMImage(&mimg);
        }
        iftFree(basename);
    }
    iftDestroyFileSet(&fs_imgs);
}

/**
 * @brief Write the layer's inputs in a csv file. Used in "ExtractFeatures" functions
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param csv_file    File to write the csv.
 * @param files_dir   Directory of the activation files.
 */
void writeCSVFiles(char *csv_file, char *files_dir)
{
    char ext[10], filename[200];
    iftFileSet *fs_imgs = iftLoadFileSetFromDir(files_dir, 1);

    FILE *fp = fopen(csv_file, "w");
    for (int i = 0; i < fs_imgs->n; i++)
    {
        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));

        char *basename = iftFilename(fs_imgs->files[i]->path, ext);
        sprintf(filename, "%s%s", basename, ext);

        fprintf(fp, "%s\n", filename);
        iftFree(basename);
    }
    fclose(fp);
    iftDestroyFileSet(&fs_imgs);
}

/**
 * @brief Selects a set of kernels from a trained model.
 * This function updates the .npy model file, write a .json file with the selected kernels,
 * call 'FLIMExtractFeaturesFromLayer' to update the feature kernels in 'layer' folder
 * @param param_dir   Directory of the trained model.
 * @param orig_dir    Directory with the original images.
 * @param layer       Layer to select kernels.
 * @param seeds_dir   Directory with the seeds files.
 * @param string_kernels   String with the kernels to select.
 * @param labels_dir  Directory with the superpixel labels map.
 * @param write_img   If true, write the kernels in a .pgm file.
 */
void selectKernelsManual(char *param_dir, char *orig_dir, int layer, char *seeds_dir, char *string_kernels, char *labels_dir, bool write_img)
{
    char kernel_bank_path[255], manual_kernels_file[255], arch_file[255];

    sprintf(kernel_bank_path, "%s/conv%d-kernels.npy", param_dir, layer);
    if (!iftFileExists(kernel_bank_path))
        iftError("Arquivo de kernels não existe", "selectKernelsManual");

    sprintf(manual_kernels_file, "%s/manual_kernels%d.json", param_dir, layer);

    iftMatrix *kernels = iftReadMatrix(kernel_bank_path);
    int nkernels = kernels->ncols;

    int nselected_kernels = 0;
    iftDestroyMatrix(&kernels);

    bool *selected_kernels = (bool *)calloc(nkernels, sizeof(bool));
    bool *neg_kernels = (bool *)calloc(nkernels, sizeof(bool));

    char *ptr = strtok(string_kernels, ",");

    while (ptr != NULL)
    {
        bool negative = false;
        int i = 0, min_kernel = 0, max_kernel = 0;
        if ((ptr[i] < 48 || ptr[i] > 57) && ptr[i] != '-')
            break;

        if (ptr[i] == '-')
        {
            negative = true;
            i++;
        }

        while (ptr[i] != '\0')
        {
            if (ptr[i] < 48 || ptr[i] > 57)
                break;
            min_kernel = min_kernel * 10 + (ptr[i] - '0');
            i++;
        }
        if (ptr[i] == '-')
            i++;
        while (ptr[i] != '\0')
        {
            if (ptr[i] < 48 || ptr[i] > 57)
                break;
            max_kernel = max_kernel * 10 + (ptr[i] - '0');
            i++;
        }

        if (min_kernel <= nkernels)
        {
            selected_kernels[min_kernel - 1] = true;
            if (negative)
                neg_kernels[min_kernel - 1] = true;
            nselected_kernels++;
        }

        min_kernel++;
        while (min_kernel <= max_kernel && min_kernel <= nkernels)
        {
            selected_kernels[min_kernel - 1] = true;
            nselected_kernels++;
            min_kernel++;
        }
        ptr = strtok(NULL, ",");
    }
    /*
     while(ptr != NULL)
     {
        //printf("%s \n", ptr);
      int i = 0, min_kernel = 0, max_kernel = 0;
      if(ptr[i] < 48 || ptr[i] > 57) break;

      while(ptr[i] != '\0' && ptr[i] != '-'){
        if(ptr[i] < 48 || ptr[i] > 57) break;
        min_kernel = min_kernel*10 + (ptr[i] - '0');
        i++;
      }
      if(ptr[i] == '-') i++;
      while(ptr[i] != '\0'){
        if(ptr[i] < 48 || ptr[i] > 57) break;
        max_kernel = max_kernel*10 + (ptr[i] - '0');
        i++;
      }

      if(min_kernel <= nkernels){
        selected_kernels[min_kernel] = true;
        nselected_kernels++;
      }

      min_kernel++;
      while(min_kernel <= max_kernel && min_kernel <= nkernels){
        selected_kernels[min_kernel] = true;
        nselected_kernels++;
        min_kernel++;
      }
      ptr = strtok(NULL, ",");
     }*/

    if (nselected_kernels == 0)
        iftError("selectKernelsManual", "No kernels selected\n");

    iftIntArray *arr = iftCreateIntArray(nselected_kernels);
    int j = 0;
    //printf("selected_kernels: ");
    for (int i = 0; i < nkernels; i++)
    {
        if (selected_kernels[i])
        {
            arr->val[j] = i;
            if (neg_kernels[i])
            {
                arr->val[j] *= -1;
            }
            //printf("%d ", arr->val[j]);
            j++;
        }
    }
    //printf("\n");

    iftDict *json_selected_kernels = iftCreateDict();
    iftInsertIntoDict("selected_kernels", arr, json_selected_kernels);
    iftWriteJson(json_selected_kernels, manual_kernels_file);
    iftDestroyDict(&json_selected_kernels);
    iftDestroyIntArray(&arr);

    free(selected_kernels);
    free(neg_kernels);

    iftMatrix *output_kernels = iftFLIMSelectKernelsManual(kernel_bank_path, manual_kernels_file);
    iftWriteMatrix(output_kernels, kernel_bank_path);
    iftDestroyMatrix(&output_kernels);

    // Updating input of next layer with selected kernels
    ExtractFeaturesFromLayerForTraining(param_dir, orig_dir, layer, seeds_dir, labels_dir, write_img);

    sprintf(arch_file, "%s/arch.json", param_dir);
    iftFLIMArch *arch = iftReadFLIMArch(arch_file);
    arch->layer[layer - 1].noutput_channels = nselected_kernels;
    iftWriteFLIMArch(arch, arch_file);
    iftDestroyFLIMArch(&arch);
    return;
}


/** @brief Insert in "heap" at most k immediate neighbors to the nodes in "roots" that are
   most similar (Euclidean distance) with the features in feature_reference.
 * @param graph: graph with node relations and features
 * @param roots: initial nodes indexes to get the neighbors
 * @param k: number of neighbors to get
 * @param dissimilarity: array to store the dissimilarity between the reference node features and the found neighbors.
 * @param feature_reference: reference node features
 * @param heap: heap of nodes to store the immediate neighbors found
 * @warning Only neighbors with white state in heap may be inserted to it.
 * @author Isabela
 * @date abr, 2023
 */
void getNeighborsFLIMGraph(iftFLIMGraph *graph, iftSet *roots,
                           int k, double *dissimilarity,
                           double *feature_reference,
                           iftDHeap *heap,
                           bool most_similar)
{
    assert(roots != NULL);
    iftSet *tmp_roots = roots;
    double initial_dist = most_similar ? 0.0 : INFINITY;

    while (tmp_roots != NULL)
    {

        iftFLIMGraphNodeList *neigbors_list = graph->nodes[tmp_roots->elem].neighbors_list_head;
        while (neigbors_list != NULL)
        {
            if (heap->color[neigbors_list->node->index] == IFT_WHITE)
            {
                double dist = initial_dist;
                for (int i = 0; i < graph->num_feats; i++)
                {
                    dist += (feature_reference[i] - neigbors_list->node->feats[i]) *
                            (feature_reference[i] - neigbors_list->node->feats[i]);
                }

                if (heap->last + 1 == k)
                {
                    int last = iftRemoveDHeap(heap);
                    if ((most_similar && dist < dissimilarity[last]) ||
                        (!most_similar && dist > dissimilarity[last]))
                    {
                        dissimilarity[neigbors_list->node->index] = dist;
                        iftInsertDHeap(heap, neigbors_list->node->index);
                    }
                    else
                        iftInsertDHeap(heap, last);
                }
                else
                {
                    dissimilarity[neigbors_list->node->index] = dist;
                    iftInsertDHeap(heap, neigbors_list->node->index);
                }
            }
            neigbors_list = neigbors_list->next;
        }
        tmp_roots = tmp_roots->next;
    }
}

/** @brief Insert in heap the k neighbors of node_index according to the graph.
 * roots, dissimilarity, and auxiliaryHeap may have NULL value.
 * @param node_index: reference node to get the neighbors
 * @param graph: graph with node relations and features
 * @param k: number of neighbors
 * @param dissimilarity: (optional) auxiliary structure to compute the dissimilarity between the node and the neighbors
 * @param auxiliaryHeap: (optional) heap of nodes to store the immediate neighbors
 * @param heap: heap of nodes to store the k neighbors
 * @param dilation: number of skips to get neighbors (>= 1) (similar to dilation in convolutional network)
 * @author Isabela
 * @date abr, 2023
 */
void getKNeigborsFLIMGraph(int node_index, iftFLIMGraph *graph, int k,
                           double *dissimilarity, iftSet *roots,
                           iftDHeap *auxiliaryHeap, iftDHeap *heap,
                           int dilation,
                           bool most_similar)
{
    bool has_auxiliaryHeap = false;
    bool has_heap = false;
    bool has_dissimilarity = false;

    if (auxiliaryHeap != NULL)
        has_auxiliaryHeap = true;
    if (heap != NULL)
        has_heap = true;
    if (dissimilarity != NULL)
        has_dissimilarity = true;

    // MaxHeap para armazenar até k vizinhos mais similares.
    // Se a quantidade de vizinhos for maior que k, remove o menos similar
    // do heap (que está no topo) usando "pop" e verifica quem é o mais similar
    // para nova inserção.
    if (!has_auxiliaryHeap)
    {
        auxiliaryHeap = iftCreateDHeap(graph->num_nodes, dissimilarity);
        if (most_similar)
            auxiliaryHeap->removal_policy = MAXVALUE;
    }

    // MinHeap armazena todos os vizinhos que serão incluídos na matriz de saída.
    // O minHeap fornece a ordem de inclusão na matriz de saída
    // (do mais similar para o menos similar).
    // A raíz do mineap deve sempre ser o vértice inicial,
    // pois possui maior similaridade consigo mesmo.
    if (!has_heap)
    {
        heap = iftCreateDHeap((int)graph->num_nodes, dissimilarity);
        if (!most_similar)
            heap->removal_policy = MAXVALUE;
    }
    if (!has_dissimilarity)
        dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));

    dissimilarity[node_index] = 0.0;
    iftInsertDHeap(auxiliaryHeap, node_index);
    iftRemoveDHeap(auxiliaryHeap);
    iftInsertDHeap(heap, node_index);
    iftInsertSet(&roots, node_index); // initial nodes

    int found_neighbors = 1;
    int skips_performed = 1;

    while (found_neighbors < k)
    {
        int miss_neighbors = k - found_neighbors;
        getNeighborsFLIMGraph(graph, roots,
                              miss_neighbors,
                              dissimilarity,
                              graph->nodes[node_index].feats,
                              auxiliaryHeap,
                              most_similar);

        iftDestroySet(&roots);
        if (iftEmptyDHeap(auxiliaryHeap))
            break; // there are no neighbors to include

        bool new_iter = (auxiliaryHeap->last + 1 < miss_neighbors || skips_performed != dilation);

        // adiciona os k vizinhos selecionados (no auxiliaryHeap) em um minHeap
        while (!iftEmptyDHeap(auxiliaryHeap))
        {
            int node_index = iftRemoveDHeap(auxiliaryHeap);
            if (new_iter)
                iftInsertSet(&roots, node_index);
            if (skips_performed == dilation)
            {
                iftInsertDHeap(heap, node_index);
                found_neighbors++;
            }
        }
        if (skips_performed != dilation)
            skips_performed++;
    }
    if (auxiliaryHeap != NULL && has_auxiliaryHeap)
        iftResetDHeap(auxiliaryHeap);

    if (!has_dissimilarity)
        free(dissimilarity);
    if (!has_auxiliaryHeap)
        iftDestroyDHeap(&auxiliaryHeap);
}


/**
 * @brief Compute the kernel bank from a dataset.
 * @param graph Graph with the nodes and their weights.
 * @param M set of labeled graph nodes.
 * @param kernel_size Number of neighbors to compose a kernel.
 * @param nsamples Number of connected components of labeled nodes on the graph.
 * @return iftDataSet* kernel bank
 */
iftDataSet *ComputeSeedDataSetGraph(iftFLIMGraph *graph,
                                    iftLabeledSet *S,
                                    int *kernel_size,
                                    int nsamples,
                                    bool most_similar)
{

    int kernel_neighbors = iftMin(graph->num_nodes, kernel_size[0] * kernel_size[1]);
    int tensor_size = graph->num_feats * kernel_neighbors;

    iftDataSet *Z = iftCreateDataSet(nsamples, tensor_size);

    double *dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));

    // auxiliaryHeap para armazenar até k vizinhos mais similares se
    // most_similar é verdadeiro. Caso contrário, armazena os k menos similares.
    // Se a quantidade de vizinhos for maior que k, remove o elemento da raíz
    // e verifica quem é o mais (ou menos) similar para nova inserção.
    iftDHeap *auxiliaryHeap = NULL;
    auxiliaryHeap = iftCreateDHeap((int)(graph->num_nodes), dissimilarity);
    if (most_similar)
        auxiliaryHeap->removal_policy = MAXVALUE;

    // heap armazena todos os vizinhos que serão incluídos na matriz de saída.
    // O heap fornece a ordem de inclusão na matriz de saída
    // A raíz do heap deve sempre ser o vértice inicial
    iftDHeap *heap = NULL;
    heap = iftCreateDHeap((int)(graph->num_nodes), dissimilarity);
    if (!most_similar)
        heap->removal_policy = MAXVALUE;

    iftSet *roots = NULL;

    Z->nclasses = 0;
    int s = 0;
    iftLabeledSet *seed = S;

    while (seed != NULL)
    {
        int p = seed->elem;

        Z->sample[s].id = s;
        Z->sample[s].truelabel = seed->label;
        if (Z->sample[s].truelabel > Z->nclasses)
            Z->nclasses = Z->sample[s].truelabel;

        getKNeigborsFLIMGraph(p, graph, kernel_neighbors,
                              dissimilarity, roots,
                              auxiliaryHeap, heap, 1,
                              most_similar);

        int j = 0;

        while (!iftEmptyDHeap(heap))
        {
            int node_index = iftRemoveDHeap(heap);
            for (int i = 0; i < graph->num_feats; i++)
            {
                Z->sample[s].feat[j] = graph->nodes[node_index].feats[i];
                j++;
            }
        }
        iftResetDHeap(heap);

        // caso não tenha vizinhos suficientes, completa o kernel com zeros
        while (j / graph->num_feats < kernel_neighbors)
        {
            for (int i = 0; i < graph->num_feats; i++)
            {
                Z->sample[s].feat[j] = 0.0;
                j++;
            }
        }
        s++;
        seed = seed->next;
    }

    iftDestroyDHeap(&auxiliaryHeap);
    iftDestroyDHeap(&heap);
    iftFree(dissimilarity);
    iftDestroySet(&roots);

    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);

    return (Z);
}

iftMatrix *LearnKernelBankGraph(iftFLIMGraph *graph, iftLabeledSet *M,
                                int *kernel_size, int nsamples,
                                int nkernels_per_image, int nkernels_per_marker,
                                bool most_similar)
{
    int tensor_size = graph->num_feats * kernel_size[0] * kernel_size[1];

    // Z contém as informações dos kernels.
    // Z->sample contém um vetor com os kernels.
    // Z->nclasses = numero de marcadores
    // Cada posição é um kernel e contém o rótulo da semente que o gerou. Ou seja, pode conter kernels de mesmo rótulo
    // As features de cada kernel correspondem às cores da semente e de suas posições vizinhas
    iftDataSet *Z = ComputeSeedDataSetGraph(graph, M, kernel_size, nsamples, most_similar);

    // vetor de matrizes de kernels, Cada matriz é banco de kernels
    // cuja quantidade de matrizes (kernels) é igual ao número de marcadores
    iftMatrix **kernels = (iftMatrix **)calloc(Z->nclasses + 1, sizeof(iftMatrix *));
    int total_nkernels = 0;

    // para cada rótulo c, ou seja, para cada marcador
    // calcula seus kernels (uma matriz), armazenando-os em kernels[c]
    for (int c = 1; c <= Z->nclasses; c++)
    {
        iftDataSet *Z1 = iftExtractSamplesFromClass(Z, c);
        int nkernels = nkernels_per_marker;
        /* 0: kmeans, 1: iterated watershed and 2: OPF c clusters */
        kernels[c] = iftComputeKernelBank(Z1, &nkernels, 0, iftEuclideanDistance);
        total_nkernels += nkernels;
        iftDestroyDataSet(&Z1);
    }

    // kernelbank une os kernels de todos os marcadores
    // Seu tamanho é [nkernels][graph->num_feats * kernel_size[0] * kernel_size[1]],
    // em que nkernels é
    iftMatrix *kernelbank = iftCreateMatrix(total_nkernels, tensor_size);

    int k = 0;
    for (int c = 1; c <= Z->nclasses; c++)
    {
        for (int col = 0; col < kernels[c]->ncols; col++, k++)
        {
            for (int row = 0; row < kernels[c]->nrows; row++)
            {
                iftMatrixElem(kernelbank, k, row) = iftMatrixElem(kernels[c], col, row);
            }
        }
    }

    for (int c = 0; c <= Z->nclasses; c++)
    {
        iftDestroyMatrix(&kernels[c]);
    }
    iftFree(kernels);
    iftDestroyDataSet(&Z);

    if (kernelbank->ncols > nkernels_per_image)
    { /* force a number of kernels per image */
        iftMatrix *Rkernels = iftSelectRelevantKernelsByKmeans(kernelbank, nkernels_per_image, iftEuclideanDistance);
        iftDestroyMatrix(&kernelbank);
        kernelbank = Rkernels;
    }

    return (kernelbank);
}

iftFLIMGraph **
FLIMGraphConvolutionalLayer(iftFLIMGraph **graph, int ngraphs,
                            iftImage **mask, iftFLIMArch *arch,
                            int layer, int layer_index,
                            int atrous_factor, char *param_dir)
{
    // Set input parameters

    int ninput_channels = graph[0]->num_feats;
    char *basename = NULL;
    iftFLIMGraph **output_graph = (iftFLIMGraph **)calloc(ngraphs, sizeof(iftFLIMGraph *));

    // Read consensus parameters of the current layer

    char filename[200];
    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftMatrix *kernelbank = iftReadMatrix(filename);
    float *mean = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftReadMeanStdev(filename, mean, stdev, ninput_channels);

    // BIAS: use environment variable to use the bias idea
    float *bias = NULL;
    char *use_bias = getenv("USE_BIAS");

    if (use_bias != NULL)
    {
        bias = iftReadBias(filename);
    }

    // Apply convolutional layer
    for (int i = 0; i < ngraphs; i++)
    {
        // marker-based normalization

        // BIAS: bias do not need normalization, but fill the matrix with mean
        // cria uma matriz de features de tamanho [n][k*m], em que n, k e m são
        // a quantidade de vértices, vizinhos e features, respectivamente
        iftMatrix *imgM;

        if (use_bias == NULL)
        {
            NormalizeGraphByZScore(graph[i], mean, stdev);
            imgM = iftGraphToFeatureMatrix(graph[i], arch->layer[layer].kernel_size[0] * arch->layer[layer].kernel_size[1], NULL, true);
        }
        else
        {
            imgM = iftGraphToFeatureMatrix(graph[i], arch->layer[layer].kernel_size[0] * arch->layer[layer].kernel_size[1], mean, true);
        }

        // convolution
        iftMatrix *conv = iftMultMatrices(imgM, kernelbank); // (n, m*k) x (m*k, k*k) = (n, k*k)
        iftDestroyMatrix(&imgM);

        iftMImage *activ = iftMatrixToMImage(conv, conv->nrows, 1, 1, kernelbank->ncols, 'c');
        iftDestroyMatrix(&conv);
        
        // activation in place
        if (arch->layer[layer].relu) { /* ReLU in place */
#pragma omp parallel for
	        for (int p = 0; p < activ->n; p++) {
	            for (int b = 0; b < activ->m; b++) {
	                /* BIAS: check for the environment variable */
	                if (use_bias!=NULL){
		                activ->val[p][b] += bias[b];
	                }
	                if (activ->val[p][b] < 0)
		                activ->val[p][b] = 0;
	            }
	        }
        }

        // pooling

        if (strcmp(arch->layer[layer].pool_type, "no_pool") != 0)
        {
            iftMImage *pool = NULL;
            output_graph[i] = iftMImageToFLIMGraph(activ, graph[i]);
            iftIsValidFLIMGraph(output_graph[i]);

            if (strcmp(arch->layer[layer].pool_type, "avg_pool") == 0)
            { // ignore the stride to learn the model
                pool = iftFLIMGraphAtrousAveragePooling(activ,
                                                 output_graph[i],
                                                 arch->layer[layer].pool_size[0],
                                                 arch->layer[layer].pool_size[1],
                                                 arch->layer[layer].pool_size[2],
                                                 atrous_factor,
                                                 arch->layer[layer].pool_stride,
                                                 true);
                iftDestroyMImage(&activ);
                activ = pool;
            }
            else
            {
                if (strcmp(arch->layer[layer].pool_type, "max_pool") == 0)
                { // ignore the stride to learn the model
                    pool = iftFLIMGraphAtrousMaxPooling(activ,
                                                 output_graph[i],
                                                 arch->layer[layer].pool_size[0],
                                                 arch->layer[layer].pool_size[1],
                                                 arch->layer[layer].pool_size[2],
                                                 atrous_factor,
                                                 arch->layer[layer].pool_stride,
                                                 true);
                    iftDestroyMImage(&activ);
                    activ = pool;
                }
                else
                {
                    iftWarning("Invalid pooling type has been ignore", "FLIMConvolutionalLayer");
                }
            }

            for (int j = 0; j < output_graph[i]->num_nodes; j++)
            {
                for (int b = 0; b < output_graph[i]->num_feats; b++)
                {
                    output_graph[i]->nodes[j].feats[b] = activ->val[j][b];
                }
            }
        }
        else
        {
            output_graph[i] = iftMImageToFLIMGraph(activ, graph[i]);
        }
        iftDestroyMImage(&activ);
    }

    iftDestroyMatrix(&kernelbank);
    iftFree(mean);
    iftFree(stdev);

    // BIAS: free bias array
    if (use_bias != NULL)
    {
        iftFree(bias);
    }

    return output_graph;
}

iftMatrix *ConsensusKernelbankGraph(iftFileSet *fs_seeds, char *inputdata_dir, int noutput_channels, float stdev_factor)
{
    int nkernels = fs_seeds->n;

    assert(nkernels > 0);

    iftMatrix **kernels = (iftMatrix **)calloc(nkernels, sizeof(iftMatrix *));
    int ncols = 0;
    int nrows = 0;
    char filename[200];

    /* Load kernels from the training images */
    for (int i = 0; i < nkernels; i++)
    {
        char *basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");
        sprintf(filename, "%s/%s-kernels.npy", inputdata_dir, basename);
        kernels[i] = iftReadMatrix(filename);
        ncols += kernels[i]->ncols;
        iftFree(basename);
    }
    nrows = kernels[0]->nrows;

    /* Copy all kernels into a single matrix */

    iftMatrix *Mkernels = iftCreateMatrix(ncols, nrows);
    int k = 0;
    for (int i = 0; i < nkernels; i++)
    {
        for (int col = 0; col < kernels[i]->ncols; col++, k++)
        {
            for (int row = 0; row < kernels[i]->nrows; row++)
            {
                iftMatrixElem(Mkernels, k, row) = iftMatrixElem(kernels[i], col, row);
            }
        }
    }

    for (int i = 0; i < nkernels; i++)
    {
        iftDestroyMatrix(&kernels[i]);
    }
    iftFree(kernels);

    /* Reduce the number of kernels into the desired number of output channels */

    iftMatrix *Rkernels = NULL;

    if (Mkernels->ncols <= noutput_channels)
    {
        Rkernels = iftCopyMatrix(Mkernels);
    }
    else
    {
        if (Mkernels->ncols > Mkernels->nrows)
        {
            Rkernels = iftSelectRelevantKernelsByPCA(Mkernels, noutput_channels, stdev_factor);
        }
        else
        {
            Rkernels = iftSelectRelevantKernelsByKmeans(Mkernels, noutput_channels, iftEuclideanDistance);
        }
    }

    iftDestroyMatrix(&Mkernels);

    return (Rkernels);
}


