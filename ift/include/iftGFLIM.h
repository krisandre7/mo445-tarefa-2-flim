/**
 * @file 
 * @brief GFLIM: Graph-based Feature Learning From Image Markers -- functions to
 * learn the parameters of a sequence of convolutional layers from
 * labeled nodes in graphs and employ the resulting model for feature extraction.
 * 
 * @note <b>Programs:</b>
 * * @ref iftFLIM-LearnModel.c = Learns the CNN model
 * * @ref iftFLIM-ExtractFeatures.c = Extracts image features.
 */

#ifndef IFT_GFLIM_H
#define IFT_GFLIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "iftFLIM.h"
#include "iftMatrix.h"
#include "iftDataSet.h"
#include "iftMImage.h"
#include "ift/core/io/Dir.h"
#include "ift/core/io/Json.h"


/***************/
typedef struct iftFLIMGraphNode iftFLIMGraphNode;
typedef struct iftFLIMGraphNodeList iftFLIMGraphNodeList;

typedef struct iftFLIMGraphNodeList{
    iftFLIMGraphNode *node;
    iftFLIMGraphNodeList *next;
}iftFLIMGraphNodeList;

typedef struct iftFLIMGraphNode{
    double *feats;
    unsigned long numNeighbors;
    unsigned long numPixels;
    unsigned long index;
    iftFLIMGraphNodeList *neighbors_list_head;
    iftFLIMGraphNodeList *neighbors_list_tail;
}iftFLIMGraphNode;


typedef struct _iftFLIMGraph {
  iftFLIMGraphNode *nodes;
  unsigned long num_nodes;
  unsigned long num_feats;
} iftFLIMGraph;



iftFLIMGraph *iftReadFLIMGraph(char *filename);

bool iftIsValidFLIMGraph(iftFLIMGraph *graph);

void iftDestroyFLIMGraph(iftFLIMGraph **graph);

void iftFLIMGraphLearnLayer(char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftFLIMArch *arch, char *output_dir);


iftMImage *iftGraphToMImage(iftFLIMGraph *graph, iftImage *labels);

void iftFLIMGraphExtractFeaturesFromLayer(char *orig_dir, char *graph_list, iftFLIMArch *arch, char *param_dir, int layer_index,
                                     char *feat_dir, char *object_dir, int device);

void iftImageToFLIMGraph(char *image_filename, char *labels_filename, 
                      char *graph_filename, int num_init_seeds, 
                      int num_superpixels);


void iftGraphToMIMG(char *graph_dir, char *labels_dir, char *output_dir);


/***************/
  
  
#ifdef __cplusplus
}
#endif


#endif
