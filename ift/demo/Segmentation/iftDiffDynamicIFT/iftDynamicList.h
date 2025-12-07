//
// Created by ilan on 04/12/2021.
//

#ifndef ROBOT_EXPERIMENTS_IFTDYNAMICLIST_H
#define ROBOT_EXPERIMENTS_IFTDYNAMICLIST_H

#include "ift.h"

// dynamic_list => dynamic_tree

typedef struct ift_dynamic_list {
    struct ift_dynamic_list *prev;
    struct ift_dynamic_list *next;
  int root;
  int count; // nnodes
  int bands; // nfeats
  float *sum; // sum_feat 
} iftDynamicList; // iftDynamicTree 

// iftInsertDynamicTree

iftDynamicList *iftCreateDynamicList(iftDynamicList **list, int root, int bands) {
    iftDynamicList *node = (iftDynamicList *) iftAlloc(1, sizeof(iftDynamicList));

    node->root  = root;
    node->bands = bands;
    node->count = 0;
    node->sum   = iftAllocFloatArray(bands);
    node->prev  = NULL;
    node->next  = *list;

    if (*list)
        (*list)->prev = node;
    list = &node;

    return node;
}

// iftRemoveDynamicTree

void iftRemoveDynamicListNode(iftDynamicList **list) {
    iftDynamicList *next = (*list)->next;
    iftDynamicList *prev = (*list)->prev;

    if (prev != NULL) {
        prev->next = next;
    }

    if (next != NULL) {
        next->prev = prev;
    }

    iftFree((*list)->sum);
    iftFree(*list);
    list = NULL;
}

// Mover para iftUpdateDynamicTree 
void iftRemoveDynamicListVoxel(iftDynamicList *node, float *voxel_color) {
    if (node->count) {
        for (int i = 0; i < node->bands; i++)
            node->sum[i] -= voxel_color[i];
        node->count--;
    }
}

// Mover para iftUpdateDynamicTree 
void iftInsertDynamicListVoxel(iftDynamicList *node, float *voxel_color) {
    for (int i = 0; i < node->bands; i++)
        node->sum[i] += voxel_color[i];
    node->count++;
}

// Mover para iftDynamicArcWeight
float iftDynamicListDistanceVoxel(iftDynamicList *node, float *voxel_color) {
    float distance = 0;
    for (int i = 0; i < node->bands; i++) {
        float mean = node->sum[i] / node->count;
        distance += (mean - voxel_color[i]) * (mean - voxel_color[i]);
    }
    return distance;
}

// iftDestroyDynamicTrees

void iftDestroyDynamicList(iftDynamicList **list) {
    iftDynamicList  *node = *list;

    while (node) {
        iftDynamicList *next = (*list)->next;
        iftFree(node->sum);
        iftFree(node);
        node = next;
    }

    list = NULL;
}

#endif //ROBOT_EXPERIMENTS_IFTDYNAMICLIST_H
