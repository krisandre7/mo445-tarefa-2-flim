// version 00.00.02

#ifndef _MYOPF3_H_
#define _MYOPF3_H_

#include "oldift.h"
#include "preproc.h"

Features3 *my_LMSSceneFeats(Scene *scn, int nscales);

// Compute node dist density for classification node 
void NodeDD(Subgraph *sg, CNode *node);

int   my_Findkmax(Subgraph *sgTrain);
float MaxEuclDist3(Features3 *f);

// Estimate the best k
int BestkByMaxContrast(Subgraph *sgTrain,
		       Subgraph *sgEval, int kmax);

float EvaluateProbContrast(Subgraph *sgTrain, Subgraph *sgEval);

float DDF2ObjProbability_1(float dist0, float dist1);
float DDF2ObjProbability_2(float dist0, float dist1, float maxdist);

void  my_Learning(Subgraph **sgtrain, Subgraph **sgeval, 
		  int iterations, int kmax);
void  SwapAllAtRandom(Subgraph **sgtrain, Subgraph **sgeval);

Subgraph *my_BestkSubgraph3(Features3 *f, Set *Si, Set *Se, int nsamples);
float    *my_ProbabilityArray3(Subgraph *sg, Features3 *f);

#endif


