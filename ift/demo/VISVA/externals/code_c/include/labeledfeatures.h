
#ifndef _LABELEDFEATURES_H_
#define _LABELEDFEATURES_H_

#include "oldift.h"

#include "featmap.h"

/*Use this struture to store training, evaluation 
  or test datasets to use in supervised learning 
  to project a classifier.*/

typedef struct _LabeledFeatures {
  int n;
  int *label;
  FeatMap *fmap;
} LabeledFeatures;


/*All data assigned with this method will be 
  automatically deleted by "DestroyLabeledFeatures" as 
  appropriate (i.e. it takes ownership of the fmap).*/
LabeledFeatures *CreateLabeledFeatures(FeatMap *fmap);
void             DestroyLabeledFeatures(LabeledFeatures **lf);

/*Creates a copy of the original dataset.*/
LabeledFeatures *CloneLabeledFeatures(LabeledFeatures *lf);

/*Write the dataset to disk in binary format.*/
void  WriteLabeledFeatures(LabeledFeatures *lf, 
			   char *filename);

/*Read a dataset from the disk in binary format.*/
LabeledFeatures *ReadLabeledFeatures(char *filename);

/*It shuffles the samples at random.*/
void RandomizeLabeledFeatures(LabeledFeatures *lf);

/*Returns a subset with samples of all classes in the 
  proportion given by "rate". This subset is removed from the
  original dataset. You should call "RandomizeLabeledFeatures"
  first to ensure the selection of random samples.*/
LabeledFeatures *RemoveLabeledFeaturesSamples(LabeledFeatures **lf, 
					      float rate);

/*Concatenate two datasets.*/
LabeledFeatures *MergeLabeledFeatures(LabeledFeatures *lf1, 
				      LabeledFeatures *lf2);

#endif


