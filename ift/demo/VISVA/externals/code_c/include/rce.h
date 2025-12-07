
#ifndef _RCE_H_
#define _RCE_H_

#include "oldift.h"

#include "featmap.h"
#include "labeledfeatures.h"


/*Reduced Coulomb energy networks (RCE)*/

typedef struct _RCEClassifier {
  int n;
  FeatMap *fmap;
  real *radius;
  //Distance function
  real (*distance)(real*, real*, int);
} RCEClassifier;


/*"lf" -> The training set.*/
RCEClassifier *RCETraining(LabeledFeatures *lf,
			   real (*distance)(real*, real*, int));

int   RCEClassifySample(RCEClassifier *rce,
			real *fv, int nfeat);
void  RCEClassifySamples(RCEClassifier *rce,
			 LabeledFeatures *lf);

#endif





