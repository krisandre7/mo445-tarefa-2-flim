#include "ift.h"

/* Data Structures for iftTrainEdges */

typedef struct ift_label_edge {
  int	  truelabel;
  int 	  p,q;
} iftLabelEdge;

typedef struct ift_train_edges_data {
  int          nfilename;
  int          n; /* number of edges*/
  char         *filename;
  iftMImage    *image;
  iftLabelEdge *labelEdge;
} iftLabelEdgesData;

typedef struct ift_train_edges {
  int                nimages;
  int                nsamplesbi;
  float              radius;
  iftLabelEdgesData* data;
} iftTrainEdges;


iftLabelEdgesData  *iftCreateLabelEdgesData(int nedges);
void  iftDestroyLabelEdgesData(iftLabelEdgesData **le);
iftTrainEdges* iftCreateTrainEdges(int nimages, int nsamplesbi);
void iftDestroyTrainEdges(iftTrainEdges **Z);

iftLabelEdgesData* iftExtractLabelEdges  (iftMImage *mimg,iftImage *imgGT, iftAdjRel* A,int nsamplesbi ,float percBorder,float radius);
iftLabelEdgesData* iftExtractLabelEdgesWG(iftMImage *mimg,iftImage *imgGT, iftAdjRel* A,int nsamplesbi ,float percBorder,float radius);
iftTrainEdges*     iftExtractTrainEdges  (char *file_imageNames,char *file_imageNamesGT,int nsamplesbi ,float percBorder,float radius);

iftDataSet* iftTrainEdgesToDataSet(iftTrainEdges *trainEdges, iftMSConvNetwork* msconvnet);

void iftWriteTrainEdges(iftTrainEdges *trainEdges,char *filenameTrainEdges);
iftTrainEdges* iftReadTrainEdges(char *filenameTrainEdges);
