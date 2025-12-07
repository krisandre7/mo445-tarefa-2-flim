#include "ift.h"

typedef struct _arcweights {
  float **val;
  int     n;
} ArcWeights;

ArcWeights *CreateArcWeights(int nlabels)
{
  ArcWeights *oindex=(ArcWeights *)calloc(1,sizeof(ArcWeights));
  int i;

  oindex->val = (float **) calloc(nlabels,sizeof(float *));
  for (i=0; i < nlabels; i++) 
    oindex->val[i] = (float *)calloc(nlabels,sizeof(float));
  
  oindex->n = nlabels;
  return(oindex);
}

void DestroyArcWeights(ArcWeights **oindex)
{
  int i;

  if ((*oindex) != NULL){
    for (i=0; i < (*oindex)->n; i++) 
      free((*oindex)->val[i]);
    free((*oindex)->val);
    free((*oindex));
    *oindex = NULL;
  }
}

void PrintArcWeights(ArcWeights *oindex)
{
  int i,j;

  printf("\n");

  for(i=0; i < oindex->n; i++) {
    for(j=0; j < oindex->n; j++) 
      printf("%5.2f ",oindex->val[i][j]);
    printf("\n");
  }
}

ArcWeights *OverlappingIndex(Subgraph *sg){ 
  int   i, j, k;
  float weight,tot,max;
  ArcWeights *oindex;

  oindex = CreateArcWeights(sg->nlabels);

  for (i = 0; i < sg->nnodes; i++){
    for (j = 0; (j < sg->nnodes); j++){
      k = sg->ordered_list_of_nodes[j];
      weight = opf_ArcWeight(sg->node[k].feat,sg->node[i].feat,sg->nfeats);

      if (weight <= sg->node[k].radius){
	oindex->val[sg->node[i].label][sg->node[k].label]++;
      }
    }
  }
  
  // Normalize the overlapping index

  for(i=0; i < sg->nlabels; i++){
    tot=0;
    for(j=0; j < sg->nlabels; j++) 
      tot += oindex->val[i][j];
    for(j=0; j < sg->nlabels; j++) 
      oindex->val[i][j] /= tot;
  }

  return(oindex);
}


int FindSubgraphRoot(Subgraph *sg, int i)
{
  if (sg->node[i].root == i)
    return(i);
  else
    return(sg->node[i].root=FindSubgraphRoot(sg, sg->node[i].root));
}

void MergeOverlapClusters(Subgraph *sg, ArcWeights *oindex)
{
  int i,j;

  printf("initial number of clusters %d\n",sg->nlabels);

  for (i=0; i < sg->nnodes; i++) {
    for (j=0; j < sg->nnodes; j++) {
      if (sg->node[i].label != sg->node[j].label)
	if (oindex->val[sg->node[i].label][sg->node[j].label] >= 0.10){
	  sg->node[i].root = FindSubgraphRoot(sg,j);
	}
    }
  }

  for (i=0; i < sg->nnodes; i++) 
    sg->node[i].root = FindSubgraphRoot(sg,i);

  j=0;
  for (i=0; i < sg->nnodes; i++) 
    if (i==sg->node[i].root){
      sg->node[sg->node[i].root].label=j;
      j++;
    }
  sg->nlabels=j;
  
  for (i=0; i < sg->nnodes; i++) 
    sg->node[i].label = sg->node[sg->node[i].root].label;

  printf("final number of clusters %d\n",sg->nlabels);
  
}

Subgraph *ReadData(char *filename)
{
  FILE *fp=fopen(filename,"r");
  Subgraph *sg = NULL;
  int i,j,f;

  fscanf(fp,"%d",&i);
  sg=CreateSubgraph(i);

  sg->nlabels=0;
  sg->nfeats = 2;

  for (i=0; i < sg->nnodes; i++){
    sg->node[i].feat = AllocFloatArray(2);
    for (j=0; j < sg->nfeats; j++) {
      fscanf(fp,"%d",&f);
      sg->node[i].feat[j]=f;
    }
    fscanf(fp,"%d",&(sg->node[i].truelabel));
    if (sg->node[i].truelabel > sg->nlabels)
      sg->nlabels = sg->node[i].truelabel;
    sg->node[i].adj=NULL;
  }
  fclose(fp);

  sg->nlabels += 1;

  return(sg);
}

Image *Create2DFeatureImage(Subgraph *sg, char opt)
{
  Image *img;
  int    x,y,i,p,xmin,xmax,ymin,ymax;

  xmin=ymin=INT_MAX; xmax=ymax=INT_MIN;
  for (i=0; i < sg->nnodes; i++){
    if (sg->node[i].feat[0] < xmin)
      xmin = sg->node[i].feat[0];
    if (sg->node[i].feat[1] < ymin)
      ymin = sg->node[i].feat[1];
    if (sg->node[i].feat[0] > xmax)
      xmax = sg->node[i].feat[0];
    if (sg->node[i].feat[1] > ymax)
      ymax = sg->node[i].feat[1];
  }
  printf("Creating an %d x %d image\n",xmax+10,ymax+10);
  img=CreateImage(xmax+10,ymax+10);

  for (i=0; i < sg->nnodes; i++) {
    x = sg->node[i].feat[0]+1;
    y = sg->node[i].feat[1]+1;
    p = x + img->tbrow[y];

    switch (opt) {
    case 0:
      PaintCircle(img,p,1.5,(int)(255*((float)sg->node[i].dens-(float)1)/(opf_MAXDENS-1+1)));
      break;
    case 1:
      PaintCircle(img,p,1.5,sg->node[i].label+1);
      break;
    case 2:
      PaintCircle(img,p,1.5,sg->node[i].truelabel);
      break;
    case 3:
      PaintCircle(img,p,1.5,255);
      break;
    default:
      PaintCircle(img,p,1.5,sg->node[i].label+1);
    }
  }

  return(img);
}


int main(int argc, char **argv)
{
  timer *t1=NULL,*t2=NULL;
  Image    *img=NULL;
  Subgraph *sg=NULL,*sgtrain=NULL,*sgeval=NULL;
  Set *S=NULL;
  ArcWeights *oindex=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=4)
    Error("Usage must be: datacluster <data.txt> <kmax> <volume or nclusters>","main");

  sg = ReadData(argv[1]);

  opf_ArcWeight=opf_EuclDist;

  //   Unsupervised

  t1 = Tic();
  opf_BestkMinCut(sg,1, atoi(argv[2]));
  opf_ElimMaxBelowVolume(sg,atoi(argv[3]));
  opf_OPFClustering(sg);
  oindex = OverlappingIndex(sg);
  PrintArcWeights(oindex);
  MergeOverlapClusters(sg, oindex);
 //  S = SelectLargestDomes(sg,atoi(argv[3]));
  //OPFByMarkers(sg, &S);

  t2 = Toc();

  /*
  int p;
  for (p=0; p < sg->nnodes; p++) {
    if (sg->node[p].root==p) sg->node[p].label=sg->node[p].truelabel;
    if (sg->node[p].truelabel > sg->nlabels)
      sg->nlabels = sg->node[p].truelabel;
  }
  sg->nlabels++;
  for (p=0; p < sg->nnodes; p++) {
    if (sg->node[p].root!=p) sg->node[p].label=sg->node[sg->node[p].root].truelabel;
  }
  printf("Accuracy %f\n",Accuracy(sg));

  */


  // Supervised
  /*
  t1 = Tic();
  SplitSubgraph(sg, &sgtrain, &sgeval, 0.50);
  SupTrainCompGraph(sgtrain);
  ClassifyCompGraph(sgtrain, sgeval);
  LearningCompGraph(&sgtrain,&sgeval,5);
  ClassifyCompGraph(sgtrain, sgeval);
  DestroySubgraph(&sg);
  sg = MergeSubgraphs(sgtrain,sgeval);
  t2 = Toc();
  */

  img=Create2DFeatureImage(sg,0);
  WriteImage(img,"density.pgm");
  DestroyImage(&img);

  img=Create2DFeatureImage(sg,1);
  WriteImage(img,"label.pgm");
  DestroyImage(&img);
  /*
  img=Create2DFeatureImage(sg,2);
  WriteImage(img,"truelabel.pgm");
  DestroyImage(&img);
  */
  DestroySubgraph(&sg);
  DestroySubgraph(&sgtrain);
  DestroySubgraph(&sgeval);
  DestroyArcWeights(&oindex);

  fprintf(stdout,"datacluster in %f ms\n",CTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}
