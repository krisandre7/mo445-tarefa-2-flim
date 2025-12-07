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
	if (oindex->val[sg->node[i].label][sg->node[j].label] >= 0.07){
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

char VerifyForestIntegrity(Subgraph *sg)
{
  int p,q;

  for (p=0; p < sg->nnodes; p++) {
    if (sg->node[sg->node[p].root].pred != NIL)
      return(0);
    if (sg->node[sg->node[p].root].root != sg->node[p].root)
      return(0);
    q = p;
    while (q != NIL){
      if (sg->node[q].label != sg->node[sg->node[q].root].label)
	return(0);
      q = sg->node[q].pred;
    }
  }
  return(1); 
}

Image *TreatOutliers(Image *label)
{
  Image *flabel=ift_CopyImage(label);
  int p,q,r,s,i,n=label->ncols*label->nrows;
  Pixel u,v;
  AdjRel *A=Circular(1.0);
  int *FIFO,first=0,last=0;
  int *color;
  Set *S = NULL;

  FIFO  = AllocIntArray(n);
  color = AllocIntArray(n);

  for(p=0; p < n; p++) {
    if (flabel->val[p]==0){
      FIFO[last]=p;
      InsertSet(&S, p);
      last++;
      while(first != last){
	q = FIFO[first];
	color[q] = BLACK;
	u.x = q % label->ncols;
	u.y = q / label->ncols;
	first++;
	for (i=1; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  r = v.x + label->tbrow[v.y];
	  if (ValidPixel(label,v.x,v.y)){
	    if (color[r] == WHITE) {
	      if (label->val[r]==0){
		FIFO[last] = r;
		color[r] = GRAY;
		InsertSet(&S, r);
		last++;
	      }else{
		flabel->val[p] = label->val[r];
		first=last=0;
		break;
	      }
	    }
	  }
	}
      }
      while (S != NULL) {
	s = RemoveSet(&S);
	color[s] = WHITE;
      }
    }
  }
  DestroyAdjRel(&A);
  free(FIFO);
  free(color);

  return(flabel);
}

void DrawLabeledRegionsInPlace(CImage *cimg, Image *label){
  int x,y,k,p,q,u,v;
  AdjRel *A;
  Image *img=cimg->C[0];

  A = Circular(1.0);

  for (p=0; p < img->ncols*img->nrows; p++) {
    x = p%img->ncols;
    y = p/img->ncols;
    for(k=1;k<A->n;k++){
      u = x+A->dx[k];
      v = y+A->dy[k];
      if(ValidPixel(img,u,v)){
	q= u + img->tbrow[v];
	if (label->val[p] != label->val[q]){
	  cimg->C[0]->val[p]=255;
	  cimg->C[1]->val[p]=255;
	  cimg->C[2]->val[p]=255;
	  break;
	}
      }
    }
  }

  DestroyAdjRel(&A);
}

CImage *CGaussianFilter(CImage *cimg1)
{
  AdjRel *A=Circular(3.0);
  Kernel *K=GaussianKernel(A,0.5);
  CImage *cimg2=(CImage *)calloc(1,sizeof(CImage));

  cimg2->C[0] = LinearFilter2(cimg1->C[0],K);
  cimg2->C[1] = LinearFilter2(cimg1->C[1],K);
  cimg2->C[2] = LinearFilter2(cimg1->C[2],K);

  DestroyAdjRel(&A);
  DestroyKernel(&K);

  return(cimg2);
}

int main(int argc, char **argv)
{
  timer *t1=NULL,*t2=NULL;
  Image    *img=NULL,*label=NULL,*flabel=NULL,*pdf=NULL;
  Subgraph *sg=NULL;
  Features *f=NULL;
  CImage *cimg2=NULL,*cimg1=NULL;
  char ext[10],*pos;
  ArcWeights *oindex=NULL;

  opf_ArcWeight=opf_EuclDist;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=4)
    Error("Usage must be: imagecluster <image> <nsamples> <kmax>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);

  t1 = Tic();


  if (strcmp(ext,"pgm")==0){
    img = ReadImage(argv[1]);
    cimg1 = (CImage *) calloc(1,sizeof(CImage *));
    cimg1->C[0]=ift_CopyImage(img);
    cimg1->C[1]=ift_CopyImage(img);
    cimg1->C[2]=ift_CopyImage(img);
    sg  = RandomSampl(img,atoi(argv[2])); 
    f   = LMSImageFeats(img,3);
  }else{
    if (strcmp(ext,"ppm")==0){
      cimg2 = ReadCImage(argv[1]);
      cimg1 = CGaussianFilter(cimg2);
      img   = ift_CopyImage(cimg2->C[1]);      
      sg    = RandomSampl(img,atoi(argv[2]));
      f     = LabCImageFeats(cimg1);   
      DestroyCImage(&cimg1);
      cimg1 = CopyCImage(cimg2);
      DestroyCImage(&cimg2);
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  SetSubgraphFeatures(sg,f);
  opf_BestkMinCut(sg,1, atoi(argv[3]));
  opf_OPFClustering(sg);
    
  oindex = OverlappingIndex(sg);
  PrintArcWeights(oindex);
  MergeOverlapClusters(sg, oindex);

  label=ImageClassKnnGraph(sg,f);
  flabel = TreatOutliers(label);
  pdf  = ImagePDF(sg,f);
  
  t2 = Toc();

  WriteImage(pdf,"pdf.pgm");
  WriteImage(flabel,"label.pgm");
  DrawLabeledRegionsInPlace(cimg1,flabel);
  WriteCImage(cimg1,"result.ppm");
  DestroySubgraph(&sg);

  DestroyCImage(&cimg1);
  DestroyImage(&pdf);
  DestroyImage(&img);
  DestroyImage(&label);
  DestroyImage(&flabel);
  DestroyCImage(&cimg2);
  DestroyFeatures(&f);
  
  DestroyArcWeights(&oindex);

  fprintf(stdout,"imagecluster in %f ms\n",CTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}
