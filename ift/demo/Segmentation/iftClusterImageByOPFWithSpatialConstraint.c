#include "ift.h"

typedef struct ift_graphnode {
    /**  Maximum arc weight from the node to its neighbors */
    float maxarcw;
    /** Corresponding root node in the graph */
    int   root;
    /** Corresponding training sample in the original dataset */
    int   sample;
    /** List of adjacent nodes */
    iftAdjSet *adj;
    /** List of adjacent nodes on plateaus of density */
    iftSet    *adjplat;
} iftGraphNode;

typedef struct ift_graph {
    /** List of nodes in the graph */
    iftGraphNode *node;
    /** List of path value of the nodes */
    float      *pathval;
    /** List of nodes ordered by its path value */
    int        *ordered_nodes;
    /** Number of nodes of the graph */
    int         nnodes;
    /** Priority queue */
    iftFHeap   *Q;
    /** Corresponding dataset */
    iftDataSet *Z;
} iftGraph;

iftGraph *iftCreateGraph(iftDataSet *Z)
{
  iftGraph *graph=(iftGraph *)iftAlloc(1,sizeof(iftGraph));
  int nnodes=Z->ntrainsamples;

  if (nnodes == 0){
      iftError("No samples for training", "iftCreateKnnGraph");
  }

  graph->nnodes = nnodes;
  graph->node   = (iftGraphNode *)iftAlloc(nnodes,sizeof(iftGraphNode));

  if (graph->node == NULL){
      iftError(MSG_MEMORY_ALLOC_ERROR, "iftCreateKnnGraph");
  }

  graph->pathval       = iftAllocFloatArray(nnodes);
  graph->ordered_nodes = iftAllocIntArray(nnodes);
  graph->Q        = iftCreateFHeap(nnodes,graph->pathval);
  graph->Z        = Z;

#pragma omp parallel for
  for (int u=0; u < graph->nnodes; u++){
    graph->node[u].adj      = NULL;
    graph->node[u].adjplat  = NULL;
    graph->node[u].sample   = IFT_NIL;
    graph->node[u].maxarcw  = 0.0;
    graph->node[u].root     = u;
  }

  int u = 0;
  for (int s=0; s < Z->nsamples; s++){
    if (Z->sample[s].status == IFT_TRAIN){
      graph->node[u].sample = s;
      u++;
    }
  }

  return(graph);
}

void iftDestroyGraph(iftGraph **graph)
{
  int u;
  iftGraph *aux=(*graph);

  if (aux!=NULL){
    for (u=0; u < aux->nnodes; u++){
      if (aux->node[u].adj != NULL){
        iftDestroyAdjSet(&(aux->node[u].adj));
      }
      if (aux->node[u].adjplat != NULL){
        iftDestroySet(&(aux->node[u].adjplat));
      }
    }
    iftFree(aux->node);
    iftFree(aux->pathval);
    iftFree(aux->ordered_nodes);
    iftDestroyFHeap(&(aux->Q));
    iftFree(aux);
    (*graph) = NULL;
  }
}

void iftPDFWithSpatialConstraint(iftGraph *graph, float df, float di)
{

  iftDataSet   *Z=graph->Z;
  float maximum= IFT_INFINITY_FLT_NEG, minimum= IFT_INFINITY_FLT;

  iftAdjRel *A;
  A=iftCircular(di);
  printf("The adjacency size is %d\n",A->n-1);
//  float K=2.0*df*df/9;
  iftMImage *mimg=(iftMImage *)Z->ref_data;

  // Compute the probability density function
#pragma omp parallel for
  for (int n=0; n < graph->nnodes; n++) {

    int s = graph->node[n].sample;
    Z->sample[s].weight = 0.0;
    int nb_adjacent_samples = 0;

    iftVoxel u = iftMGetVoxelCoord(mimg, s);
    float sample_dist;

    for (int i = 1; i < A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v)) {
        int t = iftMGetVoxelIndex(mimg, v);

        if (iftDist == NULL)
          sample_dist = Z->iftArcWeight(Z->sample[s].feat, Z->sample[t].feat, Z->alpha, Z->nfeats);
        else
          sample_dist = iftDist->distance_table[s][t];

        if (sample_dist <= df) {
//          Z->sample[s].weight += exp(-(sample_dist*sample_dist)/K);
          Z->sample[s].weight += sample_dist;

          nb_adjacent_samples++;
          /*here we save the adjacent node */
          iftInsertAdjSet(&(graph->node[n].adj), t, sample_dist);

          if (sample_dist > graph->node[n].maxarcw)
            graph->node[n].maxarcw = sample_dist;
        }

      }
    }

//    Z->sample[s].weight /= nb_adjacent_samples;
    Z->sample[s].weight = df - (Z->sample[s].weight/nb_adjacent_samples);
  }

  for (int n=0; n < graph->nnodes; n++) {
    int s = graph->node[n].sample;
    if (Z->sample[s].weight > maximum)
      maximum = Z->sample[s].weight;
    if (Z->sample[s].weight < minimum)
      minimum = Z->sample[s].weight;
  }

  if (maximum > minimum){ /* it is mandatory to keep normalization */
#pragma omp parallel for shared(graph,Z)
    for (int u=0; u < graph->nnodes; u++){
      int s = graph->node[u].sample;
      Z->sample[s].weight = ((IFT_MAXWEIGHT - 1.0) * (Z->sample[s].weight - minimum) / (maximum - minimum)) + 1.0;
    }
  }

  // Add adjacent nodes on density plateaus if one is a neighbor of the other and the contrary doesn't happen
#pragma omp parallel for
  for (int u=0; u < graph->nnodes; u++){
    iftAdjSet   *adj_u,*adj_v=NULL;
    int s = graph->node[u].sample;
    for (adj_u=graph->node[u].adj;adj_u != NULL;adj_u=adj_u->next){
      int v = adj_u->node;
      int t = graph->node[v].sample;

      if (Z->sample[t].weight == Z->sample[s].weight){
        char adjplat=0;
        for (adj_v=graph->node[v].adj;adj_v != NULL; adj_v=adj_v->next){
          if (u == adj_v->node){
            adjplat=1;
            break;
          }
        }
        if (!adjplat)
#pragma omp critical
        {
          iftInsertSet(&(graph->node[v].adjplat),u);
        }

      }
    }
  }

  iftDestroyAdjRel(&A);
}

/* function need because we have here a proper graph type*/
int iftUnsupOPFNew(iftGraph *graph)
{
  int t,l,j,v;
  float tmp;
  iftAdjSet *adj;
  iftSet    *adjplat;
  iftDataSet *Z=graph->Z;

  // Initialization

  iftResetFHeap(graph->Q);
  iftSetRemovalPolicyFHeap(graph->Q, IFT_MAXVALUE);

  for (int u = 0; u < graph->nnodes; u++){
    int s = graph->node[u].sample;
    graph->pathval[u]     = Z->sample[s].weight-1.0;
    graph->node[u].root   = u;
    Z->sample[s].group    = 0;
    Z->sample[s].status= IFT_TRAIN;
    iftInsertFHeap(graph->Q, u);
  }

  // Optimum-Path Forest Computation
  l = 1; j = 0;
  int s,u;
  while (!iftEmptyFHeap(graph->Q)){
    u=iftRemoveFHeap(graph->Q);
    graph->ordered_nodes[j]=u; j++;
    s = graph->node[u].sample;

    if (graph->node[u].root == u){ // root node
      graph->pathval[u]    = Z->sample[s].weight;
      Z->sample[s].group   = l;
      l++;
    }

    // extend optimum paths
    for (adj=graph->node[u].adj; adj != NULL; adj = adj->next) {
      v = adj->node;
      t = graph->node[v].sample;

      if (graph->Q->color[v] != IFT_BLACK ) {
        tmp = iftMin(graph->pathval[u], Z->sample[t].weight);
        if (tmp > graph->pathval[v]){
          graph->node[v].root  = graph->node[u].root;
          Z->sample[t].group   = Z->sample[s].group;
          graph->pathval[v]    = tmp;
          iftGoUpFHeap(graph->Q, graph->Q->pos[v]);
        }
      }
    }

    adjplat = graph->node[u].adjplat;  // extend optimum paths on plateaus
    while (adjplat != NULL){
      v = adjplat->elem;
      t = graph->node[v].sample;
      if (graph->Q->color[v] != IFT_BLACK ) {
        tmp = iftMin(graph->pathval[u], Z->sample[t].weight);
        if (tmp > graph->pathval[v]){
          graph->node[v].root  = graph->node[u].root;
          Z->sample[t].group   = Z->sample[s].group;
          graph->pathval[v]    = tmp;
          iftGoUpFHeap(graph->Q, graph->Q->pos[v]);
        }
      }
      adjplat = adjplat->next;
    }

  }

  iftResetFHeap(graph->Q);

  Z->ngroups = l-1;

  return(Z->ngroups);
}

int main(int argc, char *argv[])
{
  iftImage        *img=NULL,*label=NULL,*dens=NULL,*aux=NULL;
  iftDataSet      *Z=NULL;
  iftKnnGraph     *knn_graph=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  char             ext[10],*pos;
  int              s,p,q; 
  iftVoxel         u;
  timer           *t1=NULL,*t2=NULL;
  int              norm_value,kmax;

  if (argc< 8 || argc>9)
      iftError(
              "Usage: iftClusterImageByOPFTestingPDFs <image.ppm[pgm,scn]> <nb_train_samples> <kmax(percent or value)> "
              "<di(ex. 5)> "
              "<area(0..300)> "
              "<volumen>(0..5000) "
              "<do_smoothing(1:YES/0:NO) > [<gt_image(OPTIONAL)>]",
              "main");

  iftRandomSeed(IFT_RANDOM_SEED);

  img=iftReadImageByExt(argv[1]);

  /* convert the image to multi-image*/
  iftMImage *mimg;
  iftMImage *eimg;

  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAY_CSPACE);
    if (img->zsize > 1)
      A = iftSpheric(1.0);
    else
      A = iftCircular(sqrtf(2.0));
    eimg=iftExtendMImageByAdjacency(mimg,A);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);
    mimg = eimg;
  }
  else{
    mimg = iftImageToMImage(img, LAB_CSPACE);
  }

  A=NULL;
  Z= iftMImageToDataSet(mimg);

  /* read the gt image if it was given as parameter*/
  iftImage *gt=NULL;
  if (argc==9){
    gt=iftReadImageByExt(argv[8]);
    iftImageGTToDataSet(gt,Z);
    if (Z->nclasses !=2)
      iftError("The gt image must be binary","main");
  }

  t1=iftTic();
  iftGraph *graph=NULL;

  iftImage *mask1 = iftSelectImageDomain(mimg->xsize,mimg->ysize,mimg->zsize);
  iftImage *mask_sampling = iftGridSampling(mimg, mask1,atoi(argv[2]));

  int train_samples_nb=0;
#pragma omp parallel for reduction(+:train_samples_nb)
  for (int s=0;s<Z->nsamples;s++){
    int voxel=Z->sample[s].id;
    if (mask_sampling->val[voxel]){
      Z->sample[s].status=IFT_TRAIN;
      train_samples_nb++;
    }
  }
  Z->ntrainsamples=train_samples_nb;
  iftDestroyImage(&mask1);
  iftDestroyImage(&mask_sampling);
  printf("Grid Sampling produce %d samples\n",train_samples_nb);

  if (atof(argv[3]) < 1.0)
    kmax  = (int) iftMax((atof(argv[3]) * Z->ntrainsamples), 1);
  else
    kmax = (int)iftMax(atof(argv[3]),1);
  knn_graph = iftCreateKnnGraph(Z,kmax);

//   compute the df
  iftBestkByKnnGraphCut(knn_graph, iftNormalizedCut);
  float df=knn_graph->maxarcw[knn_graph->k];
  printf("df -> %.2f , best_k -> %d\n",df,knn_graph->k);

  /*compute the pdf for the entire dataset*/
  iftSetStatus(Z,IFT_TRAIN);
  graph= iftCreateGraph(Z);
  iftPDFWithSpatialConstraint(graph,df,atof(argv[4]));
  printf("pdf computed\n");

  /*close volume of the pdf to remove irrelevant maximums*/
  iftImage *dens_sample=iftCreateImage(img->xsize,img->ysize,img->zsize);
  for (int p=0;p<Z->nsamples;p++)
    dens_sample->val[p]=Z->sample[p].weight;
  iftImage *new_dens_sample=iftVolumeOpen(dens_sample,atoi(argv[6]));
  for (int p=0;p<Z->nsamples;p++)
    Z->sample[p].weight=new_dens_sample->val[p];
  iftDestroyImage(&dens_sample);
  iftDestroyImage(&new_dens_sample);

  /* apply unsup opf to the graph*/
  iftUnsupOPFNew(graph);
  printf("graph clustered\n");

  t2     = iftToc();
  fprintf(stdout,"%s\t%d\t%.3f\t%.3f\t%d\t%.2f\t",argv[1],atoi(argv[2]),atof(argv[3]),atof(argv[4]),Z->ngroups,
          iftCompTime(t1,t2));

  label = iftDataSetClusterInformationToLabelImage(Z, false);

  if (!iftIs3DImage(img)){
    iftImage *labels_orig=iftColorizeComp(label);
    iftWriteImageP6(labels_orig,"labels_orig.ppm");
    iftDestroyImage(&labels_orig);
  }

  int do_smoothing=atoi(argv[7]);
  if (do_smoothing){
    aux=label;
    label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
    iftDestroyImage(&aux);
  }

  aux = iftSelectAndPropagateRegionsAboveArea(label,atoi(argv[5]));
  iftDestroyImage(&label);
  label=aux;
  printf("%d",iftMaximumValue(label));
  printf("\n");

  if (iftIs3DImage(img)){
    iftWriteImage(label,"label.scn");
  }
  else{
    iftWriteImageP2(label,"labels.pgm");
    aux   = iftColorizeComp(label);
    iftWriteImageP6(aux,"clusters.ppm");
    iftDestroyImage(&aux);
  }

  /*compute br and ue*/
  iftImage *border = iftBorderImage(label,0);
  if (argc ==9){
    iftImage *gt_img=iftReadImageByExt(argv[8]);
    aux=iftRelabelGrayScaleImage(gt_img,0);
    iftDestroyImage(&gt_img);
    gt_img=aux;
    iftImage *gt_borders=iftBorderImage(gt_img,0);
    printf("br -> %.4f\n",iftBoundaryRecall(gt_borders, border, 2.0));
    printf("ue -> %.4f\n",iftUnderSegmentation(gt_img, label));

    iftDestroyImage(&gt_img);
    iftDestroyImage(&gt_borders);
  }

  if (iftIs3DImage(img)){
    iftWriteImage(border, "border.scn");
  }
  else
    iftWriteImageP2(border, "border.pgm");

  iftDestroyImage(&border);

  if (!iftIs3DImage(img)){

    norm_value = iftNormalizationValue(iftMaximumValue(img));
    RGB.val[0] = 0;
    RGB.val[1] = norm_value;
    RGB.val[2] = norm_value;
    YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
    B          = iftCircular(1.5);
    aux        = iftCopyImage(img);
    for (s=0; s < knn_graph->nnodes; s++) {
      p = Z->sample[knn_graph->node[s].sample].id;
      u = iftGetVoxelCoord(img,p);
      iftDrawPoint(aux,u,YCbCr,B,255);
    }
    iftWriteImageP6(aux,"samples.ppm");
    iftDestroyImage(&aux);
    iftDestroyAdjRel(&B);

    iftDestroyAdjRel(&A);
    A  = iftCircular(sqrtf(2.0));
    B          = iftCircular(0.0);
    RGB.val[0] = 0;
    RGB.val[1] = norm_value;
    RGB.val[2] = norm_value;
    YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
    aux        = iftCopyImage(img);
    iftDrawBorders(aux,label,A,YCbCr,B);

    pos = strrchr(argv[1],'.') + 1;
    sscanf(pos,"%s",ext);
    if (strcmp(ext,"png")==0)
      iftWriteImagePNG(aux,"regions.png");
    else
      iftWriteImageP6(aux,"regions.ppm");

    iftDestroyImage(&aux);
    iftDestroyAdjRel(&B);

    dens = iftDataSetWeight(Z);
    RGB.val[0] = norm_value;
    RGB.val[1] = 0;
    RGB.val[2] = 0;
    YCbCr      = iftRGBtoYCbCr(RGB,norm_value);
    B          = iftCircular(1.5);
    aux        = iftCopyImage(dens);
    int maxdens = iftMaximumValue(dens);
    int count_maxima=0;
    for (p=0; p < dens->n; p++)
      aux->val[p] = (int) (((float)dens->val[p]/maxdens)*norm_value);
    for (s=0; s < graph->nnodes; s++) {
      if (graph->node[s].root==s){
        count_maxima++;
        q = graph->Z->sample[graph->node[s].sample].id;
        u = iftGetVoxelCoord(img,q);
        iftDrawPoint(aux,u,YCbCr,B,255);
      }
    }
    printf("nb maximas -> %d\n",count_maxima);
    iftWriteImageP6(aux,"maxima.ppm");
    iftDestroyImage(&aux);
    iftDestroyAdjRel(&B);
  }

  iftDestroyAdjRel(&A);
  iftDestroyImage(&gt);
  iftDestroyImage(&img);
  iftDestroyMImage(&mimg);
  iftDestroyImage(&dens);
  iftDestroyImage(&label);
  iftDestroyDataSet(&Z);
  iftDestroyGraph(&graph);
  iftDestroyKnnGraph(&knn_graph);

  return(0);
}
