
#include "myopf3.h"

Features3 *my_LMSSceneFeats(Scene *scn, int nscales){
  Features3 *f=(Features3 *)calloc(1,sizeof(Features3));
  Scene *aux,*blur;
  int p,s;

  f->Imax = MaximumValue3(scn);
  f->xsize  = scn->xsize;
  f->ysize  = scn->ysize;
  f->zsize  = scn->zsize;
  f->nelems = f->xsize*f->ysize*f->zsize;
  f->nfeats = nscales;
  f->elem   = (FElem *)calloc(f->nelems,sizeof(FElem));
  for(p=0; p<f->nelems; p++)
    f->elem[p].feat = AllocFloatArray(f->nfeats);

  aux = scn;
  for(s=0; s<nscales; s++){

    for(p=0; p<f->nelems; p++)
      f->elem[p].feat[s] = (float)aux->data[p]/f->Imax;

    if(s+1<nscales){
      blur = GaussianBlur3(aux);
      if(aux!=scn) DestroyScene(&aux);
      aux = blur;
    }
    else if(aux!=scn) DestroyScene(&aux);
  }

  return f;
}


// Compute node dist density for classification node 
void NodeDD(Subgraph *sg, CNode *node){
  int     i,nelems;
  float   dist;
  double  value;
  
  value=0.0;nelems=1;
  for (i=0; i < sg->bestk; i++) {
    dist = opf_EuclDist(node->feat,sg->node[node->adj[i]].feat,sg->nfeats); 
    value += dist;
    nelems++;
  }
  value = value/(float)nelems;
  node->dens = (value);
}


int my_Findkmax(Subgraph *sgTrain){
  int *label=AllocIntArray(sgTrain->nlabels);
  int i,kmax=INT_MAX;
  
  for(i=0; i<sgTrain->nnodes; i++)
    label[sgTrain->node[i].truelabel]++;

  for(i=0; i<sgTrain->nlabels; i++)
    kmax = MIN(kmax,label[i]);
  
  free(label);
  return MAX(kmax/2,1);
}


float MaxEuclDist3(Features3 *f){
  float  dist;
  float *fv1,*fv2;
  int p,i;

  fv1 = AllocFloatArray(f->nfeats);
  fv2 = AllocFloatArray(f->nfeats);
  for(i=0; i<f->nfeats; i++){
    fv1[i] = FLT_MAX;
    fv2[i] = -FLT_MAX;
  }

  for(p=0; p<f->nelems; p++){
    for(i=0; i<f->nfeats; i++){
      fv1[i] = MIN(fv1[i], f->elem[p].feat[i]);
      fv2[i] = MAX(fv2[i], f->elem[p].feat[i]);
    }
  }

  dist = opf_EuclDist(fv1, fv2, f->nfeats);
  free(fv1);
  free(fv2);

  return dist;
}



// Estimate the best k
int BestkByMaxContrast(Subgraph *sgTrain,
		       Subgraph *sgEval, int kmax){
  int k;
  float maxcontr=-FLT_MAX,contr;

  // Find the best k
  for(k=1; (k <= kmax); k++){
    sgTrain->bestk=k;

    contr = EvaluateProbContrast(sgTrain,sgEval);

    if(contr >= maxcontr){
      maxcontr = contr;
      sgEval->bestk = k;
    }
  }
  sgTrain->bestk=sgEval->bestk;
  return sgTrain->bestk;
}


float EvaluateProbContrast(Subgraph *sgTrain, Subgraph *sgEval){
  CNode *node=CreateCNode(sgTrain);
  int   i,p,nobj=0,nbkg=0;
  float maxdist,prob,Pobj=0.0,Pbkg=0.0;
  float ddf0,ddf1;

  maxdist = sgTrain->maxdens;
  for(p=0; p<sgEval->nnodes; p++){
    for(i=0; i<sgEval->nfeats; i++)
      node->feat[i]=sgEval->node[p].feat[i];

    NodeArcsByTrueLabel(sgTrain, node, 0);
    NodeDD(sgTrain,node);
    ddf0 = node->dens;

    NodeArcsByTrueLabel(sgTrain, node, 1);
    NodeDD(sgTrain,node);
    ddf1 = node->dens;

    prob = DDF2ObjProbability_2(ddf0, ddf1, maxdist);

    if(sgEval->node[p].truelabel==0){
      Pbkg += prob;
      nbkg ++;
    }
    else{
      Pobj += prob;
      nobj ++;
    }
  }
  DestroyCNode(&node);

  Pobj /= (float)nobj;
  Pbkg /= (float)nbkg;
  return(Pobj-Pbkg);
}



float DDF2ObjProbability_1(float dist0, float dist1){
  float prob;

  if((dist0+dist1)<0.00001)
    prob = 0.5;
  else
    prob = dist0/(dist0+dist1);

  return prob;
}



float DDF2ObjProbability_2(float dist0, float dist1, float maxdist){
  float prob,d,ratio=0.02; //2%

  d = maxdist*(ratio*ratio);
  prob = (dist0+d)/(dist0+dist1+2.0*d);

  return prob;
}



void my_Learning(Subgraph **sgtrain, Subgraph **sgeval, 
		 int iterations, int kmax){
  int i;
  float Acc,MaxAcc=-FLT_MAX;
  Subgraph *sg=NULL;

  for(i=1; (i <= iterations)&&(MaxAcc != 1.0); i++){

    BestkByMaxContrast(*sgtrain,*sgeval, kmax);
    Acc = EvaluateProbContrast(*sgtrain, *sgeval);

    printf("\n running iteration ... %d ... Acc: %f\n",i,Acc);
    if(Acc > MaxAcc){
      MaxAcc = Acc;
      if(sg!=NULL) DestroySubgraph(&sg);
      sg = CopySubgraph(*sgtrain);	    
    }
    SwapAllAtRandom(sgtrain, sgeval);
  }
  DestroySubgraph(&(*sgtrain));
  *sgtrain = sg;
  printf("bestk %d MaxAcc %f\n",(*sgtrain)->bestk,MaxAcc);
}



//Replace data from training set by randomly samples from evaluating set
void SwapAllAtRandom(Subgraph **sgtrain, Subgraph **sgeval){
  int i,j;

  for(i=0; i<(*sgtrain)->nnodes; i++){
    while(1){
      j = RandomInteger(0,(*sgeval)->nnodes-1);
      if((*sgtrain)->node[i].truelabel == (*sgeval)->node[j].truelabel){
	SwapSNode(&((*sgtrain)->node[i]), &((*sgeval)->node[j]));
	break;
      }
    }
  }
}


float *my_ProbabilityArray3(Subgraph *sg, Features3 *f){
  CNode *node=CreateCNode(sg);
  int   i,p;
  float maxdist,*A_prob=NULL;
  float ddf0,ddf1;

  if(sg==NULL || f==NULL)
    return NULL;

  A_prob = AllocFloatArray(f->nelems);
  maxdist = sg->maxdens;
  for(p=0; p<f->nelems; p++){
    for(i=0; i<f->nfeats; i++)
      node->feat[i]=f->elem[p].feat[i];

    NodeArcsByTrueLabel(sg, node, 0);
    NodeDD(sg,node);
    ddf0 = node->dens;

    NodeArcsByTrueLabel(sg, node, 1);
    NodeDD(sg,node);
    ddf1 = node->dens;

    A_prob[p] = DDF2ObjProbability_2(ddf0, ddf1, maxdist);
  }
  DestroyCNode(&node);
  return A_prob;
}


Subgraph *my_BestkSubgraph3(Features3 *f, Set *Si, Set *Se, int nsamples){
  Subgraph *sg=NULL,*sgtrain,*sgeval;
  Set *aux;
  int ni,ne,i,kmax;
  float perc;

  ni = GetSetSize(Si);
  ne = GetSetSize(Se);

  if ((ni==0)||(ne==0))
    return(NULL);

  //--------CreateSubGraph-----------------

  sg=(Subgraph *)calloc(1,sizeof(Subgraph));  
  sg->nnodes  = ni+ne;
  sg->nlabels = 2;
  sg->node    = (SNode *)calloc(sg->nnodes,sizeof(SNode));
  if(sg->node == NULL)
    Error("Cannot allocate nodes","my_BestkSubgraph3");

  i = 0;
  aux = Si;
  while(aux != NULL){
    sg->node[i].position  = aux->elem;
    sg->node[i].truelabel = 1;
    sg->node[i].adj=NULL;
    sg->node[i].status=0; // training node 
    sg->node[i].feat=NULL;
    aux = aux->next;
    i++;
  }
  aux = Se;
  while(aux != NULL){
    sg->node[i].position  = aux->elem;
    sg->node[i].truelabel = 0;
    sg->node[i].adj=NULL;
    sg->node[i].status=0; // training node 
    sg->node[i].feat=NULL;
    aux = aux->next;
    i++;
  }

  //---------------------------------------

  SetSubgraphFeatures3(sg,f);

  if (sg->nnodes > nsamples){
    perc = (float)nsamples/(float)sg->nnodes;
  }else
    perc = 0.5;

  opf_SplitSubgraph(sg,&sgtrain,&sgeval,perc);
  DestroySubgraph(&sg);
  kmax = my_Findkmax(sgtrain);
  printf("Computed kmax: %d\n",kmax);

  sgtrain->maxdens = MaxEuclDist3(f);
  my_Learning(&sgtrain,&sgeval,3, kmax);
  DestroySubgraph(&sgeval);

  return(sgtrain);
}


