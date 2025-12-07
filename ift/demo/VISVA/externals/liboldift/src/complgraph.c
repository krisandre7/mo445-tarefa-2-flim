#include "complgraph.h"

void RandomizeComplGraph(ComplGraph *cg){
  int i,j,c,offset=0;
  
  srand((int)time(NULL));
  for(c=1; c<=cg->nclasses; c++){
    for(i=0; i<cg->nclass[c]; i++){
      j = RandomInteger(0, cg->nclass[c]-1);
      GraphNodeSwap(&cg->node[offset+j], 
		    &cg->node[offset+i]);
    }
    offset += cg->nclass[c];
  }
}

ComplGraph *RemoveSamplesFromComplGraph(ComplGraph **cg, 
					float rate){
  ComplGraph *cg0 = NULL,*cg1 = NULL,*cg2 = NULL;
  int p1,p2,j,i,c;
  int nsamples,ndata;
  int *tmp;

  cg0 = *cg;
  ndata = cg0->node[0].ndata;
  nsamples = 0;
  tmp = (int *)calloc(cg0->nclasses+1, sizeof(int));
  for(c=1; c<=cg0->nclasses; c++){
    tmp[c] = ceil(cg0->nclass[c]*rate);
    nsamples += tmp[c];
  }

  cg1 = CreateComplGraph(cg0->n-nsamples, cg0->nclasses, ndata);
  cg2 = CreateComplGraph(nsamples,        cg0->nclasses, ndata);

  for(c=1; c<=cg0->nclasses; c++){
    cg1->nclass[c] = cg0->nclass[c] - tmp[c];
    cg2->nclass[c] = tmp[c];
  }
  free(tmp);

  p1 = p2 = 0;
  for(c=1; c<=cg0->nclasses; c++){
    for(i=0; i<cg1->nclass[c]; i++){
      GraphNodeSwap(&cg0->node[p1+p2],&cg1->node[p1]);
      p1++;
    }
    for(j=0; j<cg2->nclass[c]; j++){
      GraphNodeSwap(&cg0->node[p1+p2],&cg2->node[p2]);
      p2++;
    }
  }

  //Troca cg e cg1.
  *cg = cg1;
  DestroyComplGraph(&cg0);

  return cg2;
}

ComplGraph *MergeComplGraph(ComplGraph *cg1, 
			    ComplGraph *cg2){
  ComplGraph *cg = NULL;
  int p1,p2,j,i,c;
  int ndata,nclasses,nclass;

  if(cg1==NULL && cg2==NULL) return NULL;
  if(cg1==NULL)
    return CloneComplGraph(cg2);
  if(cg2==NULL)
    return CloneComplGraph(cg1);

  ndata = cg1->node[0].ndata;
  nclasses = MAX(cg1->nclasses, cg2->nclasses);
  cg = CreateComplGraph(cg1->n+cg2->n, nclasses, ndata);

  for(c=1; c<=nclasses; c++){
    nclass = 0;
    if(c<=cg1->nclasses)
      nclass += cg1->nclass[c];
    if(c<=cg2->nclasses)
      nclass += cg2->nclass[c];
    cg->nclass[c] = nclass;
  }

  p1 = p2 = 0;
  for(c=1; c<=nclasses; c++){
    if(c<=cg1->nclasses){
      for(i=0; i<cg1->nclass[c]; i++){
	GraphNodeCopy(&cg->node[p1+p2],
		      &cg1->node[p1]);
	p1++;
      }
    }
    if(c<=cg2->nclasses){
      for(j=0; j<cg2->nclass[c]; j++){
	GraphNodeCopy(&cg->node[p1+p2],
		      &cg2->node[p2]);
	p2++;
      }
    }
  }

  return cg;
}

void ResetComplGraphError(ComplGraph *cg){
	int i;

	for (i = 0; i < cg->n; i++)
		_fast_BMapSet0(cg->node[i].flag,2);
}

void ResetComplGraph(ComplGraph *cg)
{
	int i;

	for (i = 0; i < cg->n; i++)
	{
		cg->node[i].N_r = 0;
		cg->node[i].N_w = 0;
		_fast_BMapSet0(cg->node[i].flag,0); //is_seed = 0
		_fast_BMapSet1(cg->node[i].flag,1); //is_relevant = 1
		_fast_BMapSet0(cg->node[i].flag,2); //is_error = 0
		_fast_BMapSet0(cg->node[i].flag,3); //is_outlier = 0
	}
}

void  WriteComplGraph(ComplGraph *cg, char *filename){
  char msg[512];
  FILE *fp;
  int i,c,ndata;

  fp = fopen(filename,"wb");
  if(fp == NULL){
    sprintf(msg,"Cannot save %s",filename);
    Error(msg,"WriteComplGraph");
  }

  ndata = cg->node[0].ndata;
  fwrite(&cg->n,        sizeof(int), 1, fp); 
  fwrite(&cg->nclasses, sizeof(int), 1, fp);
  fwrite(&ndata,        sizeof(int), 1, fp);

  for(c=1; c<=cg->nclasses; c++)
    fwrite(&(cg->nclass[c]), sizeof(int), 1, fp);
  
  for(i=0; i<cg->n; i++)
    fwrite(cg->node[i].data, sizeof(real), ndata, fp);

  fclose(fp);
}

ComplGraph *ReadComplGraph(char *filename){
  ComplGraph *cg = NULL;
  char msg[512];
  FILE *fp;
  int n,nclasses,ndata,i,j,c,m;

  fp = fopen(filename,"rb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"ReadComplGraph");
  }

  fread(&n,sizeof(int), 1, fp); 
  fread(&nclasses,sizeof(int), 1, fp);
  fread(&ndata,sizeof(int), 1, fp);

  cg = CreateComplGraph(n, nclasses, ndata);

  m = 0;
  for(c=1; c<=nclasses; c++){
    fread(&(cg->nclass[c]), sizeof(int), 1, fp);
    m += cg->nclass[c];
  }
  
  if(n!=m){
    sprintf(msg,"Bad or corrupted file");
    Error(msg,"ReadComplGraph");
  }
  
  c = 1;
  j = 0;
  while (c <= nclasses && cg->nclass[c] == 0) ++c;
  if (c > nclasses) {
    sprintf(msg, "Bad or corrupted file");
    Error(msg, "ReadComplGraph");
  }
  for(i=0; i<n; i++){
    cg->node[i].label = c;
    fread(cg->node[i].data, sizeof(real), ndata, fp);
    j++;
    if(j>=cg->nclass[c]){
      c++;
      while (c < nclasses && cg->nclass[c] == 0) ++c;
      j = 0;
    }
  }
  fclose(fp);

  return cg;
}

ComplGraph *CreateComplGraph(int n, 
			     int nclasses, 
			     int ndata){
  ComplGraph *cg = NULL;
  int i;

  cg = (ComplGraph *) malloc(1*sizeof(ComplGraph));
  if(cg == NULL)
    Error(MSG1,"CreateComplGraph");

  cg->n = n;
  cg->node = (GraphNode *) malloc(n*sizeof(GraphNode));
  if(cg->node == NULL)
    Error(MSG1,"CreateComplGraph");

  cg->nclasses = nclasses;
  cg->nclass = (int *) malloc((nclasses+1)*sizeof(int));
  if(cg->nclass == NULL)
    Error(MSG1,"CreateComplGraph");

  for (i = 1; i <= nclasses; i++)
	  cg->nclass[i] = 0;

  for(i=0; i<cg->n; i++){
    cg->node[i].data = (real *) malloc(ndata*sizeof(real));
    cg->node[i].ndata = ndata;
    cg->node[i].pred = NIL;
    cg->node[i].cost = REAL_MAX;
    cg->node[i].N_r= 0;
	cg->node[i].N_w= 0;
	cg->node[i].flag = BMapNew(4);
	_fast_BMapSet0(cg->node[i].flag,0);
	_fast_BMapSet1(cg->node[i].flag,1);
	_fast_BMapSet0(cg->node[i].flag,2);
	_fast_BMapSet0(cg->node[i].flag,3);
  }
    return cg;
}

void DestroyComplGraph(ComplGraph **graph){
  ComplGraph *cg = *graph;
  int i;

  if(cg == NULL)
    return;
  for(i=0; i<cg->n; i++)
  {
	  free(cg->node[i].data);
  	  BMapDestroy(cg->node[i].flag);
  }
  free(cg->node);
  free(cg->nclass);
  free(cg);
  *graph = NULL;
}

ComplGraph *CloneComplGraph(ComplGraph *cg){
  ComplGraph *clone = NULL;
  int ndata,i,c;

  ndata = cg->node[0].ndata;
  clone = CreateComplGraph(cg->n, cg->nclasses, ndata);

  for(i=0; i<cg->n; i++){
    GraphNodeCopy(&clone->node[i],
		  &cg->node[i]);
  }

  for(c=1; c<=cg->nclasses; c++)
    clone->nclass[c] = cg->nclass[c];

  return clone;
}

void ComplGraphMST(ComplGraph *cg){
  RealHeap *H;
  real *cost, dist;
  int i,p,q;
  
  /* vetor de real com os custos. */
  cost = (real *) malloc(sizeof(real)*cg->n);

  for(i=0; i<cg->n; i++)
    cost[i] = REAL_MAX;
  
  H = CreateRealHeap(cg->n+1, cost);

  cost[0] = 0.0; /* começa pelo nó 0 */
  InsertRealHeap(H, 0);

  while(!IsEmptyRealHeap(H)) {
    RemoveRealHeap(H, &p);
    /* todos os nós do grafo completo */
    for(q=0; q<cg->n; q++){ 
      if(p == q || H->color[q] == BLACK) continue;
      dist = GraphNodeDistance(&(cg->node[p]), 
			       &(cg->node[q]));
      if(dist < cost[q]){
	cg->node[q].pred = p;
	cost[q] = dist;
	if(H->color[q] == WHITE)
	  InsertRealHeap(H, q);
	else
	  GoUpRealHeap(H, H->pos[q]);
      }
    }
  }
  DestroyRealHeap(&H);

  /* atualiza os custos dos nós na estrutura 
     ComplGraph. */
  for(i=0; i<cg->n; i++)
    cg->node[i].cost = cost[i];
  
  free(cost);
}

void	ComplGraphTraining(ComplGraph *cg){
  RealHeap *H;
  real *cost, dist, aux;
  int i,j,p,q;
  
  ComplGraphMST(cg);
  //printf("MST generated.\n");
  
  /* vetor de real com os custos. */
  cost = (real *) malloc(sizeof(real)*cg->n);

  for(i=0; i<cg->n; i++){
    cost[i] = REAL_MAX;

    /* se a classe do nó i for diferente da classe 
       de seu predecessor na MST, marcar os dois nós 
       como sementes */
    j = cg->node[i].pred;
    if(j != NIL && 
       cg->node[i].label != cg->node[j].label){
	  _fast_BMapSet1(cg->node[i].flag,0);
	  _fast_BMapSet1(cg->node[j].flag,0);
    }
  }
  
  H = CreateRealHeap(cg->n+1, cost);

  /* insere as sementes na fila */
  j = 0;
  for(i=0; i<cg->n; i++){
    cg->node[i].pred = NIL;
	if(_fast_BMapGet(cg->node[i].flag,0)){
      j++;
      cost[i] = 0.0;
      InsertRealHeap(H, i);
    }
  }
  //printf("Número de sementes: %d\n", j);

  while(!IsEmptyRealHeap(H)){
    RemoveRealHeap(H, &p);
    /* todos os nós do grafo completo */
    for(q=0; q<cg->n; q++){
      if(p == q || cost[p] >= cost[q]) continue;
      aux = GraphNodeDistance(&(cg->node[p]),
			      &(cg->node[q]));
      dist = MAX(cost[p], aux);
      if(dist < cost[q]){
	cg->node[q].pred = p;
	cg->node[q].label = cg->node[p].label;
	cost[q] = dist;
	if (H->color[q] == WHITE)
	  InsertRealHeap(H, q);
	else
	  GoUpRealHeap(H, H->pos[q]);
      }
    }
  }
  DestroyRealHeap(&H);
  
  /* atualiza os custos dos nós na estrutura 
     ComplGraph. */
  for(i=0; i<cg->n; i++)
    cg->node[i].cost = cost[i];
  
  free(cost);
}

int ComplGraphTestNodeCost(ComplGraph *cg, GraphNode *node, int *p, float *cst){
  int i, j, iMin, nclasses;
  int *label; /* numero de labels diferentes */
  int *index;
  real *cost, aux, minCost;
  
  nclasses = cg->nclasses;
  minCost = REAL_MAX;
  cost = (real *) malloc(sizeof(real)*cg->n);
  for(i=0; i<cg->n; i++){
    aux = GraphNodeDistance(&(cg->node[i]), 
			    node);
    cost[i] = MAX(cg->node[i].cost, aux);
    if(cost[i] < minCost)
      minCost = cost[i];
  }
  *cst = minCost;

  label = (int *) calloc(nclasses+1, sizeof(int));
  index = (int *) calloc(nclasses+1, sizeof(int));
  
  for(i=0; i<cg->n; i++){
    if(cost[i] == minCost){
      label[cg->node[i].label]++;
      index[cg->node[i].label] = i;
    }
  }
  
  free(cost);
  
  iMin = 0;
  j = 0;
  for(i=1; i<nclasses+1; i++){
    if(label[i] > j){
      j = label[i];
      iMin = i;
    }
  }

  *p = index[iMin];
  
  free(label);
  free(index);
  
  if(iMin == 0)
    iMin = 1;
  
  return iMin;
}

int ComplGraphTestNode(ComplGraph *cg, 
			       GraphNode *node, int *p){
  int i, j, iMin, nclasses;
  int *label; /* numero de labels diferentes */
  int *index;
  real *cost, aux, minCost;
  
  nclasses = cg->nclasses;
  minCost = REAL_MAX;
  cost = (real *) malloc(sizeof(real)*cg->n);
  for(i=0; i<cg->n; i++){
    aux = GraphNodeDistance(&(cg->node[i]), 
			    node);
    cost[i] = MAX(cg->node[i].cost, aux);
    if(cost[i] < minCost)
      minCost = cost[i];
  }

  label = (int *) calloc(nclasses+1, sizeof(int));
  index = (int *) calloc(nclasses+1, sizeof(int));
  
  for(i=0; i<cg->n; i++){
    if(cost[i] == minCost){
      label[cg->node[i].label]++;
      index[cg->node[i].label] = i;
    }
  }
  
  free(cost);
  
  iMin = 0;
  j = 0;
  for(i=1; i<nclasses+1; i++){
    if(label[i] > j){
      j = label[i];
      iMin = i;
    }
  }

  *p = index[iMin];
  
  free(label);
  free(index);
  
  if(iMin == 0)
    iMin = 1;
  
  return iMin;
}

float ComplGraphEvaluating(ComplGraph *cgTraining, 
			   ComplGraph *cgEvaluating){
  int nclasses, result, c, i, nodeID;
  float Acc, **error_matrix, error;

  nclasses = cgTraining->nclasses;

  //matriz de erros ei = ei1+ei2
  //ei1 = FP    ei2 = FN
  error_matrix = (float **)calloc(nclasses+1, sizeof(float *));
  for(c=1; c<=nclasses; c++)
    error_matrix[c] = (float *)calloc(2, sizeof(float));

  for(i=0; i<cgEvaluating->n; i++){
    result = ComplGraphTestNode(cgTraining,
				&(cgEvaluating->node[i]), 
				&nodeID);
    
    if(result != cgEvaluating->node[i].label){
	  _fast_BMapSet1(cgEvaluating->node[i].flag,2);//is_error = 1
      //atualizando FP
      error_matrix[result][0]++; 
      //atualizando FN
      error_matrix[cgEvaluating->node[i].label][1]++;
      //incrementa N_w
      SetN_wValue(cgTraining, nodeID);
    }
    else{
      //incrementa N_r
      SetN_rValue(cgTraining, nodeID);
    }
  }

  for(c=1; c<=nclasses; c++){
    error_matrix[c][0] /= (float)(cgEvaluating->n - cgEvaluating->nclass[c]);
    error_matrix[c][1] /= (float)cgEvaluating->nclass[c];
  }
	
  error = 0.0;
  for(c=1; c<=nclasses; c++)
    error += (error_matrix[c][0]+error_matrix[c][1]);
  
  Acc = 1.0-(error/(2.0*nclasses));//Acurácia

  for(c=1; c<=nclasses; c++)
    free(error_matrix[c]);
  free(error_matrix);

  return Acc;
}

float ComplGraphTesting(ComplGraph *cgTraining,
			  ComplGraph *cgTesting){
  int i,c,result,nerrors,nclasses;
  int nodeID;
  float rate,**error_matrix,error;
  char msg[512];

  if(cgTraining->nclasses != cgTesting->nclasses){
    sprintf(msg,"Incompatible number of classes");
    Error(msg,"ComplGraphRetraining");
  }
  nclasses = cgTraining->nclasses;
	
  //matriz de erros ei = ei1+ei2
  //ei1 = FP    ei2 = FN
  error_matrix = (float **)calloc(nclasses+1, sizeof(float *));
  for(c=1; c<=nclasses; c++)
    error_matrix[c] = (float *)calloc(2, sizeof(float));

  nerrors = 0;
  for(i=0; i<cgTesting->n; i++){
    result = ComplGraphTestNode(cgTraining, 
				&(cgTesting->node[i]), &nodeID);
    
    if(result != cgTesting->node[i].label){
      _fast_BMapSet1(cgTesting->node[i].flag,2);//is_erro = 1
      nerrors++;
			
      error_matrix[result][0]++; //atualizando FP
      error_matrix[cgTesting->node[i].label][1]++; //atualizando FN
    }
  }

  for(c=1; c<=nclasses; c++){
		error_matrix[c][0] /= (float)(cgTesting->n - cgTesting->nclass[c]);
		error_matrix[c][1] /= (float)cgTesting->nclass[c];
  }
	
  error = 0.0;
  for(c=1; c<=nclasses; c++)
    error += (error_matrix[c][0]+error_matrix[c][1]);
  
  rate = 1.0-(error/(2.0*nclasses));//Learning Rate
  
  for(c=1; c<=nclasses; c++)
    free(error_matrix[c]);
  free(error_matrix);
  
  return rate;
}

float ComplGraphTestingReturningClassification(ComplGraph *cgTraining,
			ComplGraph *cgTesting, int *Labels){
				int i,c,result,nerrors,nclasses;
				int nodeID;
				float rate,**error_matrix,error;
				char msg[512];

				if(cgTraining->nclasses != cgTesting->nclasses){
					sprintf(msg,"Incompatible number of classes");
					Error(msg,"ComplGraphRetraining");
				}
				nclasses = cgTraining->nclasses;

				//matriz de erros ei = ei1+ei2
				//ei1 = FP    ei2 = FN
				error_matrix = (float **)calloc(nclasses+1, sizeof(float *));
				for(c=1; c<=nclasses; c++)
					error_matrix[c] = (float *)calloc(2, sizeof(float));

				nerrors = 0;
				for(i=0; i<cgTesting->n; i++){
					result = ComplGraphTestNode(cgTraining, 
							&(cgTesting->node[i]), &nodeID);
					Labels[i] = result;
    
					if(result != cgTesting->node[i].label){
						_fast_BMapSet1(cgTesting->node[i].flag,2);//is_erro = 1
						nerrors++;
			
						error_matrix[result][0]++; //atualizando FP
						error_matrix[cgTesting->node[i].label][1]++; //atualizando FN
					}
				}

				for(c=1; c<=nclasses; c++){
					error_matrix[c][0] /= (float)(cgTesting->n - cgTesting->nclass[c]);
					error_matrix[c][1] /= (float)cgTesting->nclass[c];
				}
	
				error = 0.0;
				for(c=1; c<=nclasses; c++)
					error += (error_matrix[c][0]+error_matrix[c][1]);
  
				rate = 1.0-(error/(2.0*nclasses));//Learning Rate
  
				for(c=1; c<=nclasses; c++)
					free(error_matrix[c]);
				free(error_matrix);
  
				return rate;
			}

void IdentifyGraphNodeState(ComplGraph *cg){
  int i;

  for(i = 0; i<cg->n; i++){
	  if(cg->node[i].N_r < cg->node[i].N_w)
		  _fast_BMapSet0(cg->node[i].flag,1);//seta outlier como irrelevante
  }
}

void SetN_rValue(ComplGraph *cg, 
			   int nodeID){
  while(!_fast_BMapGet(cg->node[nodeID].flag,0)){
    cg->node[nodeID].N_r += 1; 
    nodeID = cg->node[nodeID].pred;
  }
  //setando relevance_value da raiz
  cg->node[nodeID].N_r += 1; 
}

void SetN_wValue(ComplGraph *cg, 
			   int nodeID){
  while(!_fast_BMapGet(cg->node[nodeID].flag,0)){
    cg->node[nodeID].N_w += 1; 
    nodeID = cg->node[nodeID].pred;
  }
  //setando relevance_value da raiz
  cg->node[nodeID].N_w += 1; 
}

void DisplayGraphStatus(ComplGraph *cg, char *msg){
	int i, noutliers = 0, nnrelevants = 0, nerrors = 0, nrelevants = 0, nseeds = 0;

	for(i=0; i<cg->n; i++){
      if(_fast_BMapGet(cg->node[i].flag,3))
	      noutliers++;
	  if(!_fast_BMapGet(cg->node[i].flag,1))
	      nnrelevants++;
	  else	 nrelevants++;
	  if(_fast_BMapGet(cg->node[i].flag,2))
	      nerrors++;
	   if(_fast_BMapGet(cg->node[i].flag,0))
	      nseeds++;
	}

	printf("\n*****************************");
	printf("\nGraph status: %s", msg);
	printf("\nnodes number: %d",cg->n);
	printf("\nseeds number: %d|%.1f%%",nseeds,(nseeds/(float)cg->n)*100);
	printf("\noutliers number: %d|%.1f%%",noutliers,(noutliers/(float)cg->n)*100);
	printf("\nrelevants number: %d|%.1f%%",nrelevants,(nrelevants/(float)cg->n)*100);
	printf("\nnot nelevants number: %d|%.1f%%",nnrelevants,(nnrelevants/(float)cg->n)*100);
	printf("\nerrors number: %d|%.1f%%",nerrors,(nerrors/(float)cg->n)*100);
	printf("\n*****************************");
}

ComplGraph *RemoveNotRelevants(ComplGraph **cg){
  int numnrelevants = 0, ndata, i, j, k;
  ComplGraph *cg0 = NULL, *cg1 = NULL, *cg2 = NULL;
  cg0 = *cg;

  //verificando o numero de outliers, 
  //que sera o tamanho do novo grafo
  for(i = 0; i < cg0->n; i++)
    if(!_fast_BMapGet(cg0->node[i].flag,1))//if !is_relevant
      numnrelevants++;

  if(numnrelevants==0) return NULL;

  ndata = cg0->node[0].ndata;
  cg1 = CreateComplGraph(numnrelevants, cg0->nclasses, ndata);
  cg2 = CreateComplGraph(cg0->n-numnrelevants, cg0->nclasses, ndata);

  for (i = 1; i <= cg1->nclasses; i++){
	  cg1->nclass[i] = 0;
	  cg2->nclass[i] = 0;
  }

  //copiando nao relevantes de cg0 para cg1
  j = 0;
  k = 0;
  for (i = 0; i < cg0->n; i++){
	if(_fast_BMapGet(cg0->node[i].flag,1)){ //if is relevant
      GraphNodeSwap(&cg2->node[j], &cg0->node[i]);
      cg2->nclass[cg2->node[j].label]++;
      j++;
    }
    else{ //copia para o grafo de rejeitados
      GraphNodeSwap(&cg1->node[k], &cg0->node[i]);
      cg1->nclass[cg1->node[k].label]++;
      k++;
    }
  }
	
  *cg = cg2;
  DestroyComplGraph(&cg0);

  return cg1;
}

void SwapNotRelevantsbySamples(ComplGraph *cgTraining, 
			 ComplGraph *cgEvaluating){
  int *offset=NULL,*end=NULL;
  int i,j,c,numnrelevants,nclasses;
  bool flag;

  nclasses = cgEvaluating->nclasses;
  
  offset = (int *)calloc(nclasses+1, sizeof(int));
  end    = (int *)calloc(nclasses+1, sizeof(int));
  for(c=2; c<=nclasses; c++)
    offset[c] = offset[c-1] + cgEvaluating->nclass[c-1];

  for(c=1; c<=nclasses; c++)
    end[c] = offset[c] + cgEvaluating->nclass[c];

  numnrelevants = 0;
  for(i=0; i<cgTraining->n; i++)//contando numero de irrelevantes
    if(!_fast_BMapGet(cgTraining->node[i].flag,1))
      numnrelevants++;
  
  for(i=0; i<cgTraining->n; i++){
	  if(numnrelevants == 0)
		  break;

	  if(!_fast_BMapGet(cgTraining->node[i].flag,1)){
		  c = cgTraining->node[i].label;
		  flag = false;

		  for(j=offset[c]; j<end[c]; j++){
			  /*verifica se a amostra a ser trocada em Z2 e relevante, pois caso seja irrelevante significa que ela ja estava em Z1 nessa mesma iteracao
			  a unica condicao de uma amostra ir para Z2 e ela sendo irrelevante*/
			  if(_fast_BMapGet(cgEvaluating->node[j].flag,1)){
				flag = true;
				break;
			  }
		  }

		  offset[c] = j;

		  if(flag){
			  
			  // faz a troca SOMENTE ELEMENTOS DE MESMA CLASSE
			  GraphNodeSwap(&cgTraining->node[i], &cgEvaluating->node[j]);

			  //elemento de Z2 entra como relevante em Z1
			  _fast_BMapSet1(cgTraining->node[i].flag,1); //is_relevant = 1

			  numnrelevants--;
			  offset[c]++;
		  }
	  }
  }
	
 
  free(offset);
  free(end);
}

void SwapErrorbyNotRelevant(ComplGraph *cgTraining, 
			    ComplGraph *cgEvaluating){
  int *offset=NULL,*end=NULL;
  int i,j,c,numnrelevants,nclasses;
  bool flag;

  nclasses = cgTraining->nclasses;

  offset = (int *)calloc(nclasses+1, sizeof(int));
  end    = (int *)calloc(nclasses+1, sizeof(int));
  for(c=2; c<=nclasses; c++)
    offset[c] = offset[c-1] + cgTraining->nclass[c-1];
  
  for(c=1; c<=nclasses; c++)
    end[c] = offset[c] + cgTraining->nclass[c];

  numnrelevants = 0;
  for(i=0; i<cgTraining->n; i++)
	  if(!_fast_BMapGet(cgTraining->node[i].flag,1))//if !is relevant
			numnrelevants++;

  for(i=0; i<cgEvaluating->n; i++){
	  if(numnrelevants == 0)
		  break;

	  //if is error and relevant e pq ele nunca esteve em Z1
	  if((_fast_BMapGet(cgEvaluating->node[i].flag,2)) && (_fast_BMapGet(cgEvaluating->node[i].flag,1))){
		  c = cgEvaluating->node[i].label;
		  flag = false;

		  for(j=offset[c]; j<end[c]; j++){
			  //if((Lc[cgTraining->node[j].label] > 0) && (!_fast_BMapGet(cgTraining->node[j].flag,1)) &&((!_fast_BMapGet(cgTraining->node[j].flag,0)) || 
			  if(!_fast_BMapGet(cgTraining->node[j].flag,1)){//verifica se o no em Z1 e irrelevante
				  flag = true;
				  break;
			  }
		  }

		  offset[c] = j;

		  if(flag){
			  // faz a troca SOMENTE ELEMENTOS DE MESMA CLASSE
			  GraphNodeSwap(&cgTraining->node[j], &cgEvaluating->node[i]);

			  //seta erro para zero
			  _fast_BMapSet0(cgEvaluating->node[i].flag,2); //is_error = 0

			  //elemento errado entra como relevante
			  _fast_BMapSet1(cgTraining->node[j].flag,1); //is_relevant = 1

			  //atualiza capacidade de treinamento
			  numnrelevants--;
			  offset[c]++;
		  }
	  }
  }
  
  free(offset);
  free(end);
}

void SwapErrorbyNotPrototypes(ComplGraph *cgTraining, ComplGraph *cgEvaluating){
				      int *offset=NULL,*end=NULL;
				      int i,j,c,numnprototypes,nclasses;
				      bool flag;

				      nclasses = cgTraining->nclasses;

				      offset = (int *)calloc(nclasses+1, sizeof(int));
				      end    = (int *)calloc(nclasses+1, sizeof(int));
				      for(c=2; c<=nclasses; c++)
					      offset[c] = offset[c-1] + cgTraining->nclass[c-1];
  
				      for(c=1; c<=nclasses; c++)
					      end[c] = offset[c] + cgTraining->nclass[c];

				      numnprototypes = 0;
				      for(i=0; i<cgTraining->n; i++)
					      if(!_fast_BMapGet(cgTraining->node[i].flag,0))//if !is seed
						      numnprototypes++;

				      for(i=0; i<cgEvaluating->n; i++){
					      if(numnprototypes == 0)
						      break;

	  //if is error
					      if(_fast_BMapGet(cgEvaluating->node[i].flag,2)){
						      c = cgEvaluating->node[i].label;
						      flag = false;

						      for(j=offset[c]; j<end[c]; j++){
							      if(!_fast_BMapGet(cgTraining->node[j].flag,0)){//verifica se o no em Z1 não e prototipo
								      flag = true;
								      break;
							      }
						      }

						      offset[c] = j;

						      if(flag){
			  // faz a troca SOMENTE ELEMENTOS DE MESMA CLASSE
							      GraphNodeSwap(&cgTraining->node[j], &cgEvaluating->node[i]);

			  //seta erro para zero
							      _fast_BMapSet0(cgEvaluating->node[i].flag,2); //is_error = 0

			  //elemento errado entra como protipo
							      _fast_BMapSet1(cgTraining->node[j].flag,0); //is_seed = 1

			  //atualiza capacidade de treinamento
							      numnprototypes--;
							      offset[c]++;
						      }
					      }
				      }
  
				      free(offset);
				      free(end);
}

void SwapErrorbySamples(ComplGraph *cgTraining, ComplGraph *cgEvaluating){
	int *offset=NULL,*end=NULL;
	int i,j,c,nclasses;

	nclasses = cgTraining->nclasses;

	offset = (int *)calloc(nclasses+1, sizeof(int));
	end    = (int *)calloc(nclasses+1, sizeof(int));
	for(c=2; c<=nclasses; c++)
		offset[c] = offset[c-1] + cgTraining->nclass[c-1];
  
     for(c=1; c<=nclasses; c++)
		 end[c] = offset[c] + cgTraining->nclass[c];

     for(i=0; i<cgEvaluating->n; i++){
		 //if is error
		 if(_fast_BMapGet(cgEvaluating->node[i].flag,2)){
			 c = cgEvaluating->node[i].label;
			 j=offset[c];

			if(j < end[c]){
				// faz a troca SOMENTE ELEMENTOS DE MESMA CLASSE
				GraphNodeSwap(&cgTraining->node[j], &cgEvaluating->node[i]);

				//seta erro para zero
				_fast_BMapSet0(cgEvaluating->node[i].flag,2); //is_error = 0

				offset[c]++;
			}
		}
	}

	free(offset);
	free(end);
}

ComplGraph *RemoveOutliers(ComplGraph **cg){
  int numoutlier = 0, ndata, i, j, k;
  ComplGraph *cg0 = NULL, *cg1 = NULL, *cg2 = NULL;
  cg0 = *cg;

  //verificando o numero de outliers, 
  //que sera o tamanho do novo grafo
  for(i = 0; i < cg0->n; i++)
    if(_fast_BMapGet(cg0->node[i].flag,3))//if is_oulier
      numoutlier++;

  if(numoutlier==0) return NULL;

  ndata = cg0->node[0].ndata;
  cg1 = CreateComplGraph(numoutlier, cg0->nclasses, ndata);
  cg2 = CreateComplGraph(cg0->n-numoutlier, cg0->nclasses, ndata);

  for (i = 1; i <= cg1->nclasses; i++){
	  cg1->nclass[i] = 0;
	  cg2->nclass[i] = 0;
  }

  //copiando nao outliers de cg0 para cg1
  j = k = 0;
  for (i = 0; i < cg0->n; i++){
	if(!_fast_BMapGet(cg0->node[i].flag,3)){ //if !is_oulier
      GraphNodeCopy(&cg2->node[j], &cg0->node[i]);
      cg2->nclass[cg2->node[j].label]++;
      j++;
    }
    else{ //copia para o grafo de rejeitados
      GraphNodeCopy(&cg1->node[k], &cg0->node[i]);
      cg1->nclass[cg1->node[k].label]++;
      k++;
    }
  }
	
  *cg = cg2;
  DestroyComplGraph(&cg0);

  return cg1;
}

void ComplGraphLearning(ComplGraph **cgTr, ComplGraph **cgEval, int iterations, char *FileName){
	int i;
	float AccZ2;
	ComplGraph *cgTraining = NULL, *cgEvaluating = NULL;
	FILE *fp = NULL;

	fp = fopen(FileName,"w");

	cgTraining = *cgTr;
	cgEvaluating = *cgEval;
  
	for (i = 1; i<= iterations; i++){
	  printf("\nrunning iteration %d ...",i);	

	  //reseta grafos
  	  ResetComplGraph(cgTraining); 	  
	  ResetComplGraphError(cgEvaluating);

	  //treina conjunto Z1
	  ComplGraphTraining(cgTraining); 

	  //aplicando teste
	  AccZ2 = ComplGraphTesting(cgTraining, cgEvaluating); //avalia conjunto Z2
	  
	  //trocando erros em Z2 por não protótipos de Z1
	  SwapErrorbyNotPrototypes(cgTraining, cgEvaluating);

	  fprintf(fp,"%d	%f\n",i,AccZ2);
	  printf(" AccZ2: %f",AccZ2);
	}

	//treina conjunto Z1
	ComplGraphTraining(cgTraining); 

	fclose(fp);

	*cgTr = cgTraining;
	*cgEval = cgEvaluating;
}

/*MERCY ESTA EM ESTUDOS **********************************************************
//void ComplGraphMercy(ComplGraph **cgTr, ComplGraph **cgEval){
	void ComplGraphMercy(ComplGraph *cgTraining, ComplGraph *cgEvaluating){
	int i, j = 0;
	bool OK ;
	//ComplGraph *cgAux = NULL;
	//ComplGraph *cgTraining = NULL, *cgEvaluating = NULL;
	//	ComplGraph *cgAux = NULL;
	//ComplGraph *cgTraining = NULL, *cgEvaluating = NULL;
	
	//cgTraining = *cgTr;
	//cgEvaluating = *cgEval;

	//primeiro tenta trocar elemento da classe i de Z2 com elemento da classe j de Z1
	for (i = 0; i < cgEvaluating->n; i++)
	{
		OK = 0;
		if(_fast_BMapGet(cgEvaluating->node[i].flag,2)){//if is erro
			while((!OK) && (j < cgTraining->n)){//tenta procurar alguem de Z1 para substitui-lo
				if(!_fast_BMapGet(cgTraining->node[j].flag,1)){//if !is relevant
					GraphNodeSwap(&cgTraining->node[j], &cgEvaluating->node[i]);
					_fast_BMapSet1(cgTraining->node[j].flag,1);//entra em Z1 como relevante
					_fast_BMapSet0(cgEvaluating->node[i].flag,2);//entra em Z2 com erro 0
					//atualizando nclasses
					cgTraining->nclass[cgTraining->node[j].label]++;
					cgTraining->nclass[cgEvaluating->node[i].label]--;
					cgEvaluating->nclass[cgEvaluating->node[i].label]++;
					cgEvaluating->nclass[cgTraining->node[j].label]--;

					OK = 1;
					j++;
				}
				else j++;
			}
			if(j == cgTraining->n) break;
		}
	}

	 cgAux = RemoveNotRelevants(&cgTraining);
	 DestroyComplGraph(&cgAux);

	 *cgTr = cgTraining;
	 *cgEval = cgEvaluating;
}
*****************************************************************************************************/

void ComplGraphSortLabel(ComplGraph **cg, char order)
{
	  ComplGraph *cgAux = NULL, *cgSort = NULL;
	  Curve *cur = NULL;
	  int i;
	  cgAux = *cg;
	  
	  if(cgAux == NULL) exit(-1);

	  cur = CreateCurve(cgAux->n+1);

	  for (i = 0; i < cgAux->n; i++)
	  {
		  cur->X[i] = i;
		  cur->Y[i] = cgAux->node[i].label;
	  }

	  SortCurve(cur, 0,(cur->n)-2,order);
	  cgSort = CreateComplGraph(cgAux->n, cgAux->nclasses, cgAux->node[0].ndata);

	  for (i = 1; i <= cgAux->nclasses; i++)
		  cgSort->nclass[i] = cgAux->nclass[i];

	  for (i = 0; i< cgSort->n; i++)
		  GraphNodeCopy(&cgSort->node[i], &cgAux->node[(int)cur->X[i]]);
		
	  DestroyCurve(&cur);
	  *cg = cgSort;
   	  DestroyComplGraph(&cgAux);
}

ComplGraph *OPF(char *descriptorFileName, 
		char *accZ2FileName, float TrRate, float EvalRate, float TsRate, int iterations, char *accZ3FileName, int running){
  ComplGraph *cgTraining = NULL, *cgTesting = NULL, *cgEvaluating = NULL;
  float AccZ3;
  //timer tic, toc; 
  char msg[512];
  FILE *fp = NULL;

  printf("%f",TrRate+EvalRate+TsRate);
  if((TrRate+EvalRate+TsRate) != 1.0)
  {
	  sprintf(msg,"\nWrong set percentage");
	  Error(msg,"OPF");
  }

  //gettimeofday(&tic,NULL);

  //reading sets
  cgTesting = ReadComplGraph(descriptorFileName);
  RandomizeComplGraph(cgTesting);
  cgEvaluating = RemoveSamplesFromComplGraph(&cgTesting, TrRate+EvalRate);
  cgTraining = RemoveSamplesFromComplGraph(&cgEvaluating, TrRate/(float)(TrRate+EvalRate));

  //executa aprendizado
  ComplGraphLearning(&cgTraining, &cgEvaluating, iterations, accZ2FileName);

  //executa golpe de misericordia
  //ComplGraphMercy(&cgTraining, &cgEvaluating);

  //testa em Z3
  AccZ3 = ComplGraphTesting(cgTraining, cgTesting);
  printf("\nAccZ3 = %f",AccZ3);

  fp = fopen(accZ3FileName, "a");
  fprintf(fp,"%d %f \n",running, AccZ3);
  fclose(fp);

  DestroyComplGraph(&cgEvaluating);
  DestroyComplGraph(&cgTesting);

  //gettimeofday(&toc,NULL);
  //totaltime = (toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001;
  //printf("\nTime: %f ms\n",totaltime);

  return cgTraining;
}
