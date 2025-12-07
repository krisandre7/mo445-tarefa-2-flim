#ifndef _COMPLGRAPH_H_
#define _COMPLGRAPH_H_

//#include <values.h>
#include <time.h>
#include "common.h"
#include "curve.h"
#include "comptime.h"
#include "realheap.h"
#include "scene.h"
#include "set.h"

typedef struct _graphnode {
  real *data;
  int   ndata;

  int   label;
  int   pred;
  real  cost;
  int N_r;
  int N_w;
  BMap *flag; 
  /*bitmap de flags (0 - FALSE 1 - TRUE)
  posicao	significado
		0			is_seed
		1			is_relevant
		2			is_error
		3			is_outlier
*/
} GraphNode;


typedef struct _complgraph {
  int n;
  GraphNode *node;

  int nclasses;
  int *nclass;

} ComplGraph;


/*Functions that works with graphs	*************************************************/

/*Todas funcoes abaixo assumem que os n� do
grafo sao armazenados de forma ordenada pelo label. */

/*ordena um grafo de acordo com suas labels
order =
			INCREASING (crescente)
			DECREASING(decrescente)
*/
void ComplGraphSortLabel(ComplGraph **cg, char order);

//Embaralha os n� de uma mesma classe de forma aleatoria, preservando a ordenacao dos labels.
void RandomizeComplGraph(ComplGraph *cg);

//Reseta as configuracoes de um grafo
void ResetComplGraph(ComplGraph *cg);


//Reseta o erro dos nos de um grafo
void ResetComplGraphError(ComplGraph *cg);

/*Grava o "ComplGraph" para disco em formato binario onde os vetores de caracteristicas sao gravados de 
forma sequencial. O cabecalho armazena o tamanho dos vetores, bem como o numero de classes e o numero
de registros por classe.*/
void  WriteComplGraph(ComplGraph *cg, char *filename);

//Libera instancia de um grafo
void DestroyComplGraph(ComplGraph **graph);

// Calcula a �vore geradora m�ima de um grafo
void ComplGraphMST(ComplGraph *cg);

/* Treina um ComplGraph. �calculada a �vore geradora m�ima, e os n� pai e filho com 
   labels diferentes viram sementes de um watershed. */
void	ComplGraphTraining(ComplGraph *cg);

//Identifica status do no: outlier ou irrelevante
void IdentifyGraphNodeState(ComplGraph *cg);

//Incrementa numero de classificacoes corretas de um no (seu caminho ate sua raiz)
void SetN_rValue(ComplGraph *cg, int nodeID);

//Incrementa numero de classificacoes incorretas de um no (seu caminho ate sua raiz)
void SetN_wValue(ComplGraph *cg, int nodeID);

//Troca erros de Z2 por irrelevantes de Z1
void SwapErrorbyNotRelevant(ComplGraph *cgTraining, ComplGraph *cgEvaluating);

//Troca amostras quaisquer de Z2 por irrelevantes de Z1
void SwapNotRelevantsbySamples(ComplGraph *cgTraining, ComplGraph *cgEvaluating);

//Troca erros de Z2 por não protótipos de Z1
void SwapErrorbyNotPrototypes(ComplGraph *cgTraining, ComplGraph *cgEvaluating);

//Troca erros de Z2 por amostras de Z1
void SwapErrorbySamples(ComplGraph *cgTraining, ComplGraph *cgEvaluating);

//Mostra status do grafo (porcentagem e numeros de nos, outliers, relevantes, irrelevantes e erros
void DisplayGraphStatus(ComplGraph *cg, char *msg);

//aplica golpe de misericordia EM ESTUDOS ******************************
//void ComplGraphMercy(ComplGraph *cgTr, ComplGraph *cgEval);

//Executa o algoritmo de treinamento
void ComplGraphLearning(ComplGraph **cgTr, ComplGraph **cgEval, int iterations, char *FileName);

//Copia de nos
static inline void GraphNodeCopy(GraphNode *dest, GraphNode *src){
  memcpy(dest->data,
	 src->data, 
	 src->ndata*sizeof(real));
  dest->ndata = src->ndata;
  dest->label = src->label;
  dest->pred  = src->pred;
  dest->cost  = src->cost;
  dest->N_r = src->N_r;
  dest->N_w = src->N_w;
  BMapCopy(dest->flag,src->flag);
}

//Faz a troca entre nos
static inline void GraphNodeSwap(GraphNode *a, GraphNode *b){
  GraphNode tmp;
  
  tmp = *a;
  *a = *b;
  *b = tmp;
}


//Le o "ComplGraph" a partir de um arquivo na forma binaria.
ComplGraph *ReadComplGraph(char *filename);

/*Retorna um subconjunto do grafo "cg" com amostras de todas classes na proporcao dada por "rate".
O subconjunto retornado eh removido do grafo "cg" original. A funcao "RandomizeComplGraph" deve ser chamada
previamente para garantir amostras aleatorias.*/
ComplGraph *RemoveSamplesFromComplGraph(ComplGraph **cg, float rate);

//Junta dois complete graphs. A ordenacao dos labels eh preservada.
ComplGraph *MergeComplGraph(ComplGraph *cg1, ComplGraph *cg2);

//Cria um grafo
ComplGraph *CreateComplGraph(int n, int nclasses, int ndata);

//Clona um grafo
ComplGraph *CloneComplGraph(ComplGraph *cg);

//Remove outliers
ComplGraph *RemoveOutliers(ComplGraph **cg);

//Remove irrelevantes
ComplGraph *RemoveNotRelevants(ComplGraph **cg);

//Executa algoritmo OPF
//Parametros
//P1  - descriptorFileName: nome do arquivo que contem o descritor da base de dados
//P2 - accZ2FileName: nome do arquivo que ira armazenar a acuracia em Z2 (curva de aprendizado, uma para cada iteration)
//P3 - TrRate: porcentagem do conjunto de treinamento (Z1)
//P5 - EvalRate: porcentagem do conjunto de avaliacao (Z2)
//P6 - TsRate: porcentagem do conjunto de teste (Z3)
//P7 - iterations: numero de iteracoes para se obter a curva de aprendizado
//P8 - running: numero de rodadas do algoritmo
//P9 - accZ3FileName: nome do arquivo que ira armazenar a acuracia em Z3 (uma para cada running)
ComplGraph *OPF(char *descriptorFileName, char *accZ2FileName, float TrRate, float EvalRate, float TsRate, int iterations, char *accZ3FileName, int running);

//Calcula a distacia euclidiana entre 2 nos
static inline real GraphNodeDistance(GraphNode *a, GraphNode *b){
  real sum = 0.0f;
  int i;

  for(i=0; i<a->ndata; i++)
    sum += (b->data[i]-a->data[i])*(b->data[i]-a->data[i]);
  return (real)sqrtf(sum);
}


//Classifica um no retornando sua label
int ComplGraphTestNode(ComplGraph *cg, GraphNode *node, int *p);

/*Classifica nos do conjunto Evaluating (classificacao complexa)
Essa classificacao calcula, para cada no, o numero de acertos e erros*/
float ComplGraphEvaluating(ComplGraph *cgTraining, ComplGraph *cgEvaluating);

//Calcula capacidade de treinamento das classes de um grafo
float *ComplGraphCalculateLearningCapacity(ComplGraph *cg);

/*************************************************************************************/
//Remove outliers
ComplGraph *RemoveOutliers(ComplGraph **cg);

//Remove irrelevantes
ComplGraph *RemoveNotRelevants(ComplGraph **cg);

//Classifica um no retornando seu label
int ComplGraphTestNode(ComplGraph *cg, GraphNode *node, int *p);

//Classifica um no retornando seu label e o custo do caminho otimo
int ComplGraphTestNodeCost(ComplGraph *cg, GraphNode *node, int *p, float *cst);

/*Classifica nos do conjunto Evaluating (classificacao complexa)
Essa classificacao calcula, para cada no, o numero de acertos e erros*/
float ComplGraphEvaluating(ComplGraph *cgTraining, ComplGraph *cgEvaluating);

//Classifica nos do conjunto Testing (classificacao simples) 
float ComplGraphTesting(ComplGraph *cgTraining, ComplGraph *cgTesting);

//Classifica nos do conjunto Testing (classificacao simples), porém retorna as labels associadas durante a classificacao
float ComplGraphTestingReturningClassification(ComplGraph *cgTraining,
					       ComplGraph *cgTesting, int *Labels);
/*************************************************************************************/

#endif
