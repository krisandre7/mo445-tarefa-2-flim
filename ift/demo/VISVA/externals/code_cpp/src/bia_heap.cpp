
#include "bia_heap.h"

namespace bia{
  namespace Heap{

    inline int GetDad(int i){
      return ((i - 1) / 2);
    }

    inline int GetLeftSon(int i){
      return (2 * i + 1);
    }

    inline int GetRightSon(int i){ 
      return (2 * i + 2);
    }

    void SetTheRemovalPolicy(Heap *H, char policy){
      if(H->removal_policy != policy){
	H->removal_policy = policy;
	Reset(H);
      }
    }

    void GoUp(Heap *H, int i) {
      int j = GetDad(i);

      if(H->removal_policy == MINVALUE){
	
	while((i > 0) && (H->cost[H->spel[j]] > H->cost[H->spel[i]])) {
	  bia::SwapInt(&H->spel[j], &H->spel[i]);
	  H->pos[H->spel[i]] = i;
	  H->pos[H->spel[j]] = j;
	  i = j;
	  j = GetDad(i);
	}
      }
      else{ /* removal_policy == MAXVALUE */
	
	while ((i > 0) && (H->cost[H->spel[j]] < H->cost[H->spel[i]])) {
	  bia::SwapInt(&H->spel[j], &H->spel[i]);
	  H->pos[H->spel[i]] = i;
	  H->pos[H->spel[j]] = j;
	  i = j;
	  j = GetDad(i);
	}
      }
    }

    void GoDown(Heap *H, int i) {
      int j, left = GetLeftSon(i), right = GetRightSon(i);
      
      j = i;
      if(H->removal_policy == MINVALUE){
	
	if ((left <= H->last) && 
	    (H->cost[H->spel[left]] < H->cost[H->spel[i]]))
	  j = left;
	if ((right <= H->last) && 
	    (H->cost[H->spel[right]] < H->cost[H->spel[j]]))
	  j = right;
      }
      else{ /* removal_policy == MAXVALUE */
	
	if ((left <= H->last) && 
	    (H->cost[H->spel[left]] > H->cost[H->spel[i]]))
	  j = left;
	if ((right <= H->last) && 
	    (H->cost[H->spel[right]] > H->cost[H->spel[j]]))
	  j = right;
      }

      if(j != i) {
	bia::SwapInt(&H->spel[j], &H->spel[i]);
	H->pos[H->spel[i]] = i;
	H->pos[H->spel[j]] = j;
	GoDown(H, j);
      }
    }


    bool IsFull(Heap *H) {
      if (H->last == (H->n - 1))
	return true;
      else
	return false;
    }

    bool IsEmpty(Heap *H) {
      if (H->last == -1)
	return true;
      else
	return false;
    }

    Heap *Create(int n, float *cost) {
      Heap *H = NULL;
      int i;
      
      if(cost == NULL) {
	bia::Warning((char *)"Cannot create heap without cost map",
		     (char *)"Heap::Create");
	return NULL;
      }
      
      H = (Heap *) malloc(sizeof(Heap));
      if(H != NULL){
	H->n       = n;
	H->cost    = cost;
	H->color   = (char *) malloc(sizeof(char) * n);
	H->spel   = (int *) malloc(sizeof(int) * n);
	H->pos     = (int *) malloc(sizeof(int) * n);
	H->last    = -1;
	H->removal_policy = MINVALUE;
	if (H->color == NULL || H->pos == NULL || H->spel == NULL)
	  bia::Error((char *)MSG1,
		     (char *)"Heap::Create");
	for(i=0; i<H->n; i++){
	  H->color[i] = WHITE;
	  H->pos[i]   = -1;
	  H->spel[i] = -1;
	}
      }
      else
	bia::Error((char *)MSG1,
		   (char *)"Heap::Create");
      
      return H;
    }


    void Destroy(Heap **H) {
      Heap *aux = *H;
      if (aux != NULL) {
	if (aux->spel != NULL)  free(aux->spel);
	if (aux->color != NULL) free(aux->color);
	if (aux->pos != NULL)   free(aux->pos);
	free(aux);
	*H = NULL;
      }
    }
    
    bool Insert(Heap *H, int spel) {
      if (!IsFull(H)) {
	H->last++;
	H->spel[H->last] = spel;
	H->color[spel]   = GRAY;
	H->pos[spel]     = H->last;
	GoUp(H, H->last); 
	return true;
      } else 
	return false;
    }

    bool Remove(Heap *H, int *spel) {
      if (!IsEmpty(H)) {
	*spel = H->spel[0];
	H->pos[*spel]   = -1;
	H->color[*spel] = BLACK;
	H->spel[0]      = H->spel[H->last];
	H->pos[H->spel[0]] = 0;
	H->spel[H->last] = -1;
	H->last--;
	GoDown(H, 0);
	return true;
      } else 
	return false;
    }


    void Update(Heap *H, int p, float value){
      H->cost[p] = value;
      
      if(H->color[p] == BLACK) printf("ferrou\n");
      
      if(H->color[p] == WHITE)
	Insert(H, p);
      else
	GoUp(H, H->pos[p]);
    }


    void Reset(Heap *H){
      int i;
      for (i=0; i < H->n; i++) {
	H->color[i] = WHITE;
	H->pos[i]   = -1;
	H->spel[i] = -1;
      }
      H->last = -1;
    }


  } //end Heap namespace
} //end bia namespace


