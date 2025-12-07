#include "border.h"
#include "common.h"

Border *CreateBorder(int nmax)
{
  Border *border=NULL;
  int i;
  
  border = (Border *) calloc(1,sizeof(Border));
  if(border==NULL) Error(MSG1,"CreateBorder");
  border->voxels = (BorderNode *) calloc(nmax,sizeof(BorderNode));
  if(border->voxels==NULL) Error(MSG1,"CreateBorder");
  for(i=0;i<nmax;i++) { // initialization of all nodes with NIL
    border->voxels[i].next = NIL;
    border->voxels[i].prev = NIL;
  }
  border->nmax  = nmax;
  border->ncur  = 0;
  border->first = NIL;
  border->last  = NIL;
  return border;
}

void DestroyBorder(Border **border)
{
  Border *aux=NULL;
  
  aux = *border;
  if(aux != NULL){
    if(aux->voxels != NULL) free(aux->voxels);
    free(aux);
    *border = NULL;
  }
}

void InsertBorder(Border *border, int index)
{
  if(border->first == NIL) { // list is empty
    if(border->last != NIL) Error(" Construction error","InsertBorder"); // optional
    border->first = index;
    border->last  = index;
    border->voxels[index].prev = NIL;
    border->voxels[index].next = NIL;
	}
  else {
    border->voxels[index].prev = border->last;
    border->voxels[index].next = NIL;
    if(border->voxels[border->last].next != NIL) Error(" Construction error 2", "InsertBorder"); // optional
    border->voxels[border->last].next = index;
    border->last = index;
  }
  (border->ncur)++; // optional
}

void RemoveBorder(Border *border, int index)
{	
  if(border->ncur == 0) Error("Cannot remove element from an empty list","RemoveBorder"); // optional
  
  if(index == border->last) border->last = border->voxels[index].prev; // remove the last voxel
  else border->voxels[border->voxels[index].next].prev = border->voxels[index].prev; // previous of the next is previous of the current
  
  if(index == border->first) border->first = border->voxels[index].next; // remove the first voxel
  else border->voxels[border->voxels[index].prev].next = border->voxels[index].next; // next of the previous is next of the current
  
  border->voxels[index].next = border->voxels[index].prev = NIL; // next and previous of the current are NIL
  (border->ncur)--; // optional
}

int IsBorder(Border *border, int index)
{
  if((border->voxels[index].next == NIL)&&(border->voxels[index].prev == NIL)&&(border->first != index))
    return(0); // do not belong to the border
  else
    return(1); // belong to the border
}
