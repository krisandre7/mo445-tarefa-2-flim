
#include "segmobject.h"


SegmObject *CreateSegmObject(char *name,
			     int color){
  SegmObject *obj;

  obj = (SegmObject *) calloc(1,sizeof(SegmObject));
  if(obj == NULL)
    Error(MSG1,"CreateSegmObject");
  
  strcpy(obj->name, name);
  obj->color = color;
  obj->alpha = 255;
  obj->visibility = false;
  obj->mask = NULL;
  obj->seed = NULL;

  return obj;
}


void       DestroySegmObject(SegmObject **obj){
  SegmObject *aux;

  aux = *obj;
  if(aux != NULL){
    if(aux->mask != NULL)
      BMapDestroy(aux->mask);
    if(aux->seed != NULL)
      DestroyMarkerList(&aux->seed);
    free(aux);
    *obj = NULL;
  }
}


void        CopySegmObjectMask2Scene(SegmObject *obj, 
				     Scene *dest){
  int n,p;
  
  n = dest->n;
  if(n != (obj->mask)->N)
    Error("Incompatible sizes",
	  "CopySegmObjectMask2Scene");
  for(p=0; p<n; p++)
    dest->data[p] = _fast_BMapGet(obj->mask,p);
}



void        freeSegmObject(void **obj){
  DestroySegmObject((SegmObject **)obj);
}

