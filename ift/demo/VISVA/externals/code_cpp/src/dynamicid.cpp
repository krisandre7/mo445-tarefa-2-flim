
#include "dynamicid.h"

DynamicID :: DynamicID(int base_id){
  this->base_id = base_id;
  this->id = base_id;
  R = NULL;
}


DynamicID :: ~DynamicID(){
  DestroySet(&R);
}


int DynamicID :: AllocID(){
  if(R==NULL){
    id++;
    return (id-1);
  }
  else{
    return RemoveSet(&R);
  }
}


void DynamicID :: FreeID(int id){
  if(id<base_id || id>=this->id)
    return;
  if(IsInSet(R, id))
    return;
  InsertSet(&R, id);
}


