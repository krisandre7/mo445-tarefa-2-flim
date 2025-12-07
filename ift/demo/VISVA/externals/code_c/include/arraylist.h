// version 00.00.02

#ifndef _ARRAYLIST_H_
#define _ARRAYLIST_H_

#include "common.h"

typedef struct _ArrayList {
  void **array;
  int cap; //Capacity.
  int n;   //Number of objects added.
  void (*clean)(void**); //Clean function
} ArrayList;


ArrayList *CreateArrayList(int cap);
void       DestroyArrayList(ArrayList **A);

void       SetArrayListCleanFunc(ArrayList *A,
				 void (*clean)(void**));

void       AddArrayListElement(ArrayList *A, 
			       void *elem);
void      *GetArrayListElement(ArrayList *A, 
			       int index);
void       DelArrayListElement(ArrayList *A, 
			       int index);
void       DelArrayListElement_2(ArrayList *A,
				 void **elem);

void       ResizeArrayList(ArrayList *A, int n);

//Trims the capacity of this ArrayList instance 
//to be the list's current size.
void       Trim2SizeArrayList(ArrayList *A);


#endif

