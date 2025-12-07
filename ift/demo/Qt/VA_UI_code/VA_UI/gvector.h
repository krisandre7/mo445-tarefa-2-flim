//
// Created by deangeli on 5/1/17.
//

#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataTransference.h"

/* Structures */
typedef void (*FreeFunction)(void *);

typedef struct _vector {
    size_t size;
    size_t capacity;
    size_t elementSize;
    float growthFactor;
    FreeFunction freeFunction;

    void* data;
} GVector;



typedef struct _vectorIterator {
    void* pointer;
    size_t elementSize;
} VectorIterator;


/*Creation and Destruction*/
GVector* createVector(size_t capacity, size_t elementSize);
GVector* createVector(size_t capacity, size_t elementSize, void* initialVal);
GVector* createVector(size_t capacity, size_t elementSize,FreeFunction freeFunction);
GVector* createNullVector(size_t capacity, size_t elementSize);
void destroyVector(GVector **vector);
GVector* copyVector(GVector *vector);
/*******************************************/


/*Capacity and bounds*/
#define IS_INDEX_OUT_BOUNDS(vector,i) (i >=vector->size)
#define IS_INDEX_OUT_BOUNDS_INSERT(vector,i) (i > vector->size)
#define SHOULD_GROW(vector) (vector->size >= vector->capacity)

#define VECTOR_SET_CAPACITY(vector, newCapacity) \


//private
//used to grow or Shrink the vector
void setVectorCapacity(GVector* vector, size_t newCapacity);
void resizeVector(GVector* vector, size_t newSize);
void shrinkToFit(GVector* vector);
/*******************************************/

/*Element access*/
#define VECTOR_GET_ELEMENT_AS(type, vectorPointer, index) ((type*)vectorPointer->data)[index]
#define VECTOR_GET_FRONT_ELEMENT_AS(type, vectorPointer, index) ((type*)vectorPointer->data)[vectorPointer->size]

inline void*  getElementInVectorAt(const GVector* vector, size_t index){
    if(IS_INDEX_OUT_BOUNDS(vector,index)){
        printf("[getElementInVector] Invalid position\n");
        return NULL;
    }
    void* offset = (char*)vector->data + (vector->elementSize*index);
    return offset;
}

inline void* getFrontElementInVector(const GVector* vector){
    void* offset = (char*)vector->data ;
    return offset;
}

inline void* getBackElementInVector(const GVector* vector){
    void* offset = (char*)vector->data + (vector->elementSize*vector->size); ;
    return offset;
}

inline void* getDataInVector(const GVector* vector){
    return vector->data;
}


/*******************************************/

/*Modifiers*/

#define VECTOR_PUSH_BACK(type, vector, element) \
        if (SHOULD_GROW(vector)){setVectorCapacity(vector,vector->size*vector->growthFactor);} \
        ((type*)vector->data)[vector->size] = element; \
        vector->size++;

#define VECTOR_INSERT_AT(type, vector, element, index) \
        shiftVectorToRightAt(vector,index); \
        ((type*)vector->data)[vector->size] = element; \
        vector->size++;


void assignElementInVectorAt(GVector* vector,void* element, size_t index);
void assignElementInVector(GVector* vector,void* element, size_t indexBegin,size_t indexEnd);
inline void pushBackElementInVector(GVector* vector,void* element){
    if( SHOULD_GROW(vector) ){
        setVectorCapacity(vector,vector->size*vector->growthFactor);
    }
    unsigned char* offset = (unsigned char*)vector->data + (vector->elementSize*vector->size);//last position
    TRANSFER_DATA_COPY(offset,element,vector->elementSize);
    //transferData(offset,data,vector->elementSize);
    //memcpy(offset, element, vector->elementSize);
    vector->size++;
}



void popBackElementInVector(GVector* vector);
void insertElementInVectorAt(GVector* vector,void* element, size_t index);
void removeElementInVectorAt(GVector* vector, size_t index);
void removeElementsInVector(GVector* vector, size_t indexBegin,size_t indexEnd);
void swapVectors(GVector *vector1,GVector *vector2);
void clearVector(GVector *vector);

//private
void shiftVectorToRightAt(GVector* vector,size_t index);
void shiftVectorToRightAt2(GVector* vector,size_t index);
void shiftVectorToLeftAt(GVector* vector,size_t index);
void shiftVectorToLeft(GVector* vector,size_t indexBegin,size_t indexEnd);

/*******************************************/

/*Iterator*/
VectorIterator* getVectorIteratorBegin(GVector* vector);
VectorIterator* getVectorIteratorEnd(GVector* vector);
inline void* getValueInVectorIterator(VectorIterator* iterator){
    return iterator->pointer;
}
inline void incrementVectorIterator(VectorIterator* iterator){
    iterator->pointer = (char*)iterator->pointer + iterator->elementSize;
}
inline void decrementVectorIterator(VectorIterator* iterator){
    iterator->pointer = (char*)iterator->pointer - iterator->elementSize;
}

inline void* getNextValueInVectorIterator(VectorIterator* iterator){
    void* data = iterator->pointer;
    incrementVectorIterator(iterator);
    return data;
}

inline void* getPreviousValueInVectorIterator(VectorIterator* iterator){
    void* data = iterator->pointer;
    decrementVectorIterator(iterator);
    return data;
}

size_t getIteratorIndexInVector(GVector* vector, VectorIterator* iterator);

#define VECTOR_GET_ITERATOR_NEXT_AS(type, iterator) *((type*)getNextValueInVectorIterator((iterator))
#define VECTOR_GET_ITERATOR_PREVIOUS_AS(type, iterator) *((type*)getPreviousValueInVectorIterator((iterator))
#define VECTOR_GET_ITERATOR_VALUE_AS(type, iterator) *((type*)getValueInVectorIterator((iterator))

#define VECTOR_PRINT_AS(type, symbol, vector) \
    for(size_t currentIndexVector = 0; currentIndexVector < vector->size; currentIndexVector++){\
        printf(symbol,  VECTOR_GET_ELEMENT_AS(type,vector,currentIndexVector) ); \
    }\
    printf("\n");

/*******************************************/

#endif //BITBUCKET_VECTOR_H
