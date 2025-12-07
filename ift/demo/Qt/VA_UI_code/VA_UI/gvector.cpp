#define __STDC_WANT_LIB_EXT1__ 1


#include "gvector.h"


GVector* createVector(size_t capacity, size_t elementSize){
    GVector *vector = (GVector *)calloc(1,sizeof(GVector));
    vector->capacity = capacity;
    vector->size = 0;
    vector->elementSize = elementSize;
    vector->data = calloc(capacity,elementSize);
    vector->growthFactor = 2.0;
    vector->freeFunction = NULL;
    return vector;
}

GVector* createVector(size_t capacity, size_t elementSize,FreeFunction freeFunction){
    GVector *vector = (GVector *)calloc(1,sizeof(GVector));
    vector->capacity = capacity;
    vector->size = 0;
    vector->elementSize = elementSize;
    vector->data = calloc(capacity,elementSize);
    vector->growthFactor = 2.0;
    vector->freeFunction = freeFunction;
    return vector;
}

GVector* createVector(size_t capacity, size_t elementSize, void* initialValue){
    GVector* vector = createVector(capacity,elementSize);
    vector->size = vector->capacity;
    for (size_t i = 0; i < vector->size; ++i) {
        pushBackElementInVector(vector,initialValue);
    }
    return vector;
}

GVector* createNullVector(size_t capacity, size_t elementSize){
    GVector* vector = createVector(capacity,elementSize);
    vector->size = vector->capacity;
    return vector;
}

//assuming: vec->capacity > vec->size
//no cheking
//private function
/*TODO: try implement a optimized memmove*/
void shiftVectorToRightAt(GVector* vector,size_t index){
    unsigned char *source = (unsigned char *) vector->data + (vector->elementSize * index);
    //char *destination = source + vector->elementSize;
    size_t numberBytes = (vector->size - index) * vector->elementSize;
    memmove(source+vector->elementSize, source, numberBytes);
}


void shiftVectorToLeftAt(GVector* vector,size_t index){
    unsigned char *source = (unsigned char *) vector->data + (vector->elementSize * index);
    //char *destination = source + vector->elementSize;
    size_t rightElementsInBytes = (vector->size - index-1) * vector->elementSize;
    memmove(source, source + vector->elementSize, rightElementsInBytes);
}

void shiftVectorToLeft(GVector* vector,size_t indexBegin,size_t indexEnd){
    size_t stride = indexEnd-indexBegin+1;
    unsigned char *source = (unsigned char *) vector->data + (vector->elementSize * indexBegin);
    //char *destination = source + vector->elementSize*(stride);
    size_t rightElementsInBytes = (vector->size - indexBegin - stride) * vector->elementSize;
    memmove(source, source + vector->elementSize*(stride), rightElementsInBytes);
}

void insertElementInVectorAt(GVector* vector,void* element, size_t index){
    //printf("%lu %lu %lu\n",index,vector->size,vector->capacity);
    if(IS_INDEX_OUT_BOUNDS_INSERT(vector,index)){
        printf("[insertElementInVectorAt] Invalid position\n");
        return;
    }
    if( SHOULD_GROW(vector) ){
        setVectorCapacity(vector,vector->size*vector->growthFactor);
    }

    if(index == vector->size){
        unsigned char* offset = (unsigned char*)vector->data + (vector->elementSize*vector->size);//last position
        //char* data = (char*)element;
        TRANSFER_DATA_COPY(offset,((unsigned char*)element),vector->elementSize);
        //memcpy(offset, element, vector->elementSize);
        vector->size++;
        return;
    }

    unsigned char *source = (unsigned char *) vector->data + (vector->elementSize * index);
    size_t numberBytes = (vector->size - index) * vector->elementSize;
    memmove(source+vector->elementSize, source, numberBytes);
    unsigned char* offset = (unsigned char*)vector->data + (vector->elementSize*index);
    TRANSFER_DATA_COPY(offset,element,vector->elementSize);
    vector->size++;
}

void assignElementInVectorAt(GVector* vector,void* element, size_t index){

    if(IS_INDEX_OUT_BOUNDS(vector,index)){
        printf("[assignElementInVectorAt] Invalid position\n");
        return;
    }
    //TODO: decide how to deal with old pointer
    unsigned char* offset = (unsigned char*)vector->data + (vector->elementSize*index);
    //char* data = (char*)element;
    if(vector->freeFunction){
        vector->freeFunction((void*)offset);
    }
    //memcpy(offset, data, vector->elementSize);
    TRANSFER_DATA_COPY(offset,element,vector->elementSize);
}

void assignElementInVector(GVector* vector,void* element, size_t indexBegin,size_t indexEnd){

    if(IS_INDEX_OUT_BOUNDS(vector,indexBegin) || IS_INDEX_OUT_BOUNDS(vector,indexEnd)){
        printf("[assignElementInVector] Invalid position\n");
        return;
    }
    if(indexEnd < indexBegin){
        printf("[assignElementInVector] beging position is greater than end position\n");
        return;
    }
    //TODO: decide how to deal with old pointer
    size_t  chunk= (indexEnd-indexBegin)+1;
    for (size_t i = 0; i < chunk; ++i) {
        unsigned char* offset = (unsigned char*)vector->data + (vector->elementSize*i);
        //char* data = (char*)element;
        if(vector->freeFunction){
            vector->freeFunction((void*)offset);
        }
        //memcpy(offset, element, vector->elementSize);
        TRANSFER_DATA_COPY(offset,element,vector->elementSize);
    }
}


void setVectorCapacity(GVector* vector, size_t newCapacity){
     //solution 1
//    void *oldData = vector->data;
//    vector->data = calloc(newCapacity,vector->elementSize);
//    TRANSFER_DATA_COPY( vector->data,oldData,vector->size*vector->elementSize);
//    //memcpy(vector->data, oldData, vector->size*vector->elementSize);
//    vector->capacity = newCapacity;
//    free(oldData);
    //solution 2
    void *newData = realloc(vector->data,vector->elementSize*newCapacity);
    if(newData == NULL){
        printf("[setVectorCapacity] could not allocate memory");
        return;
    }
    vector->data = newData;
    vector->capacity = newCapacity;
}

void removeElementInVectorAt(GVector* vector, size_t index){
    if(IS_INDEX_OUT_BOUNDS(vector,index)){
        //printf("%lu %lu\n",vector->size,index);
        printf("[removeElementInVectorAt] Invalid position\n");
        return;
    }
    unsigned char *offset = (unsigned char *) vector->data + (vector->elementSize * index);
    if(vector->freeFunction){
        vector->freeFunction((void*)offset);
        size_t rightElementsInBytes = (vector->size - index-1) * vector->elementSize;
        memmove(offset, offset + vector->elementSize, rightElementsInBytes);
        vector->size--;
        return;
    }else{
        size_t rightElementsInBytes = (vector->size - index-1) * vector->elementSize;
        memmove(offset, offset + vector->elementSize, rightElementsInBytes);
        vector->size--;
        return;
    }
}

void removeElementsInVector(GVector* vector, size_t indexBegin,size_t indexEnd){
    if(IS_INDEX_OUT_BOUNDS(vector,indexBegin) || IS_INDEX_OUT_BOUNDS(vector,indexEnd)){
        printf("[removeElementsInVector] Invalid position\n");
        return;
    }
    if(indexEnd < indexBegin){
        printf("[removeElementsInVector] beging position is greater than end position\n");
        return;
    }
    if(vector->freeFunction){
        for (size_t i = indexBegin; i < indexEnd; ++i) {
            unsigned char *offset = (unsigned char *) vector->data + (vector->elementSize * i);
            vector->freeFunction((void*)offset);
        }
        //left shift
        size_t stride = indexEnd-indexBegin+1;
        unsigned char *source = (unsigned char *) vector->data + (vector->elementSize * indexBegin);
        size_t rightElementsInBytes = (vector->size - indexBegin - stride) * vector->elementSize;
        memmove(source, source + vector->elementSize*(stride), rightElementsInBytes);
        vector->size -= (indexEnd-indexBegin)+1;
        return;
    }else{
        size_t stride = indexEnd-indexBegin+1;
        unsigned char *source = (unsigned char *) vector->data + (vector->elementSize * indexBegin);
        size_t rightElementsInBytes = (vector->size - indexBegin - stride) * vector->elementSize;
        memmove(source, source + vector->elementSize*(stride), rightElementsInBytes);
        vector->size -= (indexEnd-indexBegin)+1;
        return;
    }
}

void popBackElementInVector(GVector* vector){
    if(vector->freeFunction){
        unsigned char* offset = (unsigned char*)vector->data + (vector->elementSize*vector->size);
        vector->freeFunction((void*)offset);
    }
    vector->size--;
}

void popFrontElementInVector(GVector* vector){

    removeElementInVectorAt(vector, 0);//already decreasing
}

void swapVectors(GVector *vector1,GVector *vector2){
    vector2->size = vector2->size + vector1->size;
    vector1->size = vector2->size - vector1->size;
    vector2->size = vector2->size - vector1->size;

    vector2->capacity = vector2->capacity + vector1->capacity;
    vector1->capacity = vector2->capacity - vector1->capacity;
    vector2->capacity = vector2->capacity - vector1->capacity;

    vector2->elementSize = vector2->elementSize + vector1->elementSize;
    vector1->elementSize = vector2->elementSize - vector1->elementSize;
    vector2->elementSize = vector2->elementSize - vector1->elementSize;

    FreeFunction temp_functionFree = vector2->freeFunction;
    vector2->freeFunction = vector1->freeFunction;
    vector1->freeFunction = temp_functionFree;

    void* temp_data = vector2->data;
    vector2->data = vector1->data;
    vector1->data = temp_data;
}

void clearVector(GVector *vector){

    if(vector->freeFunction){
        unsigned char* offset;
        for (size_t i = 0; i < vector->size; ++i) {
            offset = (unsigned char*)vector->data + (vector->elementSize*i);
            vector->freeFunction((void*)offset);
        }
    }

    free(vector->data);
    vector->data = NULL;
    vector->size = 0;

}

void destroyVector(GVector **vector){
    //GVector* aux = *vector;
    if(*vector == NULL || vector == NULL){
        return;
    }
    clearVector(*vector);
    free(*vector);
    *vector = NULL;
}

void resizeVector(GVector* vector, size_t newSize){
    if(newSize < vector->size){
        if(vector->freeFunction){
            unsigned char* offset;
            for (size_t i = 0; i < vector->size; ++i) {
                offset = (unsigned char*)vector->data + (vector->elementSize*i);
                vector->freeFunction((void*)offset);
            }
        }
        vector->size = newSize;
        return;
    }
    if(newSize > vector->capacity){
        setVectorCapacity(vector,newSize);
        vector->size = newSize;
        return;
    }

}

void shrinkToFit(GVector* vector){
    if(vector->size == vector->capacity){
        return;
    }
    setVectorCapacity(vector,vector->size);
}

VectorIterator* getVectorIteratorBegin(GVector* vector){
    VectorIterator* iterator = (VectorIterator*)calloc(1,sizeof(VectorIterator));
    iterator->elementSize = vector->elementSize;
    iterator->pointer = vector->data;
    return iterator;
}

VectorIterator* getVectorIteratorEnd(GVector* vector){
    VectorIterator* iterator = (VectorIterator*)calloc(1,sizeof(VectorIterator));
    iterator->elementSize = vector->elementSize;
    iterator->pointer = (unsigned char*)vector->data + (vector->elementSize*vector->size);
    return iterator;
}

size_t getIteratorIndexInVector(GVector* vector, VectorIterator* iterator){
    return ((unsigned char*)iterator->pointer - (unsigned char*)vector->data)/(vector->elementSize);
}

GVector* copyVector(GVector *vector){
    if(vector == NULL){
        return NULL;
    }

    GVector* output = createNullVector(vector->size,vector->elementSize);
    memcpy(output->data, vector->data, vector->elementSize*vector->size);
    return output;
}
