#include "argumentList.h"

ArgumentList* createArgumentList(){
    ArgumentList* list = (ArgumentList*)calloc(1,sizeof(ArgumentList));
    list->head = list->tail = NULL;
    list->length = 0;
    list->currentNode = NULL;
    list->currentNodeIndex = -1;
    return list;
}

ArgumentListNode* createArgumentNode(void* element, size_t dataSize, char* typeName,FreeFunction freeFunction){
    ArgumentListNode* node = (ArgumentListNode*)calloc(1,sizeof(ArgumentListNode));
    node->data = calloc(1,dataSize);
    node->dataSize = dataSize;
    if(typeName){
        size_t stringSize = 0;
        while(typeName[stringSize] != '\0'){
            stringSize++;
        }
        node->typeName = (char*)calloc(stringSize+1,sizeof(char));
        for (size_t i = 0; i < stringSize; ++i) {
            node->typeName[i] = typeName[i];
        }
        node->typeName[stringSize] = '\0';
    }
    node->freeFunction = freeFunction;
    if(dataSize < 100){
        TRANSFER_DATA_COPY(node->data, element, dataSize);
    }else{
        memcpy(node->data, element, dataSize);
    }
    return node;
}

ArgumentListNode* createRawArgumentNode(size_t dataSize){
    ArgumentListNode* node = (ArgumentListNode*)calloc(1,sizeof(ArgumentListNode));
    node->data = calloc(1,dataSize);
    node->dataSize = dataSize;
    node->freeFunction = NULL;
    node->typeName = NULL;
    node->next = NULL;
    node->previous = NULL;
    return node;
}

void destroyArgumentList(ArgumentList **list){
    ArgumentList* aux = *list;
    ArgumentListNode *currentNode = NULL;
    if(aux == NULL){
        return;
    }

    aux->tail->next = NULL;
    aux->head->previous = NULL;

    while(aux->head != NULL) {
        currentNode = aux->head;
        aux->head = currentNode->next;
        destroyNodeArgumentList(&currentNode);
        aux->length--;
    }
    free(aux);
    aux = NULL;
}

void destroyNodeArgumentList(ArgumentListNode **node){
    ArgumentListNode *aux = *node;
    if(aux == NULL){
        return;
    }
    if(aux->freeFunction) {
        aux->freeFunction(aux->data);
    }else{
        free(aux->data);
    }
    aux->data = NULL;
    if(aux->typeName){
        free(aux->typeName);
    }
    free(aux);
    aux = NULL;
}

void clearArgumentList(ArgumentList* list){
    if(list == NULL){
        return;
    }
    ArgumentListNode *currentNode = NULL;
    list->tail->next = NULL;
    list->head->previous = NULL;
    while(list->head != NULL) {
        currentNode = list->head;
        list->head = currentNode->next;
        destroyNodeArgumentList(&currentNode);
        list->length--;
    }
}

