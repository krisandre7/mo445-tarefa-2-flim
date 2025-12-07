//
// Created by deangeli on 5/1/17.
//

#ifndef _ARGUMENTLIST_H_
#define _ARGUMENTLIST_H_

#include "iftCommon.h"
#include "dataTransference.h"

typedef void (*FreeFunction)(void *);

typedef struct _ArgumentListNode {
    void *data;
    size_t dataSize;
    struct _ArgumentListNode *next;
    struct _ArgumentListNode *previous;
    char* typeName;
    FreeFunction freeFunction;
} ArgumentListNode;

typedef struct _argumentList{
    size_t length;
    ArgumentListNode *head;
    ArgumentListNode *tail;
    ArgumentListNode *currentNode;
    int currentNodeIndex;
} ArgumentList;


//LinkedList* createLinkedList(size_t dataSize);
ArgumentList* createArgumentList();
ArgumentListNode* createArgumentNode(void* element, size_t dataSize, char* typeName = NULL,FreeFunction freeFunction = NULL);
ArgumentListNode* createRawArgumentNode(size_t dataSize);


inline void  setCursorPositionInArgumentList(ArgumentList* list, int index,ArgumentListNode* node){
    list->currentNodeIndex = index;
    list->currentNode = node;
}

inline void appendElementInArgumentList(ArgumentList *argumentList, void *element, size_t dataSize){
    ArgumentListNode* node = createArgumentNode(element,dataSize);

    if(argumentList->length == 0) {
        argumentList->head = argumentList->tail = node;
    } else {
        argumentList->tail->next = node;
        node->previous = argumentList->tail;
        argumentList->tail = node;
    }

    argumentList->length++;
}
#define ARGLIST_GET_ELEMENT_AS(type,list,index) (*((type*)getElementInArgumentList(list,index)))


inline void pushBackElementInArgumentList(ArgumentList *argumentList, void *element, size_t dataSize){
    appendElementInArgumentList(argumentList,element,dataSize);
}

inline void pushBackElementInArgumentList(ArgumentList *argumentList, ArgumentListNode* node){
    if(argumentList->length == 0) {
        argumentList->head = argumentList->tail = node;
    } else {
        argumentList->tail->next = node;
        node->previous = argumentList->tail;
        argumentList->tail = node;
    }

    argumentList->length++;
}
inline void prependElementInArgumentList(ArgumentList *argumentList, void *element, size_t dataSize){
    ArgumentListNode* node = createArgumentNode(element,dataSize);

    if(argumentList->length == 0) {
        argumentList->head = node;
        argumentList->tail = argumentList->head;
    }else{
        node->next = argumentList->head;
        argumentList->head->previous = node;
        argumentList->head = node;
    }

    argumentList->length++;
}

inline void pushFrontElementInArgumentList(ArgumentList *argumentList, void *element, size_t dataSize){
    prependElementInArgumentList(argumentList,element,dataSize);
}

#define ARGLIST_PUSH_BACK_AS(type,list,element) {\
    ArgumentListNode* node = createRawArgumentNode(sizeof(type)); \
    *( (type*)node->data) = element; \
    pushBackElementInArgumentList(list,node); \
    }
//void insertElementInListAt(LinkedList *list, void *element, size_t index);
//void removeListHead(LinkedList *list);
//void removeListTail(LinkedList *list);
//void removeElementInListAt(LinkedList *list, size_t index);
//void removeElementInListByReference(LinkedList *list, void *element);
//void removeElementInListGivenValue(LinkedList *list, void *element);
//void removeElementsInListGivenValue(LinkedList *list, void *element);
//void resetIterator(LinkedList *list);
//void* getNextElement(LinkedList *list);
//void* getPreviousElement(LinkedList *list);
inline void* getElementInArgumentList(ArgumentList *list, size_t index){
    if(index >= list->length){
        printf("[getElementInArgumentList] invalid position %lu. The list length is %lu (indexing start from 0)\n", index,list->length);
        return NULL;
    }

    if (index == 0){
        return list->head->data;
    }
    if(index == list->length-1){
        return list->tail->data;
    }

    int distance2Head = index;
    int distance2Tail = list->length -index;
    int distance2Current = index - list->currentNodeIndex;
    int currentDirection = 0; //foward
    if(distance2Current <= 0){
        currentDirection = 1;//backward
        distance2Current = -distance2Current;
    }

    if(distance2Head <= distance2Tail && distance2Head <= distance2Current){//head 2 element
        ArgumentListNode *currentNode = list->head;
        for (size_t i = 0; i < list->length; ++i) {
            if(i == index){
                setCursorPositionInArgumentList(list,i,currentNode);
                return currentNode->data;
            }else{
                currentNode = currentNode->next;
            }

        }
    }else if(distance2Tail <= distance2Current) {//tail 2 element
        ArgumentListNode *currentNode = list->tail;
        for (int i = list->length-1; i >= 0; --i) {
            if(i == (int)index){
                setCursorPositionInArgumentList(list,i,currentNode);
                return currentNode->data;
            }else{
                currentNode = currentNode->previous;
            }
        }
    }else{//current 2 element
        if(currentDirection){//element is back
            ArgumentListNode *currentNode = list->currentNode;
            for (int i = list->currentNodeIndex; i >= 0; --i) {
                if(i == (int)index){
                    setCursorPositionInArgumentList(list,i,currentNode);
                    return currentNode->data;
                }else{
                    currentNode = currentNode->previous;
                }
            }
        }else{//element is front
            ArgumentListNode *currentNode = list->currentNode;
            for (size_t i = list->currentNodeIndex; i < list->length; ++i) {
                if(i == index){
                    setCursorPositionInArgumentList(list,i,currentNode);
                    return currentNode->data;
                }else{
                    currentNode = currentNode->next;
                }
            }
        }
    }

    //unlikely hit this bit
    return NULL;
}



//LinkedListNode* getLinkedListNode(LinkedList *list, size_t index);
void destroyArgumentList(ArgumentList **list);
void destroyNodeArgumentList(ArgumentListNode **node);
void clearArgumentList(ArgumentList* list);


#endif //_ARGUMENTLIST_H_
