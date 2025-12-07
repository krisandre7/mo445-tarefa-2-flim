
#include "hashtable.h"

/* inicializando a Tabela de Hash */
HashTable *CreateHashTable(int size){
  HashTable *ht;
  int i;

  if(size<=0)
    Error("Invalid size value","CreateHashTable");
  ht = (HashTable *)malloc(sizeof(HashTable));
  if(ht==NULL)
    Error(MSG1,"CreateHashTable");
  ht->size = size;
  ht->data = (HashNode **)malloc(size*sizeof(HashNode *));
  if(ht->data==NULL)
    Error(MSG1,"CreateHashTable");

  for(i=0; i<size; i++)
    ht->data[i]=NULL;

  return ht;
}

/* Funcao para calcular posicao na tabela de hash */
int HashPosition(HashTable *ht, char *key){
  int h = 0, a = 127;

  for (; *key != '\0'; key++)
    h = (a * h + *key) % ht->size;

  return h;
}

/* insere (chave,valor) numa tabela de hash */
void InsertHashNode(HashTable *ht, char *key, void *value){
  HashNode *node;
  int i;

  i = HashPosition(ht, key);
  node = (HashNode *)malloc(sizeof(HashNode));
  node->key = key;
  node->value = value;
  node->next = ht->data[i];
  ht->data[i] = node;
}

/* busca pelo valor associado a uma chave */
void *SearchHashNode(HashTable *ht, char *key){
  HashNode *node;
  int i;

  i = HashPosition(ht, key);
  for(node=ht->data[i]; node!=NULL; node=node->next)
    if(strcmp(key, node->key)==0)
      return node->value;

  return NULL;
}


void DestroyHashTable(HashTable **ht){
  HashNode *node,*tmp;
  int i;

  if(*ht!=NULL){
    for(i=0; i<(*ht)->size; i++){
      node = (*ht)->data[i];
      while(node!=NULL){
	tmp = node;
	node = node->next;
	free(tmp);
      }
    }
    free((*ht)->data);
    free(*ht);
    *ht = NULL;
  }
}

