#ifndef _HASHTABLE_H_
#define _HASHTABLE_H_

#include <stdlib.h>
#include <string.h>
#include "common.h"

/* nó da tabela de hashing */
typedef struct _HashNode {
  char *key;               /* chave de busca      */
  void *value;          /* valor associado     */
  struct _HashNode *next;  /* apontado p/ próximo */
} HashNode;


typedef struct _HashTable {
  HashNode **data;
  int size;
} HashTable;
 
/* inicia uma tabela de hashing */
HashTable *CreateHashTable(int size);

/* insere (chave,valor) numa tabela de hashing */
void InsertHashNode(HashTable *ht, char *key, void *value);

/* busca pelo valor associado a uma chave */
void *SearchHashNode(HashTable *ht, char *key);

void DestroyHashTable(HashTable **ht);

#endif
