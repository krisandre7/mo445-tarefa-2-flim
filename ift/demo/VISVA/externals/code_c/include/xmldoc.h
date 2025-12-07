#ifndef _XMLDOC_H_
#define _XMLDOC_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include "common.h"
#include "hashtable.h"

#define XMLTOKEN_ERROR          -1
#define XMLTOKEN_SEARCH          0
#define XMLTOKEN_LEFTABRACKET    1
#define XMLTOKEN_RIGHTABRACKET   2
#define XMLTOKEN_EQUAL           3
#define XMLTOKEN_STRING          4
#define XMLTOKEN_QUOTED          5
#define XMLTOKEN_INT             6
#define XMLTOKEN_FLOAT           7
#define XMLTOKEN_BLANK           8
#define XMLTOKEN_HEADERBEGIN     9
#define XMLTOKEN_HEADEREND      10
#define XMLTOKEN_COMMENTBEGIN   11
#define XMLTOKEN_COMMENTEND     12
#define XMLTOKEN_LEFTCLOSETAG   13
#define XMLTOKEN_RIGHTCLOSETAG  14
#define XMLTOKEN_INTERROGATION  15
#define XMLTOKEN_MINUS          16
#define XMLTOKEN_DIVIDE         17


/* descrição de um atributo */
typedef struct _XMLTagAttrib {
  char *name;                 /*  nome do atributo      */
  char *value;                /*  valor do atributo     */
  struct _XMLTagAttrib *next; /*  apontator p/ próximo  */
} XMLTagAttrib;


/* descrição de uma tag */
typedef struct _XMLTag{ 
  char *name;         /*   nome da tag                   */
  //int  closed;        /*   0 = 'aberta',  1 = 'fechada'  */
  XMLTagAttrib *attrList; /*   lista de atributos          */
  struct _XMLNode *childList;  /*   lista de 'filhos'      */
} XMLTag;


/* descreve o 'conteúdo' de um elemento da página */               
typedef union { 
  char *text;         /*  texto        */
  char *comment;      /*  comentario   */
  XMLTag tag;         /*  tag          */
} XMLNodeContents;

/* tipos possíveis de elementos de um documento */
typedef enum {TEXT, COMMENT, TAG } XMLNodeType;


typedef struct _XMLNode {
  XMLNodeType type;
  XMLNodeContents contents;
  struct _XMLNode *next;
  struct _XMLNode *parent;
} XMLNode;


typedef struct _XMLDocument {
  HashTable *htab;     /*  tabela de hashing     */
  XMLNode   *root;    /*  tag principal */
  char *version;
  char *encoding;
} XMLDocument;


typedef struct _XMLToken {
  char val[1024];
  int type;
} XMLToken;


void Indentation(int space, FILE *file);

/* cria uma nova página */
XMLDocument *CreateXMLDocument(XMLNode *root, char *version, char *encoding);

void DestroyXMLDocument(XMLDocument **doc);

/* insere um elemento a uma página como filho da tag 'parent' */
void AppendXMLTagChild(XMLDocument *doc, XMLNode *parent, XMLNode *son);

XMLNode *GetXMLTag(XMLDocument *doc, char *id);

/* agrega um atributo a uma tag */
void AddXMLTagAttrib(XMLNode *p, char *name, char *value);

/* cria um 'elemento' do tipo 'tag XML' */
XMLNode *NewXMLTag(char *name);

/* cria um 'elemento'  do tipo comentário */
XMLNode *NewXMLComment(char *comment);

/* cria um 'elemento' do tipo texto */                          
XMLNode *NewXMLText(char *text);

/* escreve num arquivo o texto XML correspondente a uma página */
void WriteXMLDocument(XMLDocument *doc, char *filename);

/* le um arquivo XML */
XMLDocument *ReadXMLDocument(char *filename);

#endif


