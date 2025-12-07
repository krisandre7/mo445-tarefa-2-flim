
#include "xmldoc.h"

XMLNode *ReadXMLTag(FILE *file, int *c, int *closed);
int      NextXMLToken(FILE *file, XMLToken *token, int last_c);
void     WriteXMLNode(XMLNode *p, FILE *f, int space);


void Indentation(int space, FILE *file){
  int i; 

  for(i=0;i<space;i++) 
    fprintf(file," ");
}


/* cria uma nova página (vazia) */
XMLDocument *CreateXMLDocument(XMLNode *root, char *version, char *encoding) {
  XMLDocument *doc; 
  XMLTagAttrib *atr;
  
  if(root->type!=TAG)
    return NULL;
  
  doc = (XMLDocument *)malloc(sizeof(XMLDocument));;
  if(doc==NULL)
    Error(MSG1,"CreateXMLDocument");
  doc->htab = CreateHashTable(512);
  doc->version  = version;
  doc->encoding = encoding;
  doc->root = root;
  root->parent = NULL;
  root->next = root;

  /* Percorrendo os atributos para ver se ele nao tem um id. */
  for(atr = root->contents.tag.attrList; atr!=NULL; atr=atr->next) {
    if(strcmp(atr->name,"id")==0) {
      InsertHashNode(doc->htab, atr->value, root);
      break;
    }
  }
  
  return doc;
}


void DestroyXMLDocument(XMLDocument **doc){
  XMLNode *node,*tmp,*parent;
  XMLTagAttrib *attrList,*aux;

  if(doc==NULL)  return;
  if(*doc==NULL) return;

  free((*doc)->version);
  free((*doc)->encoding);
  DestroyHashTable(&((*doc)->htab));

  parent = (*doc)->root->next;
  (*doc)->root->next = NULL;
  do{
    node = parent;
    while(node!=NULL){
      switch(node->type){
      case TAG:
	if(node->contents.tag.childList!=NULL){
	  tmp = node;
	  node = tmp->contents.tag.childList->next;
	  tmp->contents.tag.childList->next = NULL;
	  tmp->contents.tag.childList = NULL;
	}
	else{
	  free(node->contents.tag.name);
	  attrList = node->contents.tag.attrList;
	  while(attrList!=NULL){
	    free(attrList->name);
	    free(attrList->value);
	    aux = attrList;
	    attrList = attrList->next;
	    free(aux);
	  }
	  parent = node->parent;
	  tmp = node;
	  node = node->next;
	  free(tmp);
	}
	break;
      case COMMENT:
	free(node->contents.comment);
	parent = node->parent;
	tmp = node;
	node = node->next;
	free(tmp);
	break;
      case TEXT:
	free(node->contents.text);
	parent = node->parent;
	tmp = node;
	node = node->next;
	free(tmp);
	break;
      }
    }
    
  }while(parent!=NULL);

  free(*doc);
}


XMLNode *GetXMLTag(XMLDocument *doc, char *id) {
  XMLNode *node;

  node = SearchHashNode(doc->htab, id);
  return node;
}

/* insere um elemento a uma página como filho da tag 'parent' */
void AppendXMLTagChild(XMLDocument *doc, XMLNode *parent, XMLNode *son) {
  XMLTagAttrib *atr;
  XMLNode *aux;

  if(doc==NULL || parent==NULL || son==NULL) return;

  /* Verifica se parent pertence ao documento doc */
  aux = parent;
  while(aux->parent!=NULL)
    aux = aux->parent;

  if(aux != doc->root){
    Warning("XMLNode must be a member of the XMLDocument.","AppendXMLTagChild");
    return;
  }

  /* agrega um 'filho' a uma 'tag' */
  /* Observacao, lista circular, pois a impressao deve ser em ordem */
  if(parent->type!=TAG) {
    Error("Caution! You should append a child on a tag element!","AppendXMLTagChild");
  }
  else if(parent->contents.tag.childList==NULL) {
    parent->contents.tag.childList = son;
    son->next = son;
    son->parent = parent;
  }
  else {
    son->next = parent->contents.tag.childList->next;
    parent->contents.tag.childList->next = son;
    parent->contents.tag.childList = son;
    son->parent = parent;
  }
  
  /* Percorrendo os atributos de p para ver se ele nao tem um id. */
  if(son->type==TAG) {
    for(atr = son->contents.tag.attrList; atr!=NULL; atr=atr->next) {
      if(strcmp(atr->name,"id")==0) {
	InsertHashNode(doc->htab, atr->value, son);
	break;
      }
    }
  }
}

/* agrega um atributo a uma tag */
void AddXMLTagAttrib(XMLNode *p, char *name, char *value) {
  XMLTagAttrib *at;

  if(p->type!=TAG) {
    Error("Caution! You should add an attribute to a tag element!","AddXMLTagAttrib");
  }
  at = (XMLTagAttrib *)malloc(sizeof(XMLTagAttrib));
  at->name = name;
  at->value = value;
  at->next = p->contents.tag.attrList;
  p->contents.tag.attrList = at;
}


/* cria um 'elemento' do tipo 'tag XML' */
XMLNode *NewXMLTag(char *name) {
  XMLNode *node;
  
  node = (XMLNode *)malloc(sizeof(XMLNode));
  node->type = TAG;
  node->contents.tag.name = name;
  node->contents.tag.attrList = NULL;
  node->contents.tag.childList = NULL;
  node->next = node;
  node->parent = NULL;

  return node;
}

/* cria um 'elemento'  do tipo comentário */
XMLNode *NewXMLComment(char *comment) {
  XMLNode *node;

  node = (XMLNode *)malloc(sizeof(XMLNode));
  node->type = COMMENT;
  node->contents.comment = comment;
  node->next = node;
  node->parent = NULL;

  return node;
}

/* cria um 'elemento' do tipo texto */                          
XMLNode *NewXMLText(char *text) {
  XMLNode *node;
	
  node = (XMLNode *)malloc(sizeof(XMLNode));
  node->type = TEXT;
  node->contents.text = text;
  node->next = node;
  node->parent = NULL;
  
  return node;
}

/* Essa funcao deve ler um elemento:
	caso esse elemento seja uma TAG: 
		- Imprime o cabeçalho da tag
		- Chamada recursiva para seus filhos
		- Fechar a tag caso ela seja fechada
	caso esse elemento seja um TEXTO: imprimir o texto
	caso esse elemento seja um COMENTARIO: imprimir o comentario
*/
void WriteXMLNode(XMLNode *p, FILE *f, int space) {
  XMLTagAttrib *atr;
  XMLNode *child;

  if(p->type==TAG) {
    /* Inicializando TAG */
    Indentation(space, f);
    fprintf(f,"<%s",p->contents.tag.name);
		
    /* imprimindo atributos */
    for(atr=p->contents.tag.attrList; atr!=NULL; atr=atr->next)
      fprintf(f," %s=\"%s\" ",atr->name,atr->value);
		
    /* imprimindo todos os filhos em ordem */
    child = p->contents.tag.childList;
    if(child==NULL)
      fprintf(f,"/>\n");
    else {
      fprintf(f,">\n");
      child = child->next;
      do {
	WriteXMLNode(child, f, space+5);
	child = child->next;
      }while(child!=p->contents.tag.childList->next);
    }
    /* finalizando a tag se for fechada */
    if(p->contents.tag.childList!=NULL) {
      Indentation(space, f);
      fprintf(f,"</%s>\n",p->contents.tag.name);
    }
  }
  else if(p->type==TEXT) {
    Indentation(space, f);
    fprintf(f,"%s\n",p->contents.text);
  }
  else {
    Indentation(space, f);
    fprintf(f,"<!--%s-->\n",p->contents.comment);		
  }
}

/* escreve num arquivo o texto XML correspondente a um documento */
void WriteXMLDocument(XMLDocument *doc, char *filename) {
  FILE *f;
  
  f = fopen(filename,"w");
  fprintf(f,"<?xml version=\"%s\" encoding=\"%s\" ?>\n",doc->version,doc->encoding);
  WriteXMLNode(doc->root, f, 0);
}

int NextXMLToken(FILE *file, XMLToken *token, int last_c){
  int c, i, flag;
  int state;
  
  i = flag = 0;
  state = XMLTOKEN_SEARCH;

  c = last_c;
  do{
    
    switch(state){
    case XMLTOKEN_SEARCH:
      if(c=='<'){
	state = token->type = XMLTOKEN_LEFTABRACKET;
	token->val[i] = c;
	i++;
      }
      else if(c=='>'){
	token->type = XMLTOKEN_RIGHTABRACKET;
	token->val[0] = c;
	token->val[1] = '\0';
	flag = 1;
      }
      else if(c=='='){
	token->type = XMLTOKEN_EQUAL;
	token->val[0] = c;
	token->val[1] = '\0';
	flag = 1;
      }
      else if((c>='A' && c<='Z')||(c>='a' && c<='z')||(c=='_')){
	state = token->type = XMLTOKEN_STRING;
	token->val[i] = c;
	i++;
      }
      else if((c>='0' && c<='9')){
	state = token->type = XMLTOKEN_INT;
	token->val[i] = c;
	i++;
      }
      else if(c=='-'){
	state = token->type = XMLTOKEN_MINUS;
	token->val[i] = c;
	i++;
      }
      else if(c=='/'){
	state = token->type = XMLTOKEN_DIVIDE;
	token->val[i] = c;
	i++;
      }
      else if(c=='.'){
	state = token->type = XMLTOKEN_FLOAT;
	token->val[i] = c;
	i++;
      }
      else if(c=='\"'){
	state = token->type = XMLTOKEN_QUOTED;
      }
      else if((c==' ')||(c=='\n')||(c=='\t')){
	state = token->type = XMLTOKEN_BLANK;
	token->val[i] = c;
	i++;
      }
      else if(c=='?'){
	state = token->type = XMLTOKEN_INTERROGATION;
	token->val[i] = c;
	i++;
      }
      break;
    case XMLTOKEN_BLANK:
      if((c==' ')||(c=='\n')||(c=='\t')){
	token->val[i] = c;
	i++;
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    case XMLTOKEN_STRING:
      if((c>='A' && c<='Z')||(c>='a' && c<='z')||(c=='_')||(c>='0' && c<='9')){
	token->val[i] = c;
	i++;
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    case XMLTOKEN_QUOTED:
      if(c=='\"'){
	token->val[i] = '\0';
	flag=1;
      }
      else{
	token->val[i] = c;
	i++;
      }
      break;
    case XMLTOKEN_INT:
      if((c>='0' && c<='9')){
	token->val[i] = c;
	i++;
      }
      else if((c=='.')){
	state = token->type = XMLTOKEN_FLOAT;
	token->val[i] = c;
	i++;
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    case XMLTOKEN_FLOAT:
      if((c>='0' && c<='9')){
	token->val[i] = c;
	i++;
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    case XMLTOKEN_LEFTABRACKET:
      if(c=='?'){
	token->type = XMLTOKEN_HEADERBEGIN;
	token->val[i] = c; 
	i++;
	token->val[i] = '\0';
	flag = 1;
      }
      else if(c=='/'){
	token->type = XMLTOKEN_LEFTCLOSETAG;
	token->val[i] = c; 
	i++;
	token->val[i] = '\0';
	flag = 1;
      }
      else if(c=='!'){
	state = token->type = XMLTOKEN_COMMENTBEGIN;
	token->val[i] = c; 
	i++;
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    case XMLTOKEN_INTERROGATION:
      if(c=='>'){
	token->type = XMLTOKEN_HEADEREND;
	token->val[i] = c; 
	i++;
	token->val[i] = '\0';
	flag = 1;
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    case XMLTOKEN_COMMENTBEGIN:
      if(c=='-'){
	token->val[i] = c;
	i++;
	if(i==4){
	  token->val[i] = '\0';
	  flag = 1;
	}
      }
      else{
	Warning("Invalid XML file", "NextXMLToken");
	token->type = XMLTOKEN_ERROR;	
	flag = 2;
      }
      break;
    case XMLTOKEN_MINUS:
      if((c>='0' && c<='9')){
	state = token->type = XMLTOKEN_INT;
	token->val[i] = c;
	i++;
      }      
      else if((c=='.')){
	state = token->type = XMLTOKEN_FLOAT;
	token->val[i] = c;
	i++;
      }
      else if(c=='-'){
	state = token->type = XMLTOKEN_COMMENTEND;
	token->val[i] = c;
	i++;
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    case XMLTOKEN_COMMENTEND:
      if(c=='>'){
	token->val[i] = c;
	i++;
	token->val[i] = '\0';
	flag = 1;
      }
      else{
	Warning("Invalid XML file", "NextXMLToken");
	token->type = XMLTOKEN_ERROR;	
	flag = 2;
      }
      break;
    case XMLTOKEN_DIVIDE:
      if(c=='>'){
	state = token->type = XMLTOKEN_RIGHTCLOSETAG;
	token->val[i] = c;
	i++;
	token->val[i] = '\0';
	flag = 1;	
      }
      else{
	token->val[i] = '\0';
	flag = 2;
      }
      break;
    }
    
    if(flag!=2)
      c = fgetc(file);
  }while(c!=EOF && flag==0);

  return c;
}

XMLNode *ReadXMLTag(FILE *file, int *c, int *closed){
  XMLNode *node;
  XMLToken token;
  char *name,*value;
  char msg[500];

  *c = NextXMLToken(file, &token, *c);
  if(token.type != XMLTOKEN_STRING){
    sprintf(msg,"Invalid XML file (last token=%s)",token.val);
    Warning(msg, "ReadXMLTag");
    return NULL;
  }
  name = (char *)malloc(sizeof(char)*(strlen(token.val)+1));
  strcpy(name, token.val);
  node = NewXMLTag(name);

  *c = NextXMLToken(file, &token, *c);
  if(token.type == XMLTOKEN_BLANK)
    *c = NextXMLToken(file, &token, *c);

  while(token.type!=XMLTOKEN_RIGHTABRACKET && token.type!=XMLTOKEN_RIGHTCLOSETAG){
    if(token.type != XMLTOKEN_STRING){
      sprintf(msg,"Invalid XML file (last token=%s)",token.val);
      Warning(msg, "ReadXMLTag");
      return NULL;
    }
    name = (char *)malloc(sizeof(char)*(strlen(token.val)+1));
    strcpy(name, token.val);

    *c = NextXMLToken(file, &token, *c);
    if(token.type == XMLTOKEN_BLANK)
      *c = NextXMLToken(file, &token, *c);
    if(token.type != XMLTOKEN_EQUAL){
      sprintf(msg,"Invalid XML file (last token=%s)",token.val);
      Warning(msg, "ReadXMLTag");
      return NULL;
    }

    *c = NextXMLToken(file, &token, *c);
    if(token.type == XMLTOKEN_BLANK)
      *c = NextXMLToken(file, &token, *c);
    if(token.type != XMLTOKEN_QUOTED){
      Warning("Invalid XML file: Attribute values must always be quoted", "ReadXMLTag");
      return NULL;
    }
    value = (char *)malloc(sizeof(char)*(strlen(token.val)+1));
    strcpy(value, token.val);

    AddXMLTagAttrib(node, name, value);

    *c = NextXMLToken(file, &token, *c);
    if(token.type == XMLTOKEN_BLANK)
      *c = NextXMLToken(file, &token, *c);
  }

  if(token.type==XMLTOKEN_RIGHTCLOSETAG)
    *closed = 1;
  else
    *closed = 0;

  return node;
}

/* le um arquivo XML */
XMLDocument *ReadXMLDocument(char *filename){
  XMLDocument *doc;
  XMLNode *root, *stack, *new;
  XMLToken token;
  char *encoding,*version,*name,*aux;
  char comment[15000];
  char text[15000];
  int c, closed;
  FILE *file;

  file = fopen(filename,"r");

  encoding = version = NULL;
  c = (int)' ';
  c = NextXMLToken(file, &token, c);
  if(token.type == XMLTOKEN_BLANK)
    c = NextXMLToken(file, &token, c);
  if(token.type != XMLTOKEN_HEADERBEGIN){
    Warning("Invalid XML file, '<?' expected", "ReadXMLDocument");
    printf("\n Token: type = %d, val = %s\n",token.type, token.val);
    fclose(file);
    return NULL;
  }  
  c = NextXMLToken(file, &token, c);
  if(token.type == XMLTOKEN_BLANK)
    c = NextXMLToken(file, &token, c);
  if(token.type != XMLTOKEN_STRING){
    Warning("Invalid XML file, string \"xml\" expected", "ReadXMLDocument");
    printf("\n Token: type = %d, val = %s\n",token.type, token.val);
    return NULL;
  }
  if(strcasecmp(token.val,"xml")!=0){
    Warning("Invalid XML file, string \"xml\" expected", "ReadXMLDocument");
    printf("\n Token: type = %d, val = %s\n",token.type, token.val);
    return NULL;
  }

  c = NextXMLToken(file, &token, c);
  if(token.type == XMLTOKEN_BLANK)
    c = NextXMLToken(file, &token, c);

  while(token.type!=XMLTOKEN_HEADEREND){
    if(token.type != XMLTOKEN_STRING){
      Warning("Invalid XML file, string expected", "ReadXMLDocument");
      return NULL;
    }
    name = (char *)malloc(sizeof(char)*(strlen(token.val)+1));
    strcpy(name, token.val);
  
    c = NextXMLToken(file, &token, c);
    if(token.type == XMLTOKEN_BLANK)
      c = NextXMLToken(file, &token, c);
    if(token.type != XMLTOKEN_EQUAL){
      Warning("Invalid XML file, '=' expected", "ReadXMLDocument");
      return NULL;
    }

    c = NextXMLToken(file, &token, c);
    if(token.type == XMLTOKEN_BLANK)
      c = NextXMLToken(file, &token, c);
    if(token.type != XMLTOKEN_QUOTED){
      Warning("Invalid XML file: Attribute values must always be quoted", "ReadXMLDocument");
      printf("\n Token: type = %d, val = %s\n",token.type, token.val);
      return NULL;
    }

    if(strcasecmp(name,"version")==0){
      free(name);
      version = (char *)malloc(sizeof(char)*(strlen(token.val)+1));
      strcpy(version, token.val);
    }
    else if(strcasecmp(name,"encoding")==0){
      free(name);
      encoding = (char *)malloc(sizeof(char)*(strlen(token.val)+1));
      strcpy(encoding, token.val);
    }
    else{
      free(name);
      Warning("Invalid XML file, invalid attribute", "ReadXMLDocument");
      printf("\n Token: type = %d, val = %s\n",token.type, token.val);
      return NULL;
    }

    c = NextXMLToken(file, &token, c);
    if(token.type == XMLTOKEN_BLANK)
      c = NextXMLToken(file, &token, c);
  }

  if(encoding==NULL || version==NULL){
    Warning("Invalid XML file, encoding or version missing", "ReadXMLDocument");
    return NULL;
  }

  c = NextXMLToken(file, &token, c);
  if(token.type == XMLTOKEN_BLANK)
    c = NextXMLToken(file, &token, c);
  if(token.type != XMLTOKEN_LEFTABRACKET){
    Warning("Invalid XML file, '<' expected", "ReadXMLDocument");
    fclose(file);
    return NULL;
  }

  stack = root = ReadXMLTag(file, &c, &closed);
  if(root==NULL){
    fclose(file);
    return NULL;
  }
  doc = CreateXMLDocument(root, version, encoding);

  if(closed){
    fclose(file);
    return doc;
  }

  c = NextXMLToken(file, &token, c);
  if(token.type == XMLTOKEN_BLANK)
    c = NextXMLToken(file, &token, c);


  while(stack!=NULL){
  
    if(token.type == XMLTOKEN_LEFTABRACKET){
      new = ReadXMLTag(file, &c, &closed);
      AppendXMLTagChild(doc, stack, new);
      if(!closed)
	stack = new;
    }
    else if(token.type == XMLTOKEN_LEFTCLOSETAG){
      c = NextXMLToken(file, &token, c);
      if(token.type == XMLTOKEN_BLANK)
	c = NextXMLToken(file, &token, c);
    
      if(token.type != XMLTOKEN_STRING){
	Warning("Invalid XML file, string expected", "ReadXMLDocument");
	return NULL;
      }
      if(strcmp(stack->contents.tag.name, token.val)==0){
	stack = stack->parent;
      }
      else{
	Warning("Invalid XML file, invalid tag", "ReadXMLDocument");
	return NULL;
      }
      c = NextXMLToken(file, &token, c);
      if(token.type == XMLTOKEN_BLANK)
	c = NextXMLToken(file, &token, c);
    
      if(token.type != XMLTOKEN_RIGHTABRACKET){
	Warning("Invalid XML file, '>' expected", "ReadXMLDocument");
	return NULL;
      }
    }
    else if(token.type == XMLTOKEN_COMMENTBEGIN){
      c = NextXMLToken(file, &token, c);
      comment[0] = '\0';
      while(token.type!=XMLTOKEN_COMMENTEND){
	strcat(comment,token.val);
	c = NextXMLToken(file, &token, c);
      }
      aux = (char *)malloc(sizeof(char)*(strlen(comment)+1));
      strcpy(aux, comment);
      new = NewXMLComment(aux);
      AppendXMLTagChild(doc, stack, new);
    }
    else{
      text[0] = '\0';
      while(token.type!=XMLTOKEN_LEFTABRACKET && token.type!=XMLTOKEN_LEFTCLOSETAG && token.type!=XMLTOKEN_COMMENTBEGIN){
	strcat(text,token.val);
	c = NextXMLToken(file, &token, c);
      }
      aux = (char *)malloc(sizeof(char)*(strlen(text)+1));
      strcpy(aux, text);
      new = NewXMLText(aux);
      AppendXMLTagChild(doc, stack, new);
      continue;
    }

    c = NextXMLToken(file, &token, c);
    if(token.type == XMLTOKEN_BLANK)
      c = NextXMLToken(file, &token, c);
  }

  fclose(file);
  
  return doc;
}



