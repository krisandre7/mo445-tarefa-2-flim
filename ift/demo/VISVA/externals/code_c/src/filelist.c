#include "filelist.h"


FileList *CreateFileList(int cap){
  FileList *L;

  L = (FileList *) calloc(1,sizeof(FileList));
  if(L == NULL)
    Error(MSG1,"CreateFileList");

  L->A = CreateArrayList(cap);
  L->n = 0;

  return L;
}


void      DestroyFileList(FileList **L){
  FileList *aux;

  aux = *L;
  if(aux != NULL){
    DestroyArrayList(&aux->A);
    free(aux);
    *L = NULL;
  }
}


FileList *ReadFileList(char *filename){
  char dir[512];
  char rel[512];
  char name[512];
  char msg[512];
  FileList *L;
  FILE *fp;
  int ret;

  fp = fopen(filename,"r");
  if(fp==NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"ReadFileList");
  }

  L = CreateFileList(100);

  dir[0]='\0';
  while(1){
    ret = fscanf(fp," %[^\n]", rel);
    if(ret==EOF)
      break;

    TrimString(rel);

    //Ignore a comment line:
    if(rel[0]=='#')
      continue;

    //Change base dir:
    if(rel[0]=='/'){
      if(rel[1]=='/'){
	rel[0] = ' ';
	rel[1] = ' ';
	TrimString(rel);
	strcpy(dir, rel);
	continue;
      }
    }

    //Add relative or absolute path:
    if(dir[0]!='\0'){
      strcpy(name, dir);
      MergeRelativePath(name, rel);
    }
    else
      strcpy(name, rel);
    AddFile(L, name);
  }
  fclose(fp);
  return L;
}


void      WriteFileList(FileList *L,
			char *filename){
  char msg[512];
  FILE *fp;
  int i;
  
  fp = fopen(filename,"w");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"WriteFileList");
  }

  for(i=0; i<L->n; i++)
    fprintf(fp,"%s\n",GetFile(L, i));

  fclose(fp);
}


void      AddFile(FileList *L, char *file){
  char *aux,*trim;
  int s;

  s = strlen(file);
  trim = (char *) calloc(s+1,sizeof(char));
  strcpy(trim, file);
  
  TrimString(trim);
  s = strlen(trim);

  aux = (char *) calloc(s+1,sizeof(char));
  strcpy(aux, trim);
  free(trim);
  L->n++;

  AddArrayListElement(L->A, (void *)aux);
}


char     *GetFile(FileList *L, int index){
  return (char *)GetArrayListElement(L->A, index);
}


bool      HasFile(FileList *L, char *file){
  char *file_i;
  int i;

  for(i=0; i<L->n; i++){
    file_i = (char *)GetArrayListElement(L->A, i);
    if(strcmp(file_i,file)==0) return true;
  }
  return false;
}


void  AddFiles(FileList *L,
	       char *path, 
	       char *basename, 
	       int first, int last,
	       char *ext){
  char name[512];
  int nfiles,i,s;

  s = strlen(path);
  if(s>0)
    if(path[s-1]=='/')
      path[s-1] = '\0';

  nfiles = last-first+1;
  for(i=0; i<nfiles; i++){
    sprintf(name,"%s/%s%03d%s",
	    path, basename, i+first, ext);
    AddFile(L, name);
  }
}


//It shuffles the files at random.
//Should use "void srand(unsigned int seed);" before calling.
void  RandomizeFileList(FileList *L){
  int i,j;
  void *tmp;
  
  for(i=0; i<L->n; i++){
    j = RandomInteger(0, L->n-1);

    tmp = L->A->array[i];
    L->A->array[i] = L->A->array[j];
    L->A->array[j] = tmp;
  }
}

void  ResizeFileList(FileList *L, int n){
  L->n = MIN(n, L->n);
  ResizeArrayList(L->A, n);
}

//Trims the capacity of this FileList instance 
//to be the list's current size.
void  Trim2SizeFileList(FileList *L){
  Trim2SizeArrayList(L->A);
}


bool FileExists(char *file){
  FILE *fp=NULL;
  fp = fopen(file,"r");
  if(fp == NULL) 
    return false;
  else{
    fclose(fp);
    return true;
  }
}


void   RemoveFileExtension(char *file){
  int n = strlen(file);

  while(n>0){
    n--;
    if(file[n]=='/') break;
    if(file[n]=='.') file[n] = '\0';
  }
}


void   RemoveFileDirectory(char *file){
  int n,i;

  n = strlen(file);
  while(n>0){
    n--;
    if(file[n]=='/') break;
  }
  
  if(file[n]=='/') n++;

  i = 0;
  while(file[n]!='\0'){
    file[i] = file[n];
    i++; n++;
  }
  file[i] = '\0';
}



//the "dir" string must have enough space for the result.
void MergeRelativePath(char *dir, char *rel){
  int s;
  
  s = strlen(dir);
  if(s>0 && dir[s-1]=='/')
    dir[s-1] = '\0';
  
  while(rel[0]=='.' || rel[0]=='/'){
    if(rel[1]=='.'){
      s = strlen(dir);
      while(s>0){
	s--;
	if(dir[s]=='/'){
	  dir[s]='\0';
	  break;
	}
      }
      rel++;
    }
    rel++;
  }
  strcat(dir,"/");
  strcat(dir,rel);
}



FileList *SubFileList(FileList *L,
		      FileListIndex *I){
  FileList *S=NULL;
  int i,index;
  S = CreateFileList(I->n);
  for(i=0; i<I->n; i++){
    index = GetFileIndex(I, i);
    AddFile(S, GetFile(L, index));
  }
  return S;
}


//--------------------------------------


FileListIndex *CreateFileListIndex(int cap){
  FileListIndex *I=NULL;

  I = (FileListIndex *) calloc(1,sizeof(FileListIndex));
  if(I == NULL)
    Error(MSG1,"CreateFileListIndex");

  I->A = AllocIntArray(cap);
  I->cap = cap;
  I->n = 0;

  return I;  
}


void           DestroyFileListIndex(FileListIndex **I){
  FileListIndex *aux;

  aux = *I;
  if(aux != NULL){
    free(aux->A);
    free(aux);
    *I = NULL;
  }
}


FileListIndex *ReadFileListIndex(char *filename){
  char msg[512];
  FileListIndex *I=NULL;
  FILE *fp;
  int ret,index;

  fp = fopen(filename,"r");
  if(fp==NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"ReadFileListIndex");
  }

  I = CreateFileListIndex(100);

  while(1){
    ret = fscanf(fp," %d", &index);
    if(ret==EOF)
      break;

    AddFileIndex(I, index);
  }
  fclose(fp);
  return I;
}


void           WriteFileListIndex(FileListIndex *I,
				  char *filename){
  char msg[512];
  FILE *fp;
  int i;
  
  fp = fopen(filename,"w");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"WriteFileList");
  }

  for(i=0; i<I->n; i++)
    fprintf(fp,"%d\n",GetFileIndex(I, i));

  fclose(fp);
}


void  AddFileIndex(FileListIndex *I, int index){
  if(I->n+1<I->cap){
    I->cap = ROUND(I->cap*1.25)+1;
    I->A = (int *)realloc(I->A, I->cap*sizeof(int));
    if(I->A == NULL)
      Error(MSG1,"AddFileIndex");
  }
  I->A[I->n] = index;
  I->n += 1;
}


int   GetFileIndex(FileListIndex *I, int i){
  if(i<I->n && i>=0)
    return I->A[i];
  else
    return NIL;
}


bool HasFileIndex(FileListIndex *I, int index){
  int i;

  for(i=0; i<I->n; i++){
    if(I->A[i]==index) return true;
  }
  return false;
}


