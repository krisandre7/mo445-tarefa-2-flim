// version 00.00.03

#ifndef _FILELIST_H_
#define _FILELIST_H_

#include "oldift.h"
#include "arraylist.h"
#include "string_addons.h"

typedef struct _fileList {
  ArrayList *A;
  int n;   //Number of files added.
} FileList;


typedef struct _fileListIndex {
  int *A;
  int n;   //Number of indexes added.
  //private:
  int cap;
} FileListIndex;


FileList *CreateFileList(int cap);
void      DestroyFileList(FileList **L);

FileList *ReadFileList(char *filename);
void      WriteFileList(FileList *L,
			char *filename);

void      AddFile(FileList *L, char *file);
char     *GetFile(FileList *L, int index);
bool      HasFile(FileList *L, char *file);

FileList *SubFileList(FileList *L,
		      FileListIndex *I);

//path:      Path where the files are,
//basename:  Basename of the files,
//first:     Number of the first file,
//last:      Number of the last file,
//ext:       File extension.
void  AddFiles(FileList *L,
	       char *path, 
	       char *basename, 
	       int first, int last,
	       char *ext);


//It shuffles the files at random.
//Should use "void srand(unsigned int seed);" before calling.
void  RandomizeFileList(FileList *L);

void  ResizeFileList(FileList *L, int n);

//Trims the capacity of this FileList instance 
//to be the list's current size.
void  Trim2SizeFileList(FileList *L);


bool      FileExists(char *file);
void      RemoveFileDirectory(char *file);
void      RemoveFileExtension(char *file);
void      MergeRelativePath(char *dir, char *rel);

//---------------------------------------

FileListIndex *CreateFileListIndex(int cap);
void           DestroyFileListIndex(FileListIndex **I);

FileListIndex *ReadFileListIndex(char *filename);
void           WriteFileListIndex(FileListIndex *I,
				  char *filename);

void  AddFileIndex(FileListIndex *I, int index);
int   GetFileIndex(FileListIndex *I, int i);
bool  HasFileIndex(FileListIndex *I, int index);

#endif

