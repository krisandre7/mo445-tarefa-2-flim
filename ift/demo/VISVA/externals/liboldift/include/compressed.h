
#ifndef _COMPRESSED_H_
#define _COMPRESSED_H_

#include "oldift.h"

#define COMPRESSED_TYPE 0
#define NORMAL_TYPE 1

void   WriteCompressedScene(Scene *scn, 
			    char *filename);
Scene *ReadCompressedScene(char *filename);

// Get the filename by the basename. Tries .scn.bz2 first and returns 
// COMPRESSED_TYPE, if found. Then tries .scn, returning NORMAL_TYPE, 
// if found. If no file was found return -1.
int GetFilebyBaseName(char *basename, char *filename);

// Read/Write accordingly to the file extension.
Scene *ReadVolume(char *filename);
void   WriteVolume(Scene *scn, char *filename);
#endif

