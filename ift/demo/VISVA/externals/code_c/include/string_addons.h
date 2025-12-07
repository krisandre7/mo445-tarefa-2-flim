
#ifndef _STRING_ADDONS_H_
#define _STRING_ADDONS_H_

#include "oldift.h"

//Removes the leading and trailing white space.
void      TrimString(char *str);
void      SubString(char *str,
		    int beginIndex,
		    int endIndex);
void      ReplaceStringCharacter(char *str,
				 char old_c,
				 char new_c);

#endif



