/*********************************************************
 * c2tcl - Header file.
 * 
 *
 *********************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <regex.h>


char *varConv[] = {
  "atoi(%s)",    /* Integer, pointer and bool */
  "atof(%s)",    /* Float and Double*/
  "atol(%s)",    /* long */
  "%s",          /* string (char *) */
  "(char)atoi(%s)"     /* char */
};

#define BUFSIZE 1024
#define MAXVARS 10

typedef enum{TINTEGER, TFLOAT, TLONG, TSTRING, TCHAR} convType; 
#define TVOID 666
#define TPOINTER TINTEGER
#define TBOOL TINTEGER
#define TDOUBLE TFLOAT

#define CHINTEGER 'd'
#define CHFLOAT 'f'
#define CHSTRING 's'
#define CHCHAR 'c'
#define CHVOID '\0'
#define CHLONG CHINTEGER
#define CHPOINTER CHINTEGER
#define CHDOUBLE CHFLOAT
#define CHBOOL CHINTEGER
#define CHUNSIGNED 'u'

typedef struct {
  char *type;
  char *name;
  char chprintf;
  convType conv;
  int isptr;
  int sptr;
} variable;


void stripSpace(char **buf);
variable *getVar(char **buf);
void fexgets(char *fbuf, int size, FILE *f);
void myerror(const char *msg);
int onVadlid(int n,variable *par[MAXVARS]);
