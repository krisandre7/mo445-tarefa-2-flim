/*
 *
 */

#include "c2tcl.h"

int main(int argc, char *argv[]) {
  FILE *fin, *fout,*headf;
  int i, n, res;
  regex_t myreg;
  char buf[BUFSIZE];
  char *p;
  variable *par[MAXVARS];
  char filename[100];
  

  if (argc != 3) {
    printf("Usage: %s <headerfile> <application>\n",argv[0]);
    return -1;
  }
  
  if ((fin = fopen(argv[1],"r")) == NULL)
    myerror("Error opening input file.");
  
  /* used for include the application file */
  strcpy(filename,"tcl_");
  p = strtok(argv[1],".");
  strcat(filename,p);
  strcat(filename,".c");
  /*---------------------------------------*/
  
  if ((fout = fopen(filename,"w+")) == NULL)
    myerror("Error opening output file");
  
/*
 * if ((res = regcomp(&myreg, "\\w+[ \t]+\\**[ \t]*\\w+[ \t]*\\(+.*\\)+[ \t]*;",
 *		     REG_EXTENDED | REG_NOSUB)) != 0) {
 */
  if ((res = regcomp(&myreg, "\\w+[ \t]+\\**[ \t]*.+[ \t]*\\(+.*\\)+[ \t]*;",
		     REG_EXTENDED | REG_NOSUB)) != 0) {
    regerror(res, &myreg, buf, BUFSIZE);
    myerror(buf);
  }

  /* Header */
  fprintf(fout, 
	  "/********************************************************\n"
	  " * This file was automatic generated with c2tcl utility *\n"
	  " * ---------------------------------------------------- *\n"
	  " * By: Andre Carvalho  (andre@fee.unicamp.br)           *\n"
	  " *     Carlos Camolesi (coliveir@fee.unicamp.br)        *\n"
	  " * ---------------------------------------------------- *\n"
	  " * USE IT AT YOUR OWN RISK, NO WARRANTY                 *\n"
	  " ********************************************************/\n\n\n");
  
 
  fprintf(fout,
	  "/* This APPLICATION is yours */\n"
          "#include \"tcl_%s.h\"\n\n",argv[2]);
  
  while (!feof(fin)) {
    fexgets(buf, BUFSIZE, fin);
    if (!regexec(&myreg, buf, 0, NULL, 0)) {
      p = buf;
      
        par[0] = getVar(&p);   // Get name and return type of function...

      stripSpace(&p);
      if (*p != '(')
	myerror("Not good, did not found '(' after function name.\n");
      
      p++; // don't want the '('... 

      stripSpace(&p); // to first parameter.
      
      i = 0;
      while ((i < MAXVARS) && (*p != ')')) {
	stripSpace(&p);
	par[++i] = getVar(&p);
	stripSpace(&p);
	if (*p == ',')
	  p++; // next parameter
	else if (*p != ')')
	  myerror("Error, expecting comma or ')', but got something else!\n");
      }
      
      if (i == MAXVARS-1)
	printf("Warning, i has reached MAXVARS value, may have errors\n \
	       @: %s\n", buf);
      n = i+1; // save the number of params
      
      // Only do if is a valid type.
      if (!noValid(n,par)) {
	// Function header
	fprintf(fout, "int ET_COMMAND_%s(ET_TCLARGS) {\n", par[0]->name);
	
      // Declaration of variable of return.
	if (par[0]->conv != TVOID)

	  fprintf(fout, "  %sret_%s;\n",par[0]->type, par[0]->name);

      // Declaration of variables of params.
	for (i = 1; i < n; i++)
	  fprintf(fout, "  %s%s;\n",par[i]->type, par[i]->name);

      // Check number of args passed by Tcl...
	fprintf(fout, "\n  if (argc != %d) {\n",n);
	fprintf(fout, "    Et_ResultF(interp, \"usage: %%s");
	for (i = 1; i < n; i++)
	  fprintf(fout, " <%s>", par[i]->name);
	fprintf(fout,"\\n\", argv[0]);\n    return TCL_ERROR;\n  }\n\n");

	// Set values of params.
	for (i = 1; i < n; i++) {
	  fprintf(fout, "  %s = ",par[i]->name);
	  if (par[i]->isptr)
	    fprintf(fout, "(%s) ", par[i]->type);
	  sprintf(buf, "argv[%d]", i);
	  fprintf(fout, varConv[par[i]->conv], buf);
	  fprintf(fout, ";\n");
	}

	if (par[0]->conv != TVOID)
	  fprintf(fout, "  ret_%s =", par[0]->name);
	fprintf(fout, "  %s(",par[0]->name);

	for(i=1; i<n; i++) {
	  if (par[i]->sptr == 1)
	    fprintf(fout, "&");
	  fprintf(fout, "%s", par[i]->name);
	  if (i < (n-1))
	    fprintf(fout,", ");
	}
	fprintf(fout,");\n\n");

	if (par[0]->conv != TVOID)
	  fprintf(fout,"  Et_ResultF(interp, \"%%%c\", ret_%s);\n\n", 
		  par[0]->chprintf, par[0]->name);

	fprintf(fout, "  return TCL_OK;\n} /* %s */\n\n\n", par[0]->name);
      
	for (i = 0; i < n; i++) {
	  free(par[i]->name);
	  free(par[i]->type);
	  free(par[i]);           // dont need these anymore!
	}
      }
    }
  }
  
  
  /* EOF */
  fprintf(fout, 
	  "/********************************************************\n"
	  " * This file was automatic generated with c2tcl utility *\n"
	  " * ---------------------------------------------------- *\n"
	  " * By: Andre Carvalho  (andre@fee.unicamp.br)           *\n"
	  " *     Carlos Camolesi (coliveir@lfee.unicamp.br)       *\n"
	  " * ---------------------------------------------------- *\n"
	  " * USE IT AT YOUR OWN RISK, NO WARRANTY         [E.O.F] *\n"
	  " ********************************************************/\n\n\n");

  fclose(fin);
  fclose(fout);
 
  regfree(&myreg);
  
  return 0;
  
  
}

/**********************************************************************
 * Functions
 **********************************************************************/

void stripSpace(char **buf)
{
  char *b;

  b = *buf;

  while ((*b == ' ') || (*b == '\t'))
    b++;

  *buf = b;
} /* stripSpace */


variable *getVar(char **buf)
{
  char *b;
  variable *v;
  char tmp[1024];
  int i, n, isptr=0,sptr=0;

  if ((v = (variable *)calloc(1,sizeof(variable))) == NULL)
    myerror("error allocating memory");

  b = *buf;
  i = 0;
  while ((*b != ' ') && (*b != '\t')) {
    tmp[i++] = *(b++);
  }
  
  tmp[i++] = ' ';

  stripSpace(&b);

  if (*b == '*') {
    isptr = 1;
    do {
      sptr++;
    } while (*(++b) == '*');
  }
  
  
  if (isptr) 
    if (sptr > 1) {
      while (sptr > 1) {
	tmp[i++] = '*';
	sptr--;
      }
    } else {
      sptr--;
      tmp[i++] = '*';
    }
    
    
  
  
  
  
  
  v->isptr = isptr;
  v->sptr = sptr;
  


  tmp[i++] = 0;
  
  if ((v->type = (char *)malloc(i)) == NULL)
    myerror("error allocating memory");
  strncpy(v->type, tmp, i);

  if (!strcmp(tmp, "int ")) {
    v->conv = TINTEGER;
    v->chprintf = CHINTEGER;
  }
  else if (!strcmp(tmp, "bool ")) {
    v->conv = TBOOL;
    v->chprintf = CHBOOL;
  }
  else if (!strcmp(tmp, "long ")) {
    v->conv = TLONG;
    v->chprintf = CHLONG;
  }
  else if (!strcmp(tmp, "float ")) {
    v->conv = TFLOAT;
    v->chprintf = CHFLOAT;
  }
  else if (!strcmp(tmp, "double ")) {
    v->conv = TDOUBLE;
    v->chprintf = CHDOUBLE;
  }
  else if (!strcmp(tmp, "void ")) {
    v->conv = TVOID;
    v->chprintf = CHVOID;
  }
  else if (!strcmp(tmp, "char ")) {
    v->conv = TCHAR;
    v->chprintf = CHCHAR;
  }
  else if (!strcmp(tmp, "char *")) {
    v->conv = TSTRING;
    v->chprintf = CHSTRING;
  }
  else if (!strcmp(tmp, "uchar ")) {
    v->conv = TCHAR;
    v->chprintf = CHCHAR;
  }
  else if (!strcmp(tmp, "uint ")) {
    v->conv = TINTEGER;
    v->chprintf = CHUNSIGNED;
  }
  else if (!strcmp(tmp, "ushort ")) {
    v->conv = TINTEGER;
    v->chprintf = CHUNSIGNED;
  }

  else if (isptr) {
    v->conv = TPOINTER;
    v->chprintf = CHPOINTER;
  }
  
  stripSpace(&b);

  i = 0;
  while ((*b != ' ') && (*b != '\t') && (*b != ',') &&
	 (*b != '(') && (*b != ')')) {
    if (*b == '[') {
      v->type = "not supported";
    }
    tmp[i++] = *(b++);
  }
  
  tmp[i++] = 0;

  if ((v->name = (char *)malloc(i)) == NULL)
    myerror("error allocating memory");

  strncpy(v->name, tmp, i);

  *buf = b;
  return v;
}

void fexgets(char *fbuf, int size, FILE *f)
{
  char ch, lastch;
  int i = 0, n;

  n = size - 1;
  ch = lastch = 0;
  while((ch != EOF) && (i < n)) {
    ch = fgetc(f);
    if (ch == '\n') {
      if (lastch == '\\')  /* if very last char in line is '\', continue */
	i--;               /* in the next line! */
      else
	ch = EOF;  /* get out! */
    } else if (ch != EOF)
      fbuf[i++] = ch;
    lastch = ch;
  }
  fbuf[i] = 0;
}

void myerror(const char *msg)
{
  fprintf(stderr, "error: %s\n", msg);
  exit(-1);
}

int noValid(int n, variable *par[MAXVARS]) {
  int i,novalid = 0;
  
  // If the type if valid...
  for (i = 1; i < n; i++)
    if((strcmp(par[i]->type, "int ")) &&
       (strcmp(par[i]->type, "bool ")) &&
       (strcmp(par[i]->type, "long ")) &&
       (strcmp(par[i]->type, "float ")) && 
       (strcmp(par[i]->type, "double ")) && 
       (strcmp(par[i]->type, "void ")) &&
       (strcmp(par[i]->type, "char ")) &&
       (strcmp(par[i]->type, "char *")) &&
       (strcmp(par[i]->type, "uint ")) &&
       (strcmp(par[i]->type, "uchar ")) &&
       (strcmp(par[i]->type, "ushort ")) &&
       (!par[i]->isptr)) {
      novalid = 1;
      break;
    }
  
  return novalid; 
}
