
#include "objgroup.h"


/* Find a maximal clique containing element 'i' in a graph 
with 'n' vertices defined by an adjacency matrix A.*/
int *MaximalClique(int **A, int n, int i){
  int  *label=AllocIntArray(n);
  int   lin,col,k;
  char  allow;

  /* label maximal clique using allowed arcs */
  lin = i;
  label[lin]=1;

  for(col=0; col<n; col++){
    if(col==lin) continue;
    if(A[lin][col]>0){ /* allowed */
      allow=1;
      for(k=0; k<n; k++) {
	if((A[col][k]==0)&&
	   (label[lin]==label[k])){ /* not allowed */
	  allow=0; break;
	}
      }
      if(allow)
	label[col]=label[lin];
    }
  }
  return (label);
}


/* Find groups of shapes with intra-similarities above a threshold
   within (0,1) */
void GroupsByMaximalCliques(RealMatrix *arcweights,
			    FileList *L, float simil,
			    char *outputdir){
  int   nimages = L->n;
  int **A=(int **)calloc(nimages,sizeof(int *));
  int **C=(int **)calloc(nimages,sizeof(int *));
  int **B=(int **)calloc(nimages,sizeof(int *));
  int  *S=AllocIntArray(nimages);
  int lin,col,i,j,s,nleft,nleft_prev,g;
  bool finish,flag=true;
  FileList *Gs=NULL;
  FileListIndex *G=NULL;
  char filename[512];
  char out[512];

  strcpy(out, outputdir);
  s = strlen(out);
  if(s>0 && out[s-1]=='/') out[s-1] = '\0';

  /* create adjacency matrix */
  for(lin=0; lin<nimages; lin++){
    A[lin] = AllocIntArray(nimages);
    for(col=0; col<nimages; col++)
      if(arcweights->val[lin][col]>=simil)
	A[lin][col]=1;
  }

  for(lin=0; lin<nimages; lin++){
    C[lin] = MaximalClique(A, nimages, lin);
    B[lin] = AllocIntArray(nimages);
    memcpy(B[lin], C[lin], nimages*sizeof(int));
    S[lin] = NIL;
  }

  nleft_prev = nimages;
  do{
    /*
    printf("Step 1: ");
    for(lin=0; lin<nimages; lin++)
      printf(" %2d ",S[lin]);
    printf("\n");
    */

    //Elimina cliques (linhas) repetidas ou contidas em outras.
    for(lin=0; lin<nimages; lin++){
      if(S[lin]!=NIL) continue;

      for(i=0; i<nimages; i++){
		if(i==lin || S[i]!=NIL) continue;
		flag = true;
		for(j=0; j<nimages; j++){
			if(C[lin][j]==0 && C[i][j]==1){
				flag=false; 
				break;
			}
		}
		if(flag){ 
			S[i] = 0;
			for(j=0; j<nimages; j++) C[i][j] = 0;
		}
      }
    }

    /*
    printf("Step 2: ");
    for(lin=0; lin<nimages; lin++)
      printf(" %2d ",S[lin]);
    printf("\n");
    */

    //Seleciona cliques (linhas) que possuem ao menos 1 elemento
    //que nao aparece nas outras.
    for(lin=0; lin<nimages; lin++){
      if(S[lin]!=NIL) continue;

      for(j=0; j<nimages; j++){
	if(C[lin][j]==0){
	  flag = false;
	  continue;
	}
	flag = true;
	for(i=0; i<nimages; i++){
	  if(i==lin || S[i]!=NIL) continue;
	  if(C[i][j]==1){ 
	    flag = false; 
	    break; 
	  }
	}
	if(flag) break;
      }
      if(flag){ 
	S[lin] = 1;
	//Apaga todos elementos da clique selecionada.
	for(j=0; j<nimages; j++){
	  if(C[lin][j]==0) continue;
	  for(i=0; i<nimages; i++)
	    C[i][j] = 0;
	}
      }
    }

    /*
    printf("Step 3: ");
    for(lin=0; lin<nimages; lin++)
      printf(" %2d ",S[lin]);
    printf("\n");
    */

    //Elimina cliques (linhas) que possuem todos elementos
    //ja selecionados em outros grupos.

    for(lin=0; lin<nimages; lin++){
      if(S[lin]!=NIL) continue;
      flag = true;
      for(j=0; j<nimages; j++)
	if(C[lin][j]==1) flag = false;
      if(flag) S[lin] = 0;
    }

    /*
    printf("Step 4: ");
    for(lin=0; lin<nimages; lin++)
      printf(" %2d ",S[lin]);
    printf("\n");
    */

    //Verifica se falta analisar alguma clique (linha).
    nleft = 0;
    finish = true;
    for(lin=0; lin<nimages; lin++)
      if(S[lin]==NIL){ finish=false; nleft++; }

    //Ciclo detectado:
    if(nleft==nleft_prev){
      printf("Ciclo detectado\n"); 
      for(lin=0; lin<nimages; lin++)
	if(S[lin]==NIL){ S[lin] = 1; break; }
    }
    nleft_prev = nleft;

  }while(!finish);


  //Grava grupos:
  g = 0;
  Gs = CreateFileList(nimages);
  for(lin=0; lin<nimages; lin++){
    if(S[lin]!=1) continue;

    G = CreateFileListIndex(nimages);

    for(j=0; j<nimages; j++)
      if(B[lin][j]==1)
	AddFileIndex(G, j);

    sprintf(filename,"%s/group%03d.txt",out,g+1);
    AddFile(Gs, filename);
    WriteFileListIndex(G, filename);
    DestroyFileListIndex(&G);
    g++;
  }
  sprintf(filename,"%s/groups.txt",out);
  WriteFileList(Gs, filename);
  printf("ngroups: %d\n",g);

  /* free memory */
  DestroyFileList(&Gs);
  for(lin=0; lin<nimages; lin++) free(A[lin]);
  free(A);
  for(lin=0; lin<nimages; lin++) free(C[lin]);
  free(C);
  for(lin=0; lin<nimages; lin++) free(B[lin]);
  free(B);
  free(S);
}



RealMatrix *ShapeSimilarityMatrix3(FileList *L){
  RealMatrix *M;
  Scene *scn[2];
  int lin,col,nimages = L->n;

  M = CreateRealMatrix(nimages, nimages);

  /* build arc-weights matrix */

  for(lin=0; lin<nimages; lin++){
    scn[0] = ReadVolume( GetFile(L, lin) );
    for(col=lin; col<nimages; col++) {
      if(col==lin)
	M->val[lin][col] = 1.0;
      else{
	scn[1] = ReadVolume( GetFile(L, col) );
	M->val[lin][col] = CentrDiceSimilarity3(scn[0],
						scn[1]);
	M->val[col][lin] = M->val[lin][col];
	DestroyScene(&scn[1]);
      }
    }
    DestroyScene(&scn[0]);
  }
  return M;
}



RealMatrix *MObjSimilarityMatrix3(FileList *L){
  RealMatrix *M;
  Scene *scn[2];
  int lin,col,nimages = L->n;

  M = CreateRealMatrix(nimages, nimages);

  /* build arc-weights matrix */

  for(lin=0; lin<nimages; lin++){
    scn[0] = ReadVolume( GetFile(L, lin) );
    for(col=lin; col<nimages; col++) {
      if(col==lin)
	M->val[lin][col] = 1.0;
      else{
	scn[1] = ReadVolume( GetFile(L, col) );
	M->val[lin][col] = CentrMObjDiceSimilarity3(scn[0],
						    scn[1]);
	M->val[col][lin] = M->val[lin][col];
	DestroyScene(&scn[1]);
      }
    }
    DestroyScene(&scn[0]);
  }
  return M;
}


