#include "csf.h"

void Force2ClustersCSF( Subgraph *sg, Scene *scn, Scene *mask, float T ) {
  int p, n_obj1, n_obj2;
  uint *mean = NULL, *size = NULL;
  
  printf("It found %d clusters\n",sg->nlabels);

  if ( sg->nlabels < 2 ) {
    printf( "ERROR!!! Graph with only one cluster !\n" );
    return;
  }
  mean = AllocUIntArray( sg->nlabels );
  size = AllocUIntArray( sg->nlabels );
  n_obj1 = 0;
  n_obj2 = 0;
  // compute mean brightness and size of every label region  
  for ( p = 0; p < sg->nnodes; p++ ) {
    mean[ sg->node[ p ].label ] += scn->data[ sg->node[ p ].position ];
    size[ sg->node[ p ].label ] += 1;
  }
  for ( p = 0; p < sg->nlabels; p++ ) mean[ p ] /= size[ p ];


  // define csf and gmwm regions
  for ( p = 0; p < sg->nnodes; p++ ) {    
    if ( mean[ sg->node[ p ].label ] <= T ) {
      sg->node[ p ].label = 1; // CSF
      n_obj1++;
    }
    else {
      sg->node[ p ].label = 0;
      n_obj2++;
    }
  }
  sg->nlabels = 2;
  free( mean );
  free( size );
}

int CSFThreshold(Scene *scn, Scene *mask)
{
  int p,n=scn->xsize*scn->ysize*scn->zsize;
  uint mean1=0,np1=0;
  uint mean2=0,np2=0;

  for (p=0; p < n; p++) 
    if (mask->data[p]){
      mean1 += scn->data[p];
      np1++;
    }

  mean1 = (mean1/np1);
  
  for (p=0; p < n; p++) {
    if ((mask->data[p])&&(scn->data[p]<=(int)mean1)){
      mean2 += scn->data[p];
      np2++;
    }
  }
  mean2 = (mean2/np2);
  
  return((int)(mean2));
  
}

