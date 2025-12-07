#include "oldift.h"
#include "compressed.h"

// 0 - l->r sagittal; 1 - r->l sagittal; 2 - u->d axial; 3 - d->u axial; 4 - f->b coronal; 5 - b->f coronal
// output: xdir - 0; ydir - 2
Scene *RotateAxis( Scene *scn, int xdir, int ydir ) 
{
  Scene *res=NULL;
  int i,j,k,p,q;

  switch( xdir ) {
  case 0 :
    switch( ydir ) {
    case 2 :
      res = CopyScene( scn );
      break;
    case 3 :
      res = CreateScene( scn->xsize, scn->ysize, scn->zsize );
      res->dx = scn->dx;
      res->dy = scn->dy;
      res->dz = scn->dz;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - k - 1 ] + res->tby[ res->ysize - j - 1 ] + i; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 4 :
      res = CreateScene( scn->xsize, scn->zsize, scn->ysize );
      res->dx = scn->dx;
      res->dy = scn->dz;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ j ] + res->tby[ res->ysize - k - 1 ] + i; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    default:
      res = CreateScene( scn->xsize, scn->zsize, scn->ysize );
      res->dx = scn->dx;
      res->dy = scn->dz;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - j - 1 ] + res->tby[ k ] + i; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    }
    break;

  case 1 :
    switch( ydir ) {
    case 2 :
      res = CreateScene( scn->xsize, scn->ysize, scn->zsize );
      res->dx = scn->dx;
      res->dy = scn->dy;
      res->dz = scn->dz;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - k - 1 ] + res->tby[ j ] + res->xsize - i - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 3 :
      res = CreateScene( scn->xsize, scn->ysize, scn->zsize );
      res->dx = scn->dx;
      res->dy = scn->dy;
      res->dz = scn->dz;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ k ] + res->tby[ res->ysize - j - 1 ] + res->xsize - i - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 4 :
      res = CreateScene( scn->xsize, scn->zsize, scn->ysize );
      res->dx = scn->dx;
      res->dy = scn->dz;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ j ] + res->tby[ k ] + res->xsize - i - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    default:
      res = CreateScene( scn->xsize, scn->zsize, scn->ysize );
      res->dx = scn->dx;
      res->dy = scn->dz;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - j - 1 ] + res->tby[ res->ysize - k - 1 ] + res->xsize - i - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    }
    break;

  case 2 :
    switch( ydir ) {
    case 0 :
      res = CreateScene( scn->ysize, scn->xsize, scn->zsize );
      res->dx = scn->dy;
      res->dy = scn->dx;
      res->dz = scn->dz;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - k - 1 ] + res->tby[ i ] + j; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 1 :
      res = CreateScene( scn->ysize, scn->xsize, scn->zsize );
      res->dx = scn->dy;
      res->dy = scn->dx;
      res->dz = scn->dz;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ k ] + res->tby[ i ] + res->xsize - j - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 4 :
      res = CreateScene( scn->zsize, scn->xsize, scn->ysize );
      res->dx = scn->dz;
      res->dy = scn->dx;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ j ] + res->tby[ i ] + k; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    default:
      res = CreateScene( scn->zsize, scn->xsize, scn->ysize );
      res->dx = scn->dz;
      res->dy = scn->dx;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - j - 1 ] + res->tby[ i ] + res->xsize - k - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    }
    break;

  case 3 :
    switch( ydir ) {
    case 0 :
      res = CreateScene( scn->ysize, scn->xsize, scn->zsize );
      res->dx = scn->dy;
      res->dy = scn->dx;
      res->dz = scn->dz;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ k ] + res->tby[ res->ysize - i - 1 ] + j; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 1 :
      res = CreateScene( scn->ysize, scn->xsize, scn->zsize );
      res->dx = scn->dy;
      res->dy = scn->dx;
      res->dz = scn->dz;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - k - 1 ] + res->tby[ res->ysize - i - 1 ] + res->xsize - j - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 4 :
      res = CreateScene( scn->zsize, scn->xsize, scn->ysize );
      res->dx = scn->dz;
      res->dy = scn->dx;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ j ] + res->tby[ res->ysize - i - 1 ] + res->xsize - k - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    default:
      res = CreateScene( scn->zsize, scn->xsize, scn->ysize );
      res->dx = scn->dz;
      res->dy = scn->dx;
      res->dz = scn->dy;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - j - 1 ] + res->tby[ res->ysize - i - 1 ] + k; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    }
    break;

  case 4 :
    switch( ydir ) {
    case 0 :
      res = CreateScene( scn->ysize, scn->zsize, scn->xsize );
      res->dx = scn->dy;
      res->dy = scn->dz;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ i ] + res->tby[ k ] + j; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 1 :
      res = CreateScene( scn->ysize, scn->zsize, scn->xsize );
      res->dx = scn->dy;
      res->dy = scn->dz;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ i ] + res->tby[ res->ysize - k - 1 ] + res->xsize - j - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 2 :
      res = CreateScene( scn->zsize, scn->ysize, scn->xsize );
      res->dx = scn->dz;
      res->dy = scn->dy;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ i ] + res->tby[ j ] + res->xsize - k - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    default:
      res = CreateScene( scn->zsize, scn->ysize, scn->xsize );
      res->dx = scn->dz;
      res->dy = scn->dy;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ i ] + res->tby[ res->ysize - j - 1 ] + k; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    }
    break;

  default:
    switch( ydir ) {
    case 0 :
      res = CreateScene( scn->ysize, scn->zsize, scn->xsize );
      res->dx = scn->dy;
      res->dy = scn->dz;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - i - 1 ] + res->tby[ res->ysize - k - 1 ] + j; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 1 :
      res = CreateScene( scn->ysize, scn->zsize, scn->xsize );
      res->dx = scn->dy;
      res->dy = scn->dz;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - i - 1 ] + res->tby[ k ] + res->xsize - j - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    case 2 :
      res = CreateScene( scn->zsize, scn->ysize, scn->xsize );
      res->dx = scn->dz;
      res->dy = scn->dy;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - i - 1 ] + res->tby[ j ] + k; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    default:
      res = CreateScene( scn->zsize, scn->ysize, scn->xsize );
      res->dx = scn->dz;
      res->dy = scn->dy;
      res->dz = scn->dx;
      for( i = 0; i < scn->xsize; i++ ) {
        for( j = 0; j < scn->ysize; j++ ) {
          for( k = 0; k < scn->zsize; k++ ) {
	    p = res->tbz[ res->zsize - i - 1 ] + res->tby[ res->ysize - j - 1 ] + res->xsize - k - 1; // res voxel
	    q = scn->tbz[ k ] + scn->tby[ j ] + i; // scn voxel
	    res->data[ p ] = scn->data[ q ];
	  }
	}
      }
      break;
    }
    break;
  }
  return res;
}

int main( int argc, char **argv ) {
  Scene *in, *out;
  int xdir, ydir;
  int xclass, yclass;

  if( argc != 5 ) {
    printf( "Usage must be: %s <input_file> <output_file> <x_dir> <y_dir>\n<\?_dir>: 0 - l->r sagittal; 1 - r->l sagittal; 2 - u->d axial; 3 - d->u axial; 4 - f->b coronal; 5 - b->f coronal;\n", argv[ 0 ] );
    return 0;
  }

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  xdir = atoi( argv[ 3 ] );
  ydir = atoi( argv[ 4 ] );

  xclass = xdir / 2;
  yclass = ydir / 2;

  if( xclass == yclass ) {
    printf( "Complicting dirs!\n" );
    return 0;
  }
  
  in  = ReadVolume( argv[ 1 ] );
  
  out = RotateAxis( in, xdir, ydir );

  WriteVolume( out, argv[ 2 ] );

  DestroyScene( &in );
  DestroyScene( &out );

  /* ---------------------------------------------------------- */

  info = mallinfo( );
  MemDinFinal = info.uordblks;
  if(MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  /* ---------------------------------------------------------- */
  
  return 0;
}
