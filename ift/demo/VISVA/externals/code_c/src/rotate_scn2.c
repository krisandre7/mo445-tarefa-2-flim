#include "oldift.h"
#include "compressed.h"

char CompSignal( char s ) {
  if( s == '+' )
    return( '-' );
  else
    return( '+' );
}

void DefineSagitalDirection( char axi[ ], char cor[ ], char sag[ ] ) {
  sag[ 2 ] = '\0';
  if( ( axi[ 1 ] != 'x' ) && ( cor[ 1 ] != 'x' ) ) {
    sag[ 1 ] = 'x';
    if( ( strcmp( axi, "+y" ) == 0 ) || ( strcmp( axi, "-z" ) == 0 ) )
      sag[ 0 ] = cor[ 0 ];
    else {
      if( cor[ 0 ] == '-' )
	sag[ 0 ] = '+';
      else
	sag[ 0 ] = '-';
    }
  }
  else if( ( axi[ 1 ] != 'y' ) && ( cor[ 1 ] != 'y' ) ) {
    sag[ 1 ] = 'y';
    // +z,+x; -x,+z; -z,-x; +x,-z ==> +y; (axi,cor)
    // +z,-x; -x,-z; -z,+x; +x,+z ==> -y; (axi,cor)
    if( ( strcmp( axi, "+z" ) == 0 ) || ( strcmp( axi, "-x" ) == 0 ) )
      sag[ 0 ] = cor[ 0 ];
    else {
      if( cor[ 0 ] == '-' )
	sag[ 0 ] = '+';
      else
	sag[ 0 ] = '-';
    }
  }
  else { // sag = 'z'
    sag[ 1 ] = 'z';
    // +y,-x; -x,-y; -y,+x; +x,+y ==> +z; (axi,cor)
    // +y,+x; -x,+y; -y,-x; +x,-y ==> -z; (axi,cor)
    if( ( strcmp( axi, "+x" ) == 0 ) || ( strcmp( axi, "-y" ) == 0 ) )
      sag[ 0 ] = cor[ 0 ];
    else {
      if( cor[ 0 ] == '-' )
	sag[ 0 ] = '+';
      else
	sag[ 0 ] = '-';
    }
  }
}

// 0 - l->r sagittal; 1 - r->l sagittal; 2 - u->d axial; 3 - d->u axial; 4 - f->b coronal; 5 - b->f coronal
// output: xdir - 0; ydir - 2
Scene *RotateAxis( Scene *scn, char axi_in_dir[ ], char cor_in_dir[ ], char axi_out_dir[ ], char cor_out_dir[ ] ) {
  char sag_in_dir[ 3 ], sag_out_dir[ 3 ];
  Scene *res=NULL;
  char xdir[ 3 ], ydir[ 3 ];
  int i,j,k,p,q;

  DefineSagitalDirection( axi_in_dir, cor_in_dir, sag_in_dir );
  DefineSagitalDirection( axi_out_dir, cor_out_dir, sag_out_dir );

  xdir[ 2 ] = ydir[ 2 ] = '\0';

  if( axi_in_dir[ 1 ] == 'x' ) {
    xdir[ 1 ] = axi_out_dir[ 1 ];
    if( axi_in_dir[ 0 ] == '+' )
      xdir[ 0 ] = axi_out_dir[ 0 ];
    else
      xdir[ 0 ] = CompSignal( axi_out_dir[ 0 ] );
  }
  else if( axi_in_dir[ 1 ] == 'y' ) {
    ydir[ 1 ] = axi_out_dir[ 1 ];
    if( axi_in_dir[ 0 ] == '+' )
      ydir[ 0 ] = axi_out_dir[ 0 ];
    else
      ydir[ 0 ] = CompSignal( axi_out_dir[ 0 ] );
  }

  if( sag_in_dir[ 1 ] == 'x' ) {
    xdir[ 1 ] = sag_out_dir[ 1 ];
    if( sag_in_dir[ 0 ] == '+' )
      xdir[ 0 ] = sag_out_dir[ 0 ];
    else
      xdir[ 0 ] = CompSignal( sag_out_dir[ 0 ] );
  }
  else if( sag_in_dir[ 1 ] == 'y' ) {
    ydir[ 1 ] = sag_out_dir[ 1 ];
    if( sag_in_dir[ 0 ] == '+' )
      ydir[ 0 ] = sag_out_dir[ 0 ];
    else
      ydir[ 0 ] = CompSignal( sag_out_dir[ 0 ] );
  }

  if( cor_in_dir[ 1 ] == 'x' ) {
    xdir[ 1 ] = cor_out_dir[ 1 ];
    if( cor_in_dir[ 0 ] == '+' )
      xdir[ 0 ] = cor_out_dir[ 0 ];
    else
      xdir[ 0 ] = CompSignal( cor_out_dir[ 0 ] );
  }
  else if( cor_in_dir[ 1 ] == 'y' ) {
    ydir[ 1 ] = cor_out_dir[ 1 ];
    if( cor_in_dir[ 0 ] == '+' )
      ydir[ 0 ] = cor_out_dir[ 0 ];
    else
      ydir[ 0 ] = CompSignal( cor_out_dir[ 0 ] );
  }
  
  printf( "xdir:%s, ydir:%s\n", xdir, ydir );

  if( strcmp( xdir, "+x" ) == 0 ) {
    if( strcmp( ydir, "+y" ) == 0  ) {
      printf("+x,-z\n");
      res = CopyScene( scn );
    }
    else if( strcmp( ydir, "-y" ) == 0 ) {
      printf("+x,-y\n");
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
    }
    else if( strcmp( ydir, "+x" ) == 0 ) {
      printf("+x,+z\n");
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
    }
    else {
      printf("+x,-z\n");
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
    }
  }
  
  else if( strcmp( xdir, "-x" ) == 0 ) {
    if( strcmp( ydir, "+y" ) == 0 ) {
      printf("-x,+y\n");
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
    }
    else if( strcmp( ydir, "-y" ) == 0 ) {
      printf("-x,-y\n");
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
    }
    else if( strcmp( ydir, "+z" ) == 0 ) {
      printf("-x,+z\n");
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
    }
    else {
      printf("-x,-z\n");
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
    }
  }

  else if( strcmp( xdir, "+y" ) == 0 ) {
    if( strcmp( ydir, "+x" ) == 0 ) {
      printf("+y,+x\n");
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
    }
    else if( strcmp( ydir, "-x" ) == 0 ) {
      printf("+y,-x\n");
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
    }
    else if( strcmp( ydir, "+z" ) == 0 ) {
      printf("+y,+z\n");
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
    }
    else {
      printf("+y,-z\n");
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
    }
  }

  else if( strcmp( xdir, "-y" ) == 0 ) {
    if( strcmp( ydir, "+x" ) == 0 ) {
      printf("-y,+x\n");
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
    }
    else if( strcmp( ydir, "-x" ) == 0 ) {
      printf("-y,-x\n");
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
    }
    else if( strcmp( ydir, "+z" ) == 0 ) {
      printf("-y,+z\n");
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
    }
    else {
      printf("-y,-z\n");
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
    }
  }

  else if( strcmp( xdir, "+z" ) == 0 ) {
    if( strcmp( ydir, "+x" ) == 0 ) {
      printf("+z,+x\n");
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
    }
    else if( strcmp( ydir, "-x" ) == 0 ) {
      printf("+z,-x\n");
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
    }
    else if( strcmp( ydir, "+y" ) == 0 ) {
      printf("+z,+y\n");
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
    }
    else {
      printf("+z,-y\n");
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
    }
  }

  else {
    if( strcmp( ydir, "+x" ) == 0 ) {
      printf("-z,+x\n");
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
    }
    else if( strcmp( ydir, "-x" ) == 0 ) {
      printf("-z,-x\n");
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
    }
    else if( strcmp( ydir, "+y" ) == 0 ) {
      printf("-z,+y\n");
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
    }
    else {
      printf("-z,-y\n");
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
    }
  }
  
  return res;
}

int main( int argc, char **argv ) {
  Scene *in, *out;
  char cor_in_dir[ 3 ], axi_in_dir[ 3 ];
  char cor_out_dir[ 3 ], axi_out_dir[ 3 ];

  if( argc != 7 ) {
    printf( "Usage must be: %s <input_file> <output_file> <cor_in_dir> <axi_in_dir> <cor_out_dir> <axi_out_dir>\n\t<*_dir> options are: +x, -x, +y, -y, +z, -z.\n", argv[ 0 ] );
    printf( "\tThe directions are the axes of coronal and axial slices of the input and output brains.\n" );
    printf( "\tThe signal must indicate increasing \'+\' or decreasing\'-\' scale from\n\tanterior/superior to posterior/inferior part of the brain.\n" );
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

  strcpy( axi_in_dir, argv[ 3 ] );
  strcpy( cor_in_dir, argv[ 4 ] );
  strcpy( axi_out_dir, argv[ 5 ] );
  strcpy( cor_out_dir, argv[ 6 ] );

  if( ( axi_in_dir[ 1 ] == cor_in_dir[ 1 ] ) || ( axi_out_dir[ 1 ] == cor_out_dir[ 1 ] ) ) {
    printf( "Comflicting directions!\n" );
    return 0;
  }
    
  in  = ReadVolume( argv[ 1 ] );
  
  out = RotateAxis( in, axi_in_dir, cor_in_dir, axi_out_dir, cor_out_dir );

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
