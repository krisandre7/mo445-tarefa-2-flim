#include "image.h"
#include "compressed.h"
#include "bz2_lib.h"

void WriteCompressedScene(Scene *scn, char *filename) {
  
  FILE   *f;
  //FILE   *in;
  ift_BZFILE *b;
  //int     nBuf = 1024;
  //char    buf[1024];
  int    Imax, n, i;
  int    bzerror;
  char   *data;
  uchar  *data8;
  ushort *data16;

  f = fopen( filename, "wb" );
  
  //if ( ( !f ) || ( !in ) ) {
  if ( !f ) {
    printf("Error on opening the compressed file %s\n", filename);
    return;
  }
  
  b = ift_BZ2_bzWriteOpen( &bzerror, f, 9, 0, 30 );
  if (bzerror != ift_BZ_OK) {
    ift_BZ2_bzWriteClose ( &bzerror, b, 0, 0, 0 );
    fclose(f);
    printf("Error on opening the file %s\n", filename);
  }
  
  //while ( (bzerror == ift_BZ_OK ) && (nBuf > 0) ) {
  if (bzerror == ift_BZ_OK) {
    /* get data to write into buf, and set nBuf appropriately */
    data = (char*) calloc(200, sizeof(int));
    n = scn->xsize*scn->ysize*scn->zsize;
    sprintf(data,"SCN\n");
    sprintf(data,"%s%d %d %d\n",data,scn->xsize,scn->ysize,scn->zsize);
    sprintf(data,"%s%f %f %f\n",data,scn->dx,scn->dy,scn->dz);
    Imax = MaximumValue3(scn);
    if (Imax < 256) {
      //printf("8bits\n");
      sprintf(data,"%s%d\n",data,8);
      ift_BZ2_bzWrite ( &bzerror, b, data, strlen( data ) );
      data8 = AllocUCharArray(n);
      for (i=0; i < n; i++) 
	data8[i] = (uchar) scn->data[i];
      ift_BZ2_bzWrite ( &bzerror, b, data8, n );
      free(data8);
    } else if (Imax < 65536) {
      //printf("16bits\n");
      sprintf(data,"%s%d\n",data,16);
      ift_BZ2_bzWrite ( &bzerror, b, data, strlen( data ) );
      data16 = AllocUShortArray(n);
      for (i=0; i < n; i++)
	data16[i] = (ushort) scn->data[i];
      ift_BZ2_bzWrite ( &bzerror, b, data16, 2 * n );
      free(data16);
    } else {
      //printf("32bits\n");
      sprintf(data,"%s%d\n",data,32);
      ift_BZ2_bzWrite ( &bzerror, b, data, strlen( data ) );
      ift_BZ2_bzWrite ( &bzerror, b, scn->data, 4 * n );
    }
    free(data);
    /*
      nBuf = fread(buf, sizeof(char), 1024, in);
      if (nBuf > 0)
      ift_BZ2_bzWrite ( &bzerror, b, buf, nBuf );
      
      if (bzerror == ift_BZ_IO_ERROR) { 
      break;
      }
    */
  }
  
  ift_BZ2_bzWriteClose ( &bzerror, b, 0, 0, 0 );
  fclose(f);

  if (bzerror == ift_BZ_IO_ERROR) {
    printf("Error on writing to %s\n", filename);
  }
  
}


Scene *ReadCompressedScene(char *filename) {
  FILE   *f;
  ift_BZFILE *b;
  int     nBuf;
  char   *buf;
  int     bzerror;
  Scene  *scn = NULL;
  int     xsize, ysize, zsize, pos;
  int     nbits, hdr_size, i, p;
  float   dx, dy, dz;
  char    s[ 2000 ];
  
  f = fopen ( filename, "r" );
  buf = ( char* ) calloc( 1025, sizeof( char ) );
  if ( !f ) {
    printf("Erro ao abrir arquivos!\n");
    return NULL;
  }
  b = ift_BZ2_bzReadOpen ( &bzerror, f, 0, 0, NULL, 0 );
  if ( bzerror != ift_BZ_OK ) {
    ift_BZ2_bzReadClose ( &bzerror, b );
    fclose(f);
    printf("Erro ao abrir cena compactada!\n");
    return NULL;
  }

  nBuf = ift_BZ2_bzRead ( &bzerror, b, buf, 1024 );
  hdr_size = 0;
  if ( ( ( bzerror == ift_BZ_OK ) || ( bzerror == ift_BZ_STREAM_END ) ) && (nBuf > 0) ) {
    sscanf( buf, "%[^\n]\n%d %d %d\n%f %f %f\n%d\n", s, &xsize, &ysize, &zsize, &dx, &dy, &dz, &nbits );
    //printf( "s:%s, xsize:%d, ysize:%d, zsize:%d, dx:%f, dy:%f, dz:%f, bits:%d\n", s, xsize, ysize, zsize, dx, dy, dz, nbits );
    if ( strcmp( s, "SCN" ) != 0 ) {
      printf( "Format must by a compressed scene.\nFound %s\n", s );
      return 0;
    }
    scn = CreateScene( xsize, ysize, zsize );
    scn->dx = dx;
    scn->dy = dy;
    scn->dz = dz;

    hdr_size = strlen( s ) + strlen( "\n" );
    for( i = 0; i < 3; i++ ) {
      sscanf( &buf[ hdr_size ], "%[^\n]\n", s );
      hdr_size += strlen( s ) + strlen( "\n" );
    }
  }
  else {
    printf("Erro ao ler cena compactada!\n");
    return 0;
  }
  
  ift_BZ2_bzReadClose ( &bzerror, b );
  rewind( f );
  b = ift_BZ2_bzReadOpen ( &bzerror, f, 0, 0, NULL, 0 );
  if ( bzerror != ift_BZ_OK ) {
    ift_BZ2_bzReadClose ( &bzerror, b );
    fclose(f);
    printf("Erro ao abrir cena compactada!\n");
    return NULL;
  }
  
  nBuf = ift_BZ2_bzRead ( &bzerror, b, buf, hdr_size );
  p = 0;
  do {
    nBuf = ift_BZ2_bzRead ( &bzerror, b, buf, 1024 );
    if ( ( ( bzerror == ift_BZ_OK ) || ( bzerror == ift_BZ_STREAM_END ) ) && ( nBuf > 0 ) ) {
      pos = 0;
      while( pos < nBuf - ( nbits / 8 ) + 1 ) {
	if( nbits == 8 ) {
	  scn->data[ p ] = ( int ) ( ( uchar ) buf[ pos ] );
	  pos++;
	}
	else if( nbits == 16 ) {
	  scn->data[ p ] = ( int ) ( ( ( uchar ) buf[ pos + 1 ] ) * 256 + ( ( uchar ) buf[ pos ] ) );
	  if ( scn->data[ p ] < 0 ) printf("dado negativo.\n");
	  pos += 2;
	}
	else {
	  scn->data[ p ] = ( int ) ( ( uchar ) buf[ pos + 3 ] );
	  for( i = 2; i >= 0; i-- ) {
	    scn->data[ p ] = scn->data[ p ] * 256 + ( ( uchar ) buf[ pos + i ] );
	  }
	  pos +=4;
	}
	p++;
      }
    }
  } while( bzerror == ift_BZ_OK );
  //printf( "scn->n:%d, p:%d\n", scn->n, p );
 
  MaximumValue3( scn );
  //printf( "max=%d\n", scn->maxval );
  ift_BZ2_bzReadClose ( &bzerror, b );
  fclose(f);
  free(buf);

  //printf( "fim!\n" );
  return( scn );
}

// Get the file name by the base name.
int GetFilebyBaseName(char *basename, char *filename) {
  FILE *ftest;
  sprintf( filename, "%s.scn.bz2", basename );
  ftest = fopen( filename, "r" );
  if( ftest != NULL ) {
    fclose( ftest );
    return( COMPRESSED_TYPE );
  }
  sprintf( filename, "%s.scn", basename );
  ftest = fopen( filename, "r" );
  if( ftest != NULL ) {
    fclose( ftest );
    return( NORMAL_TYPE );
  }
  return( -1 );
}


// Read accordingly to the file extension.
Scene *ReadVolume(char *filename){
  return ReadScene(filename);
}


// Write accordingly to the file extension.
void WriteVolume(Scene *scn, char *filename){
  WriteScene(scn,filename);
}






