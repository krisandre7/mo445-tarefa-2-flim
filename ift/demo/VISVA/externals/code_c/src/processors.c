
#include "processors.h"

#ifdef _WIN32
#include <windows.h>
#endif

int GetNumberOfProcessors(){
  long nprocs = -1;
  long nprocs_max = -1;
#ifdef _WIN32
	#ifndef _SC_NPROCESSORS_ONLN
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		#define sysconf(a) info.dwNumberOfProcessors
		#define _SC_NPROCESSORS_ONLN
	#endif
#endif
#ifdef _SC_NPROCESSORS_ONLN
  nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  if (nprocs < 1)
  {
    fprintf(stderr, "Could not determine number of CPUs online:\n%s\n", 
strerror (errno));
    exit (EXIT_FAILURE);
  }
  nprocs_max = sysconf(_SC_NPROCESSORS_CONF);
  if (nprocs_max < 1)
  {
    fprintf(stderr, "Could not determine number of CPUs configured:\n%s\n", 
strerror (errno));
    exit (EXIT_FAILURE);
  }
  //printf ("%ld of %ld processors online\n",nprocs, nprocs_max);
#else
  fprintf(stderr, "Could not determine number of CPUs");
  exit (EXIT_FAILURE);
#endif
  return nprocs;
}


/*
int NumberOfProcessors( ) {
  int nprocs = -1;
  #ifdef _WIN32
    #ifndef _SC_NPROCESSORS_ONLN
      SYSTEM_INFO info;
      GetSystemInfo(&info);
      #define sysconf(a) info.dwNumberOfProcessors
      #define _SC_NPROCESSORS_ONLN
    #endif
  #endif
  #ifdef _SC_NPROCESSORS_ONLN
    nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if( nprocs < 1 ) 
      // Could not determine the number of processors.
      return( 1 );
    return( nprocs );
  #else
    // Could not determine the number of processors.
    return( 1 );
  #endif
}
*/


