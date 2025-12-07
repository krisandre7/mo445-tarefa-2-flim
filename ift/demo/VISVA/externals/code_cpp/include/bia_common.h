/**
 * \file bia_common.h
 * \brief Header file for common definitions and function prototypes.
 */

#ifndef _BIA_COMMON_H_
#define _BIA_COMMON_H_

extern "C" {
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <float.h>

#include <xmmintrin.h>
}

/**
 * \brief Base namespace for common definitions and prototypes.
 */
namespace bia{

  // Error messages
#define MSG1  "Cannot allocate memory space"
#define MSG2  "Cannot open file"
#define MSG3  "Invalid option"
  
  typedef unsigned short ushort;
  typedef unsigned int uint;
  typedef unsigned char uchar;


  // Common definitions
#define WHITE       0
#define GRAY        1
#define BLACK       2
#define NIL        -1


  // Common operations

  /**
   * \def MAX(x,y)
   * \brief A macro that returns the maximum of \a x and \a y.
   */
#ifndef MAX
#define MAX(x,y) (((x) > (y))?(x):(y))
#endif
  /**
   * \def MIN(x,y)
   * \brief A macro that returns the minimum of \a x and \a y.
   */
#ifndef MIN
#define MIN(x,y) (((x) < (y))?(x):(y))
#endif
  
#define ROUND(x) ((x < 0)?(int)(x-0.5):(int)(x+0.5))

#define SIGN(x) ((x >= 0)?1:-1)


  /**
   * \brief Vector of four single floats.
   */
  typedef float  v4sf  __attribute__ ((vector_size(16),aligned(16)));
  
  /**
   * \brief Vector of four single integers.
   */
  typedef int    v4si  __attribute__ ((vector_size(16),aligned(16)));

  /**
   * \brief Vector of eight unsigned 8-bit integers.
   */
  typedef uchar  v8qi  __attribute__ ((vector_size(8),aligned(16)));

  /**
   * \brief Vector of sixteen unsigned 8-bit integers.
   */
  typedef uchar  v16qi __attribute__ ((vector_size(16),aligned(16)));

  /**
   * \brief Vector of eight unsigned short integers.
   */
  typedef ushort v8hi  __attribute__ ((vector_size(16),aligned(16)));
  
  
  typedef union _voxel {
    v4si v;
    int  data[4];
    struct{ int x,y,z; } c;
  } Voxel;


  /**
   * \brief It allocates 1D array of n integers.
   */
  int    *AllocIntArray(int n);

  /**
   * \brief It allocates 1D array of n floats.
   */
  float  *AllocFloatArray(int n);

  /**
   * \brief It allocates 1D array of n characters.
   */
  uchar  *AllocUCharArray(int n);

  /**
   * \brief It allocates 1D array of n ushorts.
   */
  ushort *AllocUShortArray(int n);

  void    FreeIntArray(int **a);     
  void    FreeFloatArray(float **a);
  void    FreeUCharArray(uchar **a);
  void    FreeUShortArray(ushort **a);

  /**
   * \brief It prints error message and exits the program.
   */
  void Error(char *msg,char *func);

  /**
   * \brief It prints warning message and leaves the routine.
   */
  void Warning(char *msg,char *func);

  /**
   * \brief It changes content between a and b.
   */
  inline void SwapInt(int *a, int *b){
    int c;
    c  = *a;  
    *a = *b;  
    *b = c;
  }

  /**
   * \brief It changes content between a and b.
   */
  inline void SwapFloat(float *a, float *b){
    float c;
    c  = *a;
    *a = *b;
    *b = c;
  }

  
} //end bia namespace

#endif

