
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
//#include <windows.h>
#else
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int GetNumberOfProcessors();


/*
#include<stdio.h>
#include<errno.h>
#include<string.h>
#include<stdlib.h>
#ifdef _WIN32
#include<windows.h>
#endif

// Returns the number of processors of the computer.
int NumberOfProcessors();
*/
