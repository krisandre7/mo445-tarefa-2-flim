cdef extern from "../libift/include/iftCommon.h":
    ctypedef unsigned char uchar
    ctypedef unsigned short ushort
    ctypedef unsigned int uint
    ctypedef unsigned long long ullong
    
    cdef float *iftAllocFloatArray(int n)
    
cdef char ** to_cstring_array(list_str)
cdef int_array_to_python_list(int *array, int n)
cdef int *python_list_to_int_array(pylist, int *n)
cdef float_array_to_python_list(float *array, int n)