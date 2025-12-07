from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcmp
from cpython.string cimport PyString_AsString
import sys
from cyift.common cimport *

def print_error(msg, function):
    sys.exit("\n*** PYTHON: ERROR in \"{0}\":\n{1}\n\n".format(function, msg))
    
# Convert a Python List of Strings in an Array of Strings in C
cdef char ** to_cstring_array(list_str):
    cdef char **ret = <char **>     malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyString_AsString(list_str[i])
    return ret

cdef int_array_to_python_list(int *array, int n):
    pylist = []
    
    for i in xrange(n):
        pylist.append(array[i])
        
    return pylist


cdef int *python_list_to_int_array(pylist, int *n):
    cdef int *array = <int *> calloc(len(pylist), sizeof(int))
    
    for idx, value in enumerate(pylist):
        array[idx] = value
        
    return array 



cdef float_array_to_python_list(float *array, int n):
    pylist = []
    
    for i in xrange(n):
        pylist.append(array[i])
        
    return pylist