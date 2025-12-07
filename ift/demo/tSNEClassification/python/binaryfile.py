import struct
import numpy
import array

def read_float_array(f, n):
    a = array.array('f')
    a.fromfile(f, n)
    return numpy.fromiter(a, 'float32')

def read_float_array2(f, n):
    fmt = "<%df" % n
    return numpy.array(struct.unpack(fmt, f.read(4*n)))

def write_float_array2(f, v):
    fmt='f'*len(v)
    bin=struct.pack(fmt,*v)
    f.write(bin)

def read_float(f):
    return struct.unpack('f', f.read(4))[0]

def read_int(f):
    return struct.unpack('i', f.read(4))[0]

def write_int(f, v):
    f.write(struct.pack('i', v))

def write_float(f, v):
    f.write(struct.pack('f', v))

def read_uchar(f):
	return struct.unpack('B', f.read(1))[0]

def write_uchar(f, v):
	return f.write(struct.pack('B', v))