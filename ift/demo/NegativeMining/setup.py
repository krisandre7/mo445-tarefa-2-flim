from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
import os

# It does not work via pip or sudo
NEWIFT_DIR = os.environ["NEWIFT_DIR"]
CYIFT_DIR  = os.path.join(NEWIFT_DIR, "cyift")

os.environ['CC']  = 'colorgcc'
os.environ['CXX'] = 'colorgcc'

include_dirs       = [os.path.join(CYIFT_DIR, "cyift/"), os.path.join(CYIFT_DIR, "libift/include"),
                      os.path.join(CYIFT_DIR, "libsvm"), "/usr/local/atlas/include", 
                      "/usr/local/atlas/include/atlas", ".", "./neg_mining", "./neg_mining/**", np.get_include()]
libraries          = ["ift", "svm", "lapack", "blas", "cblas", "atlas", "m"]
library_dirs       = [os.path.join(CYIFT_DIR, "libift/lib"), os.path.join(CYIFT_DIR, "libsvm"), 
                     "/usr/local/atlas/lib"]
extra_compile_args = ["-O3", "-fopenmp", "-pthread", "-std=gnu11", "-pedantic"]
extra_link_args    = ["-lgomp", "-fopenmp", "-lstdc++"] # -lstdc++ is required to the libSVM

modules = [("NM2", "./neg_mining/NM2.pyx"), ("classification", "./neg_mining/classification.pyx")]

extensions = []
for mod in modules:
    ext = Extension(mod[0], [mod[1]],
                    include_dirs       = include_dirs,
                    libraries          = libraries,
                    library_dirs       = library_dirs,
                    extra_compile_args = extra_compile_args,
                    extra_link_args    = extra_link_args)
    extensions.append(ext)

setup(
    name         = "neg_mining",
    description  = "Negative Mining package",
    url          = "http://test.com",
    author       = "Samuka Martins",
    author_email = "sbm.martins@gmail.com",
    license      = "UNICAMP",
    packages     = ["neg_mining"],
    cmdclass     = {"build_ext": build_ext },
    ext_modules  = cythonize(extensions, include_path = include_dirs)
    # If you have include files in non-standard places you can pass an 
    # include_path parameter to cythonize
)