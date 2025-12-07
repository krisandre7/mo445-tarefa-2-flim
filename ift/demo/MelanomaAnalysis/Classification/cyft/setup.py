import os
from os.path import join as pjoin
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

iftdir = os.environ['NEWIFT_DIR']

libraries = ["ift", "svm", "lapack", "blas", "m"]

include_dirs = [pjoin(iftdir, "include"), pjoin(iftdir, "libsvm/include")]
library_dirs = [pjoin(iftdir, "lib"), pjoin(iftdir, "libsvm")]

extra_compile_args = ["-O3", "-fopenmp", "-pthread", "-std=gnu11", "-pedantic"]
extra_link_args = ["-lgomp", "-fopenmp", "-lstdc++"] # -lstdc++ is required to the libSVM

extensions = [
            Extension("cyft", ["cyft.pyx"],
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    extra_compile_args = extra_compile_args,
                    extra_link_args = extra_link_args)
            ]


setup(
    name="cyft",
    cmdclass = { 'build_ext' : build_ext },
    ext_modules= extensions
)
