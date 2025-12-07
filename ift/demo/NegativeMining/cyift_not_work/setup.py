from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import fnmatch
import os
import shutil

libraries          = ["ift", "svm", "lapack", "blas", "cblas", "atlas", "m"]
include_dirs       = ["./libift/include", "./libsvm", "./cyift"]
library_dirs       = ["./libift/lib", "./libsvm", "/usr/local/atlas/lib"]
extra_compile_args = ["-O3", "-fopenmp", "-pthread", "-std=gnu11", "-pedantic"]
extra_link_args    = ["-lgomp", "-fopenmp", "-lstdc++"] # -lstdc++ is required to the libSVM

extensions = [
            Extension("cyift.common", ["./cyift/common.pyx"],
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    extra_compile_args = extra_compile_args,
                    extra_link_args = extra_link_args),
            Extension("cyift.dataset", ["./cyift/dataset.pyx"],
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    extra_compile_args = extra_compile_args,
                    extra_link_args = extra_link_args)
            ]


setup(
    name="cyift",
    description="Cython implmentation of IFT",
    url="http://test.com",
    author="Samuka Martins",
    author_email='sbm.martins@gmail.com',
    license="UNICAMP",
    packages=["cyift"],
    cmdclass = { 'build_ext' : build_ext },
    ext_modules=extensions
)