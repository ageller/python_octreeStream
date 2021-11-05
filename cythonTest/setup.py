from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        "octreeStreamCython1.pyx",
        compiler_directives={'language_level' : "3"}   # or "2" or "3str"
    )
)
