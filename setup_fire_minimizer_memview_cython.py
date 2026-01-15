from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="fire_minimize_memview_cy",
        sources=["fire_minimizer_memview_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="fire_minimizer_memview_cy",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}, annotate=True),
)
