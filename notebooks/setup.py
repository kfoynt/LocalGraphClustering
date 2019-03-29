from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules=[ Extension("affinity_kimon",
              ["affinity_kimon.pyx"],
              libraries=["m"],
              include_dirs = [np.get_include()],
              extra_compile_args = ["-O3","-ffast-math"])]

setup(
  name = "affinity_kimon",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

