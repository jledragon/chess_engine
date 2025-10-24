# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:03:29 2022

@author: jledragon
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='chess_cpp',
    ext_modules=[
        CUDAExtension(
            name='chess_cpp',
            sources=['chess_cpp_utils.cpp', 'chess_cu_utils.cu'],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
