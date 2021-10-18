from distutils.core import setup, Extension
import os
import sys

rootDir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
#Static
static_libraries = ["gaasCWrapper"]
static_lib_dir = rootDir
#Dynamic
libraries = []
library_dirs = ['/system/lib', '/system/lib64',rootDir]



if sys.platform == 'win32':
    libraries.extend(static_libraries)
    library_dirs.append(static_lib_dir)
    extra_objects = []
else: # POSIX
    extra_objects = ['{}/{}.a'.format(static_lib_dir, l) for l in static_libraries]

setup(name = 'gaasAPI', version = '1.0',
      ext_modules = [Extension('gaasAPI', 
      		     sources = ['gaasPy.c'], 
      		     include_dirs = [rootDir+"/include"],
      		     extra_link_args=["-L"+rootDir,"-lgaasCWrapper","-lcudart","-lcuda"],
      		     extra_objects=[rootDir+"/libgaasCWrapper.so"],
      		     library_dirs=[rootDir]
      		     )]
      		     )

"""
setup(name = 'gaasAPI', version = '1.0',
      ext_modules = [Extension('gaasAPI', 
      		     sources = ['gaasPy.c'], 
      		     include_dirs = [rootDir+"/include"],
      		     extra_compile_args = ["-static"],
      		     libraries=libraries,
                     library_dirs=library_dirs,
                     extra_objects=extra_objects)
      		     ]
      		     )
      		     
 """

