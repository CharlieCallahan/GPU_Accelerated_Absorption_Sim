from distutils.core import setup, Extension
import os
import sys

rootDir = os.path.normpath(os.getcwd() + os.sep + os.pardir)

#Static
static_libraries = ["gaasCWrapper"]
linkArg = "-lgaasCWrapper" #linux
static_lib_dir = rootDir

#Dynamic
libraries = []
library_dirs = ['/system/lib', '/system/lib64',rootDir]

windows = False
if sys.platform == 'win32' or sys.platform == 'win64':
    windows = True
    libraries.extend(static_libraries)
    static_libraries = ["GAAS"]
    linkArg = "-lGAAS" #windows
    library_dirs.append(static_lib_dir)
    extra_objects = []
    
else: # POSIX
    extra_objects = ['{}/{}.a'.format(static_lib_dir, l) for l in static_libraries]

if windows:
    cudaPath = os.environ['CUDA_PATH']
    cudaLibPath = cudaPath + "//lib//x64//" #location of cuda static libraries
    cudaDllPath = cudaPath + "//bin//" #location of cuda dlls
    path = os.environ['Path']
    cwd = os.getcwd()
    
    if not cudaDllPath in path:
        print("adding cuda DLL location to PATH environment variable")
        path = path + ";"+cudaDllPath
        
    print("cuda path: ",cudaPath)
    
    if (sys.argv[1] == 'install'): #copy GAAS.dll to python site packages location
        pythonPaths = sys.path
        for path in pythonPaths:
            if(path[-13:] == "site-packages"):
                print("copying GAAS.dll to ",path)
                os.system("copy /Y " + "\"" + cwd+"\\GAAS.dll\" " + "\""+path +"\"")
        
    setup(name = 'gaasAPI', version = '1.0',
          ext_modules = [Extension('gaasAPI', 
                     sources = ['gaasPy.c'], 
                     include_dirs = [],#[rootDir+"/include"],
                     extra_compile_args= ["/I"+rootDir+"/include"],
                     extra_link_args=[cudaLibPath+"cudart.lib",cudaLibPath+"cuda.lib", cwd+"//GAAS.lib", "/DLL"], #"/LIBPATH:"+rootDir,"GAAS.lib" "/NODEFAULTLIB:MSVCRT"
                     extra_objects=[cwd+"//GAAS.lib"], #rootDir+"\GAAS.lib"
                     library_dirs=[cwd]
                     )]
                     )
                     
else:    
    setup(name = 'gaasAPI', version = '1.0',
      ext_modules = [Extension('gaasAPI', 
      		     sources = ['gaasPy.c'], 
      		     include_dirs = [rootDir+"/include"],
      		     extra_link_args=["-L"+rootDir,"-lgaasCWrapper","-lcudart","-lcuda"],
      		     extra_objects=[rootDir+"/libgaasCWrapper.so"],
      		     library_dirs=[rootDir]
      		     )]
      		     )




