# Copyright (c) 2021 Charlie Callahan

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

IDIR = ./include
LIB_TARGET = gaasCWrapper.so
SRCDIR = ./src

OBJS =  \
 Gaas.o \
 file_io.o \
 Stopwatch.o \
 gaasCWrapper.o \

REBUILDABLES = $(OBJS) $(TEST_TARGET) $(INIT_TARGET)

CFLAGS =-I$(IDIR) -O3 -fPIC -lstdc++ 
#NVCCFLAGS = -I$(IDIR) -gencode arch=compute_75,code=sm_75 -lcudart -Xcompiler=-fPIC
NVCCFLAGS = -I$(IDIR) -I/usr/local/cuda-12.0/include -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -lcudart_static -Xcompiler=-fPIC -std=c++17

NVCC = nvcc #cuda compiler call
CC = g++ #cpp compiler call

clean : 
	rm -f $(OBJS)
	echo Clean done

all : $(LIB_TARGET)
	make clean
	echo All done

$(LIB_TARGET) : $(OBJS) 
	g++ -shared -o lib$@ $^ -fPIC

%.o : ${SRCDIR}/%.cpp
	$(CC) -g -o $@ -c $< $(CFLAGS)
	
%.o : ${SRCDIR}/%.c
	$(CC) -g -o $@ -c $< $(CFLAGS)
	
%.o : ${SRCDIR}/%.cu
	$(NVCC) -g -o $@ -c $< $(NVCCFLAGS)
