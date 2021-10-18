IDIR = ./include
TEST_TARGET = kernel
LIB_TARGET = gaasCWrapper.so
SRCDIR = ./src

OBJS =  \
 Gaas.o \
 file_io.o \
 Stopwatch.o \
 gaasCWrapper.o \

REBUILDABLES = $(OBJS) $(TEST_TARGET) $(INIT_TARGET)

CFLAGS =-I$(IDIR) -O3 -fPIC -lstdc++ 
NVCCFLAGS = -I$(IDIR) -gencode arch=compute_75,code=sm_75 -lcudart -Xcompiler=-fPIC

NVCC = nvcc #cuda compiler call
CC = g++ #cpp compiler call

clean : 
	rm -f $(OBJS) $(TEST_TARGET).o
	echo Clean done

all : $(TEST_TARGET) $(LIB_TARGET)
	make clean
	echo All done

$(TEST_TARGET) : $(OBJS) $(TEST_TARGET).o
	$(NVCC) -g -o $@ $^ $(NVCCFLAGS)

$(LIB_TARGET) : $(OBJS) 
	g++ -shared -o lib$@ $^ -fPIC

%.o : ${SRCDIR}/%.cpp
	$(CC) -g -o $@ -c $< $(CFLAGS)
	
%.o : ${SRCDIR}/%.c
	$(CC) -g -o $@ -c $< $(CFLAGS)
	
%.o : ${SRCDIR}/%.cu
	$(NVCC) -g -o $@ -c $< $(NVCCFLAGS)
