CC=g++
CFLAGS=-c -std=c++17 -O2 -Wall -DUSEFLOAT32
LDFLAGS=

CUDAC=nvcc
CUDACFLAGS=-dc -arch=sm_75 -DUSEFLOAT32
CUDALFLAGS=-lcudart -lm -lstdc++ -arch=sm_75 -L/usr/local/cuda-11.6/lib64

INCLUDES=-Iincludes

OBJECTS=main.o
CUDAOBJECTS=affine.o params.o funcs.o calcs.o kernel_assembleElems.o

all: main

main: $(OBJECTS) $(CUDAOBJECTS)
	$(CUDAC) -o $@ $^ $(LDFLAGS) $(CUDALFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

%.o: %.cu
	$(CUDAC) $(CUDACFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f *.o main
