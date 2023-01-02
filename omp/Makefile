CC = gcc mpicc
CXX = mpicxx
CXXFLAGS = -O3 -pthread
CFLAGS = -O3 -lm -pthread -march=native -fopenmp -msse4 -msse2 -msse3 -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize
NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
LDFLAGS  := -lm
TARGETS = lu_omp

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

lu_omp: lu_omp.cc
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?


