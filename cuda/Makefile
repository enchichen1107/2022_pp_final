NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm
EXES     := lu_cuda lu_cuda_b

alls: $(EXES)

clean:
	rm -f $(EXES)

lu_cuda_b: lu_cuda_b.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

lu_cuda: lu_cuda.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?