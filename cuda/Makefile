NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm
EXES     := blocked_cuda lu_cuda

alls: $(EXES)

clean:
	rm -f $(EXES)

lu_cuda_b: lu_cuda_b.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

lu_cuda: lu_cuda.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?