NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm
EXES     := blocked_cuda

alls: $(EXES)

clean:
	rm -f $(EXES)

blocked_cuda: blocked_lu.cu blocked_cuda.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?