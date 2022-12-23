__global__ void diagonal_phase(int i, int B, int N, float *A);
__global__ void row_phase(int i, int j, int B, int N, float *A);
__global__ void col_phase(int i, int j, int B, int N, float *A);
__global__ void right_down_phase(int i, int j, int k, int B, int N, float *A);
__global__ void blocked_lu(int B, int N, float *A, float *L);