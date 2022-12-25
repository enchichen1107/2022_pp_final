#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <ctime>
#include <chrono>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREAD 32

using namespace std;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

// this method tries to utilize share memory better
// share memory first half for the head row, others for multiplier 
extern __shared__ float shm[];
__global__ void compute_LU(int i, int B, int B_num, int n, float *A)
{
  int x = threadIdx.x;
  int y = threadIdx.y;
  int jj = (B * (blockIdx.x % B_num)) + x;
  int ii = (B * (blockIdx.y % B_num)) + y;

  // return useless blocks
  if ((ii <= (i - 1))||(jj < (i - 1)))
  {
    return;
  }

  // load head row and multiplier in shm
  if (y == (B - 1))
  {
    shm[0 * B + x] = A[(i - 1) * n + jj];
    __syncthreads();
  }

  if (x == (B - 1))
  {
    shm[1 * B + y] = A[ii * n + (i - 1)] / shm[0 * B + (i - 1)];
    A[ii * n + (i - 1)] = shm[1 * B + y];
    __syncthreads();
  }
  
  // update row
  if (jj > (i - 1))
  {
    A[ii * n + jj] = A[ii * n + jj] - (shm[0 * B + x] * shm[1 * B + y]); 
    __syncthreads();
  }
  
}

void print_matrix(float *A, int N, int n)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("%.5f ", A[i * n + j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    fprintf(stderr, "must provide exactly 3 arguments N B output_filename\n");
    return 1;
  }
  typedef std::chrono::milliseconds ms;
  auto total_starttime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();

  // parsing argument
  int N = atoi(argv[1]);
  int B = atoi(argv[2]);
  if (B >= MAX_THREAD)
  {
    B = MAX_THREAD;
  }
  char *out_filename = argv[3];

  // generate matrix
  // srand((unsigned)time(NULL));
  int n = (B * (N / B)) + ((N % B == 0)? 0 : B);
  printf("n %d\n",n);
  printf("N %d\n",N);
  int B_num = n / B;
  printf("B %d\n",B);
  printf("B_num %d\n",B_num);

  float *A = (float *)malloc(n * n * sizeof(float));
  float *L = (float *)malloc(N * N * sizeof(float));

  if ((A == NULL)||(L == NULL))
  {
    printf("malloc failed\n");
    exit(1);
  }

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i * n + j] = 1 + (rand() % 10000);
      L[i * N + j] = 0;
    }
    // ensure diagonally dominant
    A[i * n + i] = A[i * n + i] + 10000 * N;
  }
  printf("alloc mem success\n");

  // do the padding
  if (N >= MAX_THREAD)
  {
    for (int i = N; i < n; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        A[i * n + j] = 1000 + i + j;
      }
      A[i * n + i] = A[i * n + i] + 10000 * n;
    }
    for (int j = N; j < n; ++j)
    {
      for (int i = 0; i < n - N; ++i)
      {
        A[i * n + j] = 1000 + i + j;
      }
    }
  }
  printf("padding success\n");

  // print matrix before lu factorization
  if (N < 11)
  {
    printf("the matrix before lu factorization is\n");
    print_matrix(A, N, n);
  }

  // allocate and copy memory to device
  float *device_A;
  cudaMalloc(&device_A, n * n * sizeof(float));
  cudaMemcpy(device_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

  // LU factorization
  dim3 grid(B_num, B_num);
  dim3 block_thread(B, B);
  for (int i = 1; i < N; ++i)
  {
    compute_LU<<<grid, block_thread, 2 * B *sizeof(float)>>>(i, B, B_num, n, device_A); 
  }

  // copy result back to host
  cudaMemcpy(A, device_A, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(device_A);

  // extract L and U
  for (int i = 1; i < N; ++i)
  {
    for (int j = i - 1; j >= 0; --j)
    {
      L[i * N + j] = A[i * n + j];
      A[i * n + j] = 0;
    }
  }

  // assign 1 to diagonal of L
  for (int i = 0; i < N; ++i)
  {
    L[i * N + i] = 1;
  }

  // print outcome
  if (N < 11)
  {
    printf("the lu factorization outcome is\n");
    printf("U is\n");
    print_matrix(A, N, n);
    printf("L is\n");
    print_matrix(L, N, N);
  }

  // write result to output file
  ofstream out_file(out_filename);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      out_file.write((char *)&A[i * n + j], sizeof(float));
    }
  }
  for (int i = 0; i < N * N; ++i)
  {
    out_file.write((char *)&L[i], sizeof(float));
  }
  out_file.close();
  free(A);
  free(L);

  // calculate total spent time
  auto total_endtime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();
  printf("total time spent for blocked lu %ld ms\n", (total_endtime - total_starttime));
}