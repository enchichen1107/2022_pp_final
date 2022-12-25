#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <ctime>
#include <chrono>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "blocked_lu.h"

using namespace std;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

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
    fprintf(stderr, "must provide exactly 3 arguments N block_size output_filename\n");
    return 1;
  }
  typedef std::chrono::milliseconds ms;
  auto total_starttime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();

  // parsing argument
  int N = atoi(argv[1]);
  int B = atoi(argv[2]);
  char *out_filename = argv[3];

  // generate matrix
  // srand((unsigned)time(NULL));
  int n = N;
  if ((N % B) != 0)
    n = B * (N / B) + B;
  int blocks = n / B;

  float *A = (float *)malloc(n * n * sizeof(float));
  float *L = (float *)malloc(N * N * sizeof(float));
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

  // do the padding
  if ((N % B) != 0)
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

  // initialize grid dim and block dim
  int B1 = (B / 32 == 0) ? 1 : (B % 32 == 0) ? (B / 32)
                                             : (B / 32 + 1);
  dim3 grid_dim_phase1(B1, B1);
  dim3 grid_dim_phase2_1(blocks, B1);
  dim3 grid_dim_phase2_2(B1, blocks);
  dim3 grid_dim_phase3(blocks, blocks);
  dim3 block_dim(B, B);

  // blocked lu factorization
  for (int i = 0; i < blocks; ++i)
  {

    diagonal_phase<<<grid_dim_phase1, block_dim>>>(i, B, n, device_A);

    for (int j = i + 1; j < blocks; ++j)
    {
      row_phase<<<grid_dim_phase2_1, block_dim>>>(i, j, B, n, device_A);
    }
    
    for (int j = i + 1; j < blocks; ++j)
    {
      col_phase<<<grid_dim_phase2_2, block_dim>>>(i, j, B, n, device_A);

      for (int k = i + 1; k < blocks; ++k)
      {
        right_down_phase<<<grid_dim_phase3, block_dim>>>(i, j, k, B, n, device_A);
      }

    }
    
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