#include <cuda.h>
#include <cuda_runtime.h>
#include "blocked_lu.h"

__global__ void diagonal_phase(int i, int B, int n, float *A)
{
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int ii = B * (i + blockIdx.y) + y;
  const int jj = B * (i + blockIdx.x) + x;
  // for (int ii = i * B; ii < (i * B) + B - 1; ++ii)
  // {
  //   for (int jj = ii + 1; jj < (i * B) + B; ++jj)
  //   {
  //     A[(jj * n) + ii] = A[(jj * n) + ii] / A[(ii * n) + ii];

  //     for (int kk = ii + 1; kk < (i * B) + B; ++kk)
  //     {
  //       A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
  //     }
  //   }
  // }
  A[(jj * n) + ii] = A[(jj * n) + ii] / A[(ii * n) + ii];
  for (int kk = ii + 1; kk < (i * B) + B; ++kk)
  {
    A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
  }
}

__global__ void row_phase(int i, int j, int B, int n, float *A)
{
  if (blockIdx.x <= i)
    return;

  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int ii = B * (i + blockIdx.y) + y;
  const int jj = B * (i + blockIdx.x) + x;

  for (int kk = j * B; kk < (j * B) + B; ++kk)
  {
    A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
  }
}

__global__ void col_phase(int i, int j, int B, int n, float *A)
{
  if (blockIdx.y <= i)
    return;

  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int ii = B * (i + blockIdx.y) + y;
  const int jj = B * (i + blockIdx.x) + x;
  A[(jj * n) + ii] = A[(jj * n) + ii] / A[(ii * n) + ii];

  for (int kk = ii + 1; kk < (i * B) + B; ++kk)
  {
    A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
  }
}

__global__ void right_down_phase(int i, int j, int k, int B, int n, float *A)
{
  if (blockIdx.x <= i || blockIdx.y <= i)
    return;

  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int ii = B * (i + blockIdx.y) + y;
  const int jj = B * (i + blockIdx.x) + x;

  for (int kk = k * B; kk < (k * B) + B; ++kk)
  {
    A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
  }
}

// void blocked_lu(int B, int N, float *A, float *L)
// {
//   int n = N;
//   if ((N % B) != 0)
//     n = B * (N / B) + B;
//   int blocks = n / B;

//   for (int i = 0; i < blocks; ++i)
//   {

//     diagonal_phase(i, B, n, A);

//     for (int j = i + 1; j < blocks; ++j)
//     {
//       row_phase(i, j, B, n, A);
//     }

//     for (int j = i + 1; j < blocks; ++j)
//     {
//       col_phase(i, j, B, n, A);
//     }

//     for (int j = i + 1; j < blocks; ++j)
//     {
//       for (int k = i + 1; k < blocks; ++k)
//       {
//         right_down_phase(i, j, k, B, n, A);
//       }
//     }
//   }
// }