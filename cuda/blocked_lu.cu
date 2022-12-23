#include <cuda.h>
#include <cuda_runtime.h>
#include "blocked_lu.h"

__global__ void diagonal_phase(int i, int B, int n, float *A)
{
    for (int ii = i * B; ii < (i * B) + B - 1; ++ii)
  {
    for (int jj = ii + 1; jj < (i * B) + B; ++jj)
    {
      A[(jj * n) + ii] = A[(jj * n) + ii] / A[(ii * n) + ii];

      for (int kk = ii + 1; kk < (i * B) + B; ++kk)
      {
        A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
      }
    }
  }
}

__global__ void row_phase(int i, int j, int B, int n, float *A)
{
  if (blockIdx.x <= i)
    return;

  for (int ii = i * B; ii < (i * B) + (B - 1); ++ii)
  {
    for (int jj = ii + 1; jj < B; ++jj)
    {
      for (int kk = j * B; kk < (j * B) + B; ++kk)
      {
        A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
      }
    }
  }

}

__global__ void col_phase(int i, int j, int B, int n, float *A)
{
  if (blockIdx.y <= i)
    return;

  for (int ii = i * B; ii < (i * B) + B; ++ii)
  {
    for (int jj = j * B; jj < (j * B) + B; ++jj)
    {
      A[(jj * n) + ii] = A[(jj * n) + ii] / A[(ii * n) + ii];

      for (int kk = ii + 1; kk < (i * B) + B; ++kk)
      {
        A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
      }
    }
  }
}

__global__ void right_down_phase(int i, int j, int k, int B, int n, float *A)
{
  if (blockIdx.x <= i || blockIdx.y <= i)
    return;

    for (int ii = i * B; ii < (i * B) + B; ++ii)
  {
    for (int jj = j * B; jj < (j * B) + B; ++jj)
    {
      for (int kk = k * B; kk < (k * B) + B; ++kk)
      {
        A[(jj * n) + kk] = A[(jj * n) + kk] - (A[(jj * n) + ii] * A[(ii * n) + kk]);
      }
    }
  }
}
