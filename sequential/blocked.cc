#include "blocked.h"

void diagonal_phase(int i, int B, int n, float *A)
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

void row_phase(int i, int j, int B, int n, float *A)
{
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

void col_phase(int i, int j, int B, int n, float *A)
{
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

void right_down_phase(int i, int j, int k, int B, int n, float *A)
{
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

void blocked_lu(int B, int N, float *A, float *L)
{
  int n = N;
  if ((N % B) != 0)
    n = B * (N / B) + B;
  int blocks = n / B;

  for (int i = 0; i < blocks; ++i)
  {

    diagonal_phase(i, B, n, A);

    for (int j = i + 1; j < blocks; ++j)
    {
      row_phase(i, j, B, n, A);
    }

    for (int j = i + 1; j < blocks; ++j)
    {
      col_phase(i, j, B, n, A);

      for (int k = i + 1; k < blocks; ++k)
      {
        right_down_phase(i, j, k, B, n, A);
      }
    }
  }

  // extract L and U
  for (int i = 1; i < N; ++i)
  {
    for (int j = i - 1; j >= 0; --j)
    {
      L[i * N + j] = A[i * n + j];
      A[i * n + j] = 0;
    }
  }
}