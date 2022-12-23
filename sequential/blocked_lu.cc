#include "blocked_lu.h"

void diagonal_phase(int i, int B, int N, float *A)
{
  for (int ii = i * B; ii < (i * B) + B - 1; ii++)
  {
    for (int jj = ii + 1; jj < (i * B) + B; jj++)
    {
      A[(jj * N) + ii] /= A[(ii * N) + ii];

      for (int kk = ii + 1; kk < (i * B) + B; kk++)
      {
        A[(jj * N) + kk] = A[(jj * N) + kk] - (A[(jj * N) + ii] * A[(ii * N) + kk]);
      }
    }
  }
}

void row_phase(int i, int j, int B, int N, float *A)
{
  for (int ii = i * B; ii < (i * B) + (B - 1); ii++)
  {
    for (int jj = ii + 1; jj < B; jj++)
    {
      for (int kk = j * B; kk < (j * B) + B; kk++)
      {
        A[(jj * N) + kk] = A[(jj * N) + kk] - (A[(jj * N) + ii] * A[(ii * N) + kk]);
      }
    }
  }
}

void col_phase(int i, int j, int B, int N, float *A)
{
  for (int ii = i * B; ii < (i * B) + B; ii++)
  {
    for (int jj = j * B; jj < (j * B) + B; jj++)
    {
      A[(jj * N) + ii] /= A[(ii * N) + ii];

      for (int kk = ii + 1; kk < (i * B) + B; kk++)
      {
        A[(jj * N) + kk] = A[(jj * N) + kk] - (A[(jj * N) + ii] * A[(ii * N) + kk]);
      }
    }
  }
}

void right_down_phase(int i, int j, int k, int B, int N, float *A)
{
  for (int ii = i * B; ii < (i * B) + B; ii++)
  {
    for (int jj = j * B; jj < (j * B) + B; jj++)
    {
      for (int kk = k * B; kk < (k * B) + B; kk++)
      {
        A[(jj * N) + kk] = A[(jj * N) + kk] - (A[(jj * N) + ii] * A[(ii * N) + kk]);
      }
    }
  }
}

void blocked_lu(int B, int N, float *A, float *L)
{
  int num_blocks = N / B + (N % B) ? 1 : 0;
  for (int i = 0; i < num_blocks; i++)
  {

    diagonal_phase(i, B, N, A);

    for (int j = i + 1; j < num_blocks; j++)
    {
      row_phase(i, j, B, N, A);
    }

    for (int j = i + 1; j < num_blocks; j++)
    {
      col_phase(i, j, B, N, A);

      for (int k = i + 1; k < num_blocks; k++)
      {
        right_down_phase(i, j, k, B, N, A);
      }
    }
  }

  // extract L and U
  for (int i = 1; i < N; ++i)
  {
    for (int j = i - 1; j >= 0; --j)
    {
      L[i * N + j] = A[i * N + j];
      A[i * N + j] = 0;
    }
  }
}