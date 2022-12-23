#include "basic_lu.h"
void basic_lu(float *A, float *L, int N)
{
  for (int k = 0; k < N - 1; ++k)
  {
    for (int i = k + 1; i < N; ++i)
    {
      L[i * N + k] = A[i * N + k] / A[k * N + k];
    }
    for (int j = k; j < N; ++j)
    {
      for (int i = k; i < N; ++i)
      {
        A[i * N + j] = A[i * N + j] - L[i * N + k] * A[k * N + j];
      }
    }
  }
}