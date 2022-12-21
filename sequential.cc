#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
using namespace std;

void print_matrix(float *A, int N)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("%.2f ", A[i * N + j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    fprintf(stderr, "must provide exactly 1 argument for matrix size!\n");
    return 1;
  }

  // parsing argument
  int N = atoi(argv[1]);

  // generate matrix
  // srand((unsigned)time(NULL));
  float *A = (float *)malloc(N * N * sizeof(float));
  float *L = (float *)malloc(N * N * sizeof(float));
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i * N + j] = 1 + (rand() % 100);
      L[i * N + j] = 0;
    }
  }
  printf("the matrix before lu factorization is\n");
  print_matrix(A, N);

  // lu factorization
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
    L[k * N + k] = 1;
  }

  // print outcome
  printf("the lu factorization outcome is\n");
  printf("U is\n");
  print_matrix(A, N);
  printf("L is\n");
  print_matrix(L, N);
}