#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <ctime>
#include <chrono>
#include <sys/time.h>
#include <omp.h>
#include "mpi.h"

using namespace std;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

void print_matrix(float *A, int N)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("%.4f ", A[i * N + j]);
    }
    printf("\n");
  }
}

void forw_elim(float **lower_pivot, float *upper_master_row, size_t dim)
{
   if (**lower_pivot == 0)
      return;

   float k = **lower_pivot / upper_master_row[0];

   int i;
   for (i = 1; i < dim; i++) {
       upper_master_row[i] = upper_master_row[i] - k * upper_master_row[i];
   }
   **lower_pivot = k;
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    fprintf(stderr, "must provide exactly 2 arguments N output_filename\n");
    return 1;
  }
  typedef std::chrono::milliseconds ms;
  auto total_starttime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();

  // parsing argument
  int N = atoi(argv[1]);
  char *out_filename = argv[2];

  /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    //printf("%d cpus available\n", ncpus);
    
    /* initialize mpi*/
    int rank ,size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Status status;

  // generate matrix
  // srand((unsigned)time(NULL));
  float *A = (float *)malloc(N * N * sizeof(float));
  float *L = (float *)malloc(N * N * sizeof(float));
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      A[i * N + j] = 1 + (rand() % 10000);
      L[i * N + j] = 0;
    }
    // ensure diagonally dominant
    A[i * N + i] = A[i * N + i] + 10000 * N;
  }

  // print matrix before lu factorization
  if (N < 11)
  {
    printf("the matrix before lu factorization is\n");
    print_matrix(A, N);
  }
    
  // basic lu factorization

        for (int k = 0; k < N - 1; ++k) {
            float *diag_row = &A[k * N + k];
            for (int i = k + 1; i < N; ++i) {
                if (i % size == rank) {
                    float *save = &L[i * N + k]; //lower pivot
                    forw_elim(&save, diag_row, N-k);
                }
            }
            for (int i = k+1; i < N; ++i) {
                float *save = &A[i * N + k];
                MPI_Bcast(save, N-k, MPI_FLOAT, i%size, MPI_COMM_WORLD);
            }
        }

    if (rank ==0) {
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
        print_matrix(A, N);
        printf("L is\n");
        print_matrix(L, N);
      }

      // write result to output file
      ofstream out_file(out_filename);
      for (int i = 0; i < N * N; ++i)
      {
        out_file.write((char *)&A[i], sizeof(float));
      }
      for (int i = 0; i < N * N; ++i)
      {
        out_file.write((char *)&L[i], sizeof(float));
      }
      out_file.close();
    }
    
  free(A);
  free(L);
  MPI_Finalize();

  // calculate total spent time
  auto total_endtime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();
  printf("total time spent for basic lu %ld ms\n", (total_endtime - total_starttime));
}

