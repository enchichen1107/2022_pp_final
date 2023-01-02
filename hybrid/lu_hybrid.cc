#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <ctime>
#include <chrono>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>

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
void U_print (float *M, int dim)
{
   int i, j;
   float z = 0;
   for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
         if (j >= i) {
            printf("% *.*f ", 4, 4, M[i * dim + j]);
         } else {
            printf("% *.*f ", 4, 4, z);
         }
      }
      printf("\n");
   }
}

void L_print (float *M, int dim)
{
   int i, j;
   float z = 0, u = 1;
   for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
         if (j > i) {
            printf("% *.*f ", 4, 4, z);
         } else if (i == j) {
            printf("% *.*f ", 4, 4, u);
         } else {
            printf("% *.*f ", 4, 4, M[i * dim + j]);
         }
      }
      printf("\n");
   }
}
void forw_elim(float **origin, float *master_row, size_t dim)
{
   if (**origin == 0)
      return;
   float k = **origin / master_row[0];
   #pragma omp for schedule(static)
   for (int i = 1; i < dim; i++) {
      (*origin)[i] = (*origin)[i] - k * master_row[i];
   }
   **origin = k;
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
//    cpu_set_t cpu_set;
//    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
//    int ncpus = CPU_COUNT(&cpu_set);
    //printf("%d cpus available\n", ncpus);
    
  // generate matrix
  // srand((unsigned)time(NULL));
  float *A = (float *)malloc(N * N * sizeof(float));
  //float *L = (float *)malloc(N * N * sizeof(float));
  
      for (int i = 0; i < N; ++i)
      {
        for (int j = 0; j < N; ++j)
        {
          A[i * N + j] = 1 + (rand() % 10000);
          //L[i * N + j] = 0;
        }
        // ensure diagonally dominant
        A[i * N + i] = A[i * N + i] + 10000 * N;
      }

      /* initialize mpi*/
    int rank ,size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {
      // print matrix before lu factorization
      if (N < 11)
      {
        printf("the matrix before lu factorization is\n");
        print_matrix(A, N);
	printf("\n");
      }
  }

  // lu factorization
	
    int diag = 0;
    for (int k = 0; k < N - 1; ++k) {
        float *diag_row = &A[k * N + k];
        for (int i = k + 1; i < N; ++i) {
            if (i % size == rank) {
                float *save_a = &A[i * N + k]; //lower pivot
                //float *save_l = &L[i * N + k];
                #pragma omp parallel
                {
                forw_elim(&save_a, diag_row, N-k);
                }
            }
        }
        for (int i = k+1; i < N; ++i) {
            float *save = &A[i * N + k];
            MPI_Bcast(save, N-k, MPI_FLOAT, i%size, MPI_COMM_WORLD);
        }
    }
	int mx_size = N;
  
    if (rank ==0) {
      if (N < 11)
      {
        printf("the lu factorization outcome is\n");        
	/*printf("U is\n");
        print_matrix(A, N);
        printf("L is\n");
        print_matrix(L, N);*/
        printf("\n[L]\n");
        L_print(A, N);
        printf("\n[U]\n");
        U_print(A, N);
      }
      // write result to output file
      ofstream out_file(out_filename);
      float zero = 0.0;
      float one = 1.0;
      for (int i = 0; i < N; ++i) {
	for (int j = 0; j < N; ++j) {
	    if (j >= i) {
        	out_file.write((char *)&A[i*N+j], sizeof(float));
	    }
	    else { 
		out_file.write((char*)&zero, sizeof(float));
	    }
	}
      }
      for (int i = 0; i < N; ++i) {
	for (int j = 0; j < N; ++j) {
	    if (j > i) {
		out_file.write((char*)&zero, sizeof(float));
	    }
	    else if (j == i) {
		out_file.write((char*)&one, sizeof(float));
	    }
	    else {
       		 out_file.write((char *)&A[i*N+j], sizeof(float));
	    }
	}
      }
      out_file.close();
        // calculate total spent time
        auto total_endtime = duration_cast<ms>(system_clock::now().time_since_epoch()).count();
        printf("total time spent for mpi lu %ld ms\n", (total_endtime - total_starttime));
    }
    
  free(A);
  //free(L);
  MPI_Finalize();

  
}

 
