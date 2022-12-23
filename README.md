# 2022_pp_final parallel LU factorization
## Using openMP and cuda for parallelization
### To run sequential code
- cd to project file
- g++ sequential.cc -o sequential
- ./sequential N output_file_name
- N means the length of matrix, matrix size will be N*N, output_file_name is the filename which will contain the LU matrix output. ex: ./sequential 10 output_file