# 2022_pp_final parallel LU factorization
## Using openMP and cuda for parallelization
### To run sequential lu basic code
- cd to sequential file
- make lu_basic
- ./lu_basic N output_file_name
- N means the length of matrix, matrix size will be N*N, output_file_name is the filename which will contain the LU matrix output. ex: ./basic 10 .output_file
### To run sequential lu blocked code
- cd to sequential file
- make lu_blocked
- ./lu_blocked N B output_file_name
- N means the length of matrix, matrix size will be N*N, B means block factor, output_file_name is the filename which will contain the LU matrix output. ex: ./blocked 10 5 .output_file
### To run lu cuda code
- cd to cuda file
- make lu_cuda
- ./lu_cuda N output_file_name
- N means the length of matrix, matrix size will be N*N, block width will be set to min(N, 1024),  output_file_name is the filename which will contain the LU matrix output. ex: ./lu_cuda 10 .output_file