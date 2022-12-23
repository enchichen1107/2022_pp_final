# 2022_pp_final parallel LU factorization
## Using openMP and cuda for parallelization
### To run sequential basic lu code
- cd to sequential file
- make basic
- ./basic N output_file_name
- N means the length of matrix, matrix size will be N*N, output_file_name is the filename which will contain the LU matrix output. ex: ./basic 10 .output_file
### To run sequential blocked lu code
- cd to sequential file
- make blocked
- ./blocked N B output_file_name
- N means the length of matrix, matrix size will be N*N, B means block factor, output_file_name is the filename which will contain the LU matrix output. ex: ./blocked 10 5 .output_file