void diagonal_phase(int i, int B, int N, float *A);
void row_phase(int i, int j, int B, int N, float *A);
void col_phase(int i, int j, int B, int N, float *A);
void right_down_phase(int i, int j, int k, int B, int N, float *A);
void blocked_lu(int B, int N, float *A, float *L);