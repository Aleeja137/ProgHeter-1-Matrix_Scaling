/* Author: Alejandro Perez */
// Imports
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h> // For sleep
#include <string.h> // Memcpy
#include <cuda_runtime.h>

// Defines 
#define default_M 4000
#define default_N 4000
#define default_R 5
#define default_THREADS_PER_BLOCK 32


// Populate matrix
void populate_matrix(float *M, int n){
    int i;
    for (i=0; i<n; i++) M[i] = (float)rand() / (float) RAND_MAX;
}

// Print Matrix - For testing
void print_matrix(float *M, int m, int n){
    int i,j;
    for (i=0; i<m; i++)
    {
        printf("[");
        for (j=0; j<n-1; j++)
            printf("%.2f ", M[i*n+j]);
        printf("%.2f]\n",M[i*n+n-1]);
    }
}

// Check results
int mat_equal(float *A, float *B, int size, float threshold){
    int i;
    float th = threshold;  
    if (threshold < 0) th = 0.005; // Default threshold value
    for (i=0; i<size; i++) if (fabs(A[i]-B[i]) > th) return 0;
    return 1;
}

void matrix_scaling_seq(float *mat, float *rMat, float *factors, int r, int m, int n){
    int rep, i;
    float factor;

    for (i=0; i<m*n; i++) rMat[i] = factors[0]*mat[i];

    for (rep=1; rep<r; rep++)
    {
        factor = factors[rep];
        for (i=0; i<m*n; i++) rMat[i] *= factor;
    }
}

// Define CUDA kernel
__global__ void matrix_scaling_cuda(float *mat, float *rMat, float *factors, int r, int m, int n){

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int rep;

    if (global_thread_id < m*n)
    {
        rMat[global_thread_id] = mat[global_thread_id] * factors[0];
        for (rep=1; rep<r; rep++) rMat[global_thread_id] *= factors[rep];
    }

}

// Main
int main(int argc, char *argv[])
{
    // Variables
    // int i;
    int m, n, r, threads_blk; 
    float *mat, *matSeq, *matPar, *factors;
    float *mat_cuda, *matResults_cuda, *factors_cuda;
    cudaEvent_t start_seq, start_par, start_par_tot, end_seq, end_par;
    cudaEventCreate(&start_seq); cudaEventCreate(&end_seq);
    cudaEventCreate(&start_par); cudaEventCreate(&end_par);
    cudaEventCreate(&start_par_tot);

    m = default_M;
    n = default_N;
    r = default_R;
    threads_blk = default_THREADS_PER_BLOCK;

    // Check parameters
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) r = atoi(argv[3]);
    if (argc > 4) threads_blk = atoi(argv[4]);
    if (argc > 5) {fprintf(stderr, "Usage: %s nrows(m:4000) ncol(n:4000) repetitions(r:5) threads_per_block(threads_blk:32)\n", argv[0] ); exit(1);}
    if (threads_blk != 32 && threads_blk != 64 && threads_blk != 128) {fprintf(stderr, "Correct threads_blk: {32,64,128}\n"); exit(1);}

    printf("\n ---Program start---\n\n Configuration chosen --> m: %d, n: %d, r: %d, threads_blk: %d\n",m,n,r,threads_blk);
    srand(42); // Meaning of life

    // Allocate memory for mattrices and factors
    factors = (float *) malloc(r*sizeof(float));
    mat     = (float *) malloc(m*n*sizeof(float));
    matSeq  = (float *) malloc(m*n*sizeof(float));
    matPar  = (float *) malloc(m*n*sizeof(float));

    // Populate matrix M 
    populate_matrix(mat, m*n);
    
    // Same function works for generating factors
    populate_matrix(factors, r);

    // Get sequential result for later correctness analysis
    printf("Starting sequential matrix scaling ... \n");
    cudaEventRecord(start_seq);
    matrix_scaling_seq(mat, matSeq, factors, r, m, n);
    cudaEventRecord(end_seq);

    // Allocate memory on CUDA device
    cudaEventRecord(start_par_tot);
    unsigned int numBytes = m*n*sizeof(float);
    cudaMalloc((void **) &mat_cuda , numBytes);
    cudaMalloc((void **) &matResults_cuda , numBytes);
    cudaMalloc((void **) &factors_cuda, r*sizeof(float));

    // Pass data to CUDA device memory
    cudaMemcpy(mat_cuda,mat,numBytes,cudaMemcpyHostToDevice);
    cudaMemset(matResults_cuda, 0, numBytes);
    cudaMemcpy(factors_cuda, factors,r*sizeof(float),cudaMemcpyHostToDevice);

    // Calculate kernel call dimensions
    unsigned int tasks = m*n;
    dim3 dimBlock(threads_blk);                     // threads_blk threads
    dim3 dimGrid((tasks+dimBlock.x-1)/dimBlock.x);  // tasks/threads_blk blocks, rounded if not divisible
    
    // Call kernel
    printf("Starting parallel matrix scaling ... \n");
    cudaEventRecord(start_par);
    matrix_scaling_cuda<<<dimGrid, dimBlock>>>(mat_cuda, matResults_cuda, factors_cuda, r, m, n);
    cudaEventRecord(end_par);

    // Receive data from CUDA device
    cudaMemcpy(matPar, matResults_cuda, numBytes, cudaMemcpyDeviceToHost); 

    // Free memory on CUDA device
    cudaFree(mat_cuda);
    cudaFree(matResults_cuda);
    cudaFree(factors_cuda);

    // Check results
    int equal = mat_equal(matSeq, matPar, m*n, -1);
    if (equal) printf("Results are OK :)\n");
    else       printf("Results are NOT ok :'(\n");

    float time_seq, time_par, time_par_tot;
    cudaEventElapsedTime(&time_seq, start_seq, end_seq);
    cudaEventElapsedTime(&time_par, start_par, end_par);
    cudaEventElapsedTime(&time_par_tot, start_par_tot, end_par);

    printf("Time for sequential:                       %.3f ms\n",time_seq);
    printf("Time for parallel (computation only):      %.3f ms\n",time_par);
    printf("Time for parallel (memory management too): %.3f ms\n",time_par_tot);

    // Frees
    cudaEventDestroy(start_seq); cudaEventDestroy(end_seq);
    cudaEventDestroy(start_par); cudaEventDestroy(end_par);
    cudaEventDestroy(start_par_tot);
    free(factors);
    free(mat);
    free(matSeq);
    free(matPar);
    return 0;
}