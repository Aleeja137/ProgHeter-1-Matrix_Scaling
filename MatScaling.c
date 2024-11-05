/* Author: Alejandro Perez */
// Imports
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h> // For sleep
#include <string.h> // Memcpy

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

// Main
int main(int argc, char *argv[])
{
    // Variables
    // int i;
    int m, n, r, threads_blk; 
    float *mat, *matSeq, *matPar, *factors;

    m = default_M;
    n = default_N;
    r = default_R;
    threads_blk = default_THREADS_PER_BLOCK;

    // Check parameters
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) r = atoi(argv[3]);
    if (argc > 4) r = atoi(argv[3]);
    if (argc > 5) {fprintf(stderr, "Usage: %s nrows(m:4000) ncol(n:4000) repetitions(r:5) threads_per_block(threads_blk:32)\n", argv[0] ); exit(1);}

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
    matrix_scaling_seq(mat, matSeq, factors, r, m, n);

    // Allocate memory on CUDA device
    // Pass data to CUDA device memory
    // Calculate kernel call dimensions
    // Call kernel
    // Synchronize (necessary?)
    // Receive data from CUDA device
    // Free memory on CUDA device

    // Check results
    int equal = mat_equal(mat, mat, m*n, -1);
    if (equal) printf("Results are OK :)\n");
    else       printf("Results are NOT ok :'(\n");

    // Frees
    free(factors);
    free(mat);
    free(matSeq);
    free(matPar);
    return 0;
}