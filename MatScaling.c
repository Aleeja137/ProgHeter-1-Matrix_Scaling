/* Author: Alejandro Perez */
// Imports
#include <stdio.h>¡
#include <stdlib.h>
#include <math.h>¡
#include <unistd.h> // For sleep
#include <string.h> // Memcpy

// Defines 
#if !defined(N)
#define N 12
#endif

// Populate matrix
void populate_matrix(int *M, int n){
    int ind;
    // for (ind = 0; ind < n*n; ind++) M[ind] = random();
    for (ind = 0; ind < n*n; ind++) M[ind] = ind; // For simple checks
}

// Print Matrix
void print_matrix(int *M, int n){
    int i,j, size= n;
    for (i=0; i<size; i++)
    {
        printf("[");
        for (j=0; j<size-1; j++)
            printf("%d ", M[i*n+j]);
        printf("%d]\n",M[i*n+size-1]);
    }
}

// Check results
int mat_equal(int *A, int *B, int n){
    int i;
    for (i=0; i<n*n; i++)
        if (A[i]!=B[i])
            return 0;
    return 1;
}

// Main
int main(int argc, char *argv[])
{
    // Variables
    int *M, *sM, *sM_res, *Mseq, *Mpar; 
    int i, j, nproc, pid, grid_pid, grid_coords[2];
    int block_dim, n, sN, equal;
    int left_pid, right_pid, up_pid, down_pid;

    // If parameter --> error
    if (argc > 1) {fprintf(stderr, "Usage: %s\n", argv[0] ); exit(1);} 
    n = N;


    // Allocate memory for M, Mseq and Mpar 
    M    = (int *) malloc(n*n*sizeof(int));
    Mseq = (int *) malloc(n*n*sizeof(int));
    Mpar = (int *) malloc(n*n*sizeof(int));


    // Calculate submatrix dimensions
    block_dim = sqrt(nproc);
    sN = n / block_dim;

    // Allocate memory for sub matrix sM
    sM     = (int *) malloc(sN*sN*sizeof(int));
    sM_res = (int *) malloc(sN*sN*sizeof(int));

    
    // Populate matrix M 
    populate_matrix(M, n);

    // Check results
    equal = mat_equal(Mseq, Mpar, n);
    if (equal) printf("Result is OK! :)\n");
    else printf("Results are NOT OK :(\n");

    free(M);
    free(Mseq);
    free(Mpar);

    // Frees 
    free(sM);
    free(sM_res);
    return 0;
}