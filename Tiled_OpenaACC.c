%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <openacc.h>

#define TILE_SIZE 16

// Function for tiled matrix multiplication
void matrix_multiply_tiled(float *matrixA, float *matrixB, float *resultMatrix, int M, int N, int K) {
    #pragma acc parallel loop collapse(2) present(matrixA[0:M*N], matrixB[0:K*N], resultMatrix[0:M*K]) independent
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < K; j += TILE_SIZE) {
            for (int ii = i; ii < i + TILE_SIZE && ii < M; ii++) {
                for (int jj = j; jj < j + TILE_SIZE && jj < K; jj++) {
                    float sum = 0;
                    for (int k = 0; k < N; k++) {
                        sum += matrixA[ii * N + k] * matrixB[k * K + jj];
                    }
                    resultMatrix[ii * K + jj] = sum;
                }
            }
        }
    }
}

double getElapsedTime(struct timeval start, struct timeval stop) {
    return (double)(stop.tv_sec - start.tv_sec) * 1000.0 +
           (double)(stop.tv_usec - start.tv_usec) / 1000.0;
}

int main() {
    int M = 100;
    int N = 50;
    int K = 50;
    float *host_matrixA, *host_matrixB, *host_resultMatrix;

    // Allocate Memory 
    host_matrixA = (float*) malloc(M * N * sizeof(float));
    host_matrixB = (float*) malloc(N * K * sizeof(float));
    host_resultMatrix = (float*) malloc(M * K * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        host_matrixA[i] = rand() % 10;
    }
    for (int i = 0; i < N * K; i++) {
        host_matrixB[i] = rand() % 10;
    }

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    // Matrix multiplication using tiling
    matrix_multiply_tiled(host_matrixA, host_matrixB, host_resultMatrix, M, N, K);

    gettimeofday(&stop, NULL);
    double elapsed_time = getElapsedTime(start, stop);

    
    printf("The execution time for Tiled Openacc approach is: %.2f ms\n", elapsed_time);

    // Free Memory
    free(host_matrixA);
    free(host_matrixB);
    free(host_resultMatrix);

    return 0;
}
