#include <iostream>

#define TILE_SIZE 32

__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N - 1) / TILE_SIZE + 1; ++t) {
        if (row < M && t * TILE_SIZE + tx < N) 
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else 
            As[ty][tx] = 0.0f;

        if (col < K && t * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * K + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += As[ty][i] * Bs[i][tx];
        
        __syncthreads();
    }
    
    if (row < M && col < K) 
        C[row * K + col] = sum;
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    int size_A = M * N * sizeof(float);
    int size_B = N * K * sizeof(float);
    int size_C = M * K * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C = (float *)malloc(size_C);

    // Initialize host arrays
    for (int i = 0; i < M * N; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < N * K; ++i) h_B[i] = 2.0f;

    // Allocate device memory
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Copy host data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
