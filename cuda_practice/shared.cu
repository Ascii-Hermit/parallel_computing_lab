#include <stdio.h>
#include <cuda_runtime.h>

#define ROWS 3
#define COLS 3

__global__ void matrixAddShared(int *A, int *B, int *C, int rows, int cols) {
    __shared__ int tileA[ROWS][COLS];
    __shared__ int tileB[ROWS][COLS];

    int row = threadIdx.y;
    int col = threadIdx.x;

    // Load data into shared memory
    if (row < rows && col < cols) {
        tileA[row][col] = A[row * cols + col];
        tileB[row][col] = B[row * cols + col];

        // Synchronize to make sure the tiles are fully loaded
        __syncthreads();

        // Perform addition
        C[row * cols + col] = tileA[row][col] + tileB[row][col];
    }
}

int main() {
    int A[ROWS][COLS] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[ROWS][COLS] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[ROWS][COLS];

    int *d_A, *d_B, *d_C;
    size_t size = ROWS * COLS * sizeof(int);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(COLS, ROWS);
    dim3 gridDim(1, 1);

    matrixAddShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Resultant Matrix (A + B):\n");
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
