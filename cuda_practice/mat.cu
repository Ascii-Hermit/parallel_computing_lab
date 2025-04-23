__global__ void matrixMultiply2D(int *A, int *B, int *C, int rows, int cols, int common) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of C

    if (row < rows && col < cols) {
        int sum = 0;
        for (int k = 0; k < common; ++k) {
            sum += A[row * common + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}
#include <stdio.h>
#include <cuda_runtime.h>

#define ROWS 3
#define COLS 3
#define COMMON_DIM 3

__global__ void matrixMultiply2D(int *A, int *B, int *C, int rows, int cols, int common) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of C

    if (row < rows && col < cols) {
        int sum = 0;
        for (int k = 0; k < common; ++k) {
            sum += A[row * common + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

int main() {
    int A[ROWS][COMMON_DIM] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int B[COMMON_DIM][COLS] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };
    int C[ROWS][COLS];

    int sizeA = ROWS * COMMON_DIM * sizeof(int);
    int sizeB = COMMON_DIM * COLS * sizeof(int);
    int sizeC = ROWS * COLS * sizeof(int);

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);  // 16x16 threads
    dim3 gridDim((COLS + blockDim.x - 1) / blockDim.x, 
                 (ROWS + blockDim.y - 1) / blockDim.y);

    matrixMultiply2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS, COMMON_DIM);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("Result matrix:\n");
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
