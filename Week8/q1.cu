#include <stdio.h>
#include <cuda.h>

#define ROWS 3
#define COLS 3

// i)Each thread computes one row
__global__ void addRowWise(int *A, int *B, int *C, int rows, int cols) {
    int row = threadIdx.x;
    if (row < rows) {
        for (int j = 0; j < cols; j++) {
            C[row * cols + j] = A[row * cols + j] + B[row * cols + j];
        }
    }
}

// ii)Each thread computes one column
__global__ void addColumnWise(int *A, int *B, int *C, int rows, int cols) {
    int col = threadIdx.x;
    if (col < cols) {
        for (int i = 0; i < rows; i++) {
            C[i * cols + col] = A[i * cols + col] + B[i * cols + col];
        }
    }
}

// iii)Each thread computes one element
__global__ void addElementWise(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
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

    printf("Matrix A:\n");
    printMatrix((int *)A, ROWS, COLS);
    printf("\nMatrix B:\n");
    printMatrix((int *)B, ROWS, COLS);

    addRowWise<<<1, ROWS>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nRow-wise Addition:\n");
    printMatrix((int *)C, ROWS, COLS);

    addColumnWise<<<1, COLS>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nColumn-wise Addition:\n");
    printMatrix((int *)C, ROWS, COLS);

    dim3 gridDim(ROWS);
    dim3 blockDim(COLS); // not necesary to do dim3, but easier to understand
    addElementWise<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nElement-wise Addition:\n");
    printMatrix((int *)C, ROWS, COLS);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
