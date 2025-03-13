#include <stdio.h>
#include <cuda.h>

#define ROWS 3
#define COLS 3
#define COMMON_DIM 3 //A: ROWS × COMMON_DIM, B: COMMON_DIM × COLS

// i)Each thread computes one row of the result matrix
__global__ void multiplyRowWise(int *A, int *B, int *C, int rows, int cols, int common) {
    int row = threadIdx.x;
    if (row < rows) {
        for (int j = 0; j < cols; j++) {
            int sum = 0;
            for (int k = 0; k < common; k++) {
                sum += A[row * common + k] * B[k * cols + j];
            }
            C[row * cols + j] = sum;
        }
    }
}

// ii)Each thread computes one column of the result matrix
__global__ void multiplyColumnWise(int *A, int *B, int *C, int rows, int cols, int common) {
    int col = threadIdx.x;
    if (col < cols) {
        for (int i = 0; i < rows; i++) {
            int sum = 0;
            for (int k = 0; k < common; k++) {
                sum += A[i * common + k] * B[k * cols + col];
            }
            C[i * cols + col] = sum;
        }
    }
}

// iii) Each thread computes one element of the result matrix
__global__ void multiplyElementWise(int *A, int *B, int *C, int rows, int cols, int common) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        int sum = 0;
        for (int k = 0; k < common; k++) {
            sum += A[row * common + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
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
    int A[ROWS][COMMON_DIM] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[COMMON_DIM][COLS] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[ROWS][COLS];

    int *d_A, *d_B, *d_C;
    size_t sizeA = ROWS * COMMON_DIM * sizeof(int);
    size_t sizeB = COMMON_DIM * COLS * sizeof(int);
    size_t sizeC = ROWS * COLS * sizeof(int);

    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    printf("Matrix A:\n");
    printMatrix((int *)A, ROWS, COMMON_DIM);
    printf("\nMatrix B:\n");
    printMatrix((int *)B, COMMON_DIM, COLS);

    multiplyRowWise<<<1, ROWS>>>(d_A, d_B, d_C, ROWS, COLS, COMMON_DIM);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nRow-wise Multiplication:\n");
    printMatrix((int *)C, ROWS, COLS);

    multiplyColumnWise<<<1, COLS>>>(d_A, d_B, d_C, ROWS, COLS, COMMON_DIM);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nColumn-wise Multiplication:\n");
    printMatrix((int *)C, ROWS, COLS);

    dim3 gridDim(ROWS);
    dim3 blockDim(COLS);
    multiplyElementWise<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS, COMMON_DIM);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nElement-wise Multiplication:\n");
    printMatrix((int *)C, ROWS, COLS);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
