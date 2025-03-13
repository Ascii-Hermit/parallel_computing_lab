#include <stdio.h>
#include <cuda.h>

#define ROWS 3
#define COLS 3

//row sums kernel
__global__ void computeRowSums(int *matrix, int *rowSums, int rows, int cols) {
    int row = threadIdx.x;
    if (row < rows) {
        int sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[row * cols + j];
        }
        rowSums[row] = sum;
    }
}

//column sums kernel
__global__ void computeColumnSums(int *matrix, int *colSums, int rows, int cols) {
    int col = threadIdx.x;
    if (col < cols) {
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            sum += matrix[i * cols + col];
        }
        colSums[col] = sum;
    }
}

__global__ void replaceElements(int *matrix, int *rowSums, int *colSums, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        if (matrix[index] % 2 == 0) {
            matrix[index] = rowSums[row];  // replace row sum
        } else {
            matrix[index] = colSums[col];  // replace column sum
        }
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
    int matrix[ROWS][COLS] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int rowSums[ROWS], colSums[COLS];

    int *d_matrix, *d_rowSums, *d_colSums;
    size_t sizeMatrix = ROWS * COLS * sizeof(int);
    size_t sizeRowSums = ROWS * sizeof(int);
    size_t sizeColSums = COLS * sizeof(int);

    cudaMalloc((void **)&d_matrix, sizeMatrix);
    cudaMalloc((void **)&d_rowSums, sizeRowSums);
    cudaMalloc((void **)&d_colSums, sizeColSums);

    cudaMemcpy(d_matrix, matrix, sizeMatrix, cudaMemcpyHostToDevice);

    printf("Original Matrix:\n");
    printMatrix((int *)matrix, ROWS, COLS);

    computeRowSums<<<1, ROWS>>>(d_matrix, d_rowSums, ROWS, COLS);
    cudaMemcpy(rowSums, d_rowSums, sizeRowSums, cudaMemcpyDeviceToHost);

    computeColumnSums<<<1, COLS>>>(d_matrix, d_colSums, ROWS, COLS);
    cudaMemcpy(colSums, d_colSums, sizeColSums, cudaMemcpyDeviceToHost);

    dim3 gridDim(ROWS);
    dim3 blockDim(COLS);
    replaceElements<<<gridDim, blockDim>>>(d_matrix, d_rowSums, d_colSums, ROWS, COLS);

    cudaMemcpy(matrix, d_matrix, sizeMatrix, cudaMemcpyDeviceToHost);

    printf("\nModified Matrix:\n");
    printMatrix((int *)matrix, ROWS, COLS);

    cudaFree(d_matrix);
    cudaFree(d_rowSums);
    cudaFree(d_colSums);

    return 0;
}
