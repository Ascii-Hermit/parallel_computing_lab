#include <stdio.h>
#include <cuda.h>

#define ROWS 3
#define COLS 3

// factorial kernel
__global__ void computeFactorial(int *matrix, int *fact, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols && row != col) {
        int index = (row < col) ? ((col * (col - 1)) / 2 + row) : ((row * (row - 1)) / 2 + col);
        int num = matrix[row * cols + col];

        int factorial = 1;
        for (int i = 1; i <= num; i++) {
            factorial *= i;
        }
        fact[index] = factorial;
    }
}

// digit sum kernel
__global__ void computeDigitSum(int *matrix, int *digitSums, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols && row != col) {
        int index = (row < col) ? ((col * (col - 1)) / 2 + row) : ((row * (row - 1)) / 2 + col);
        int num = matrix[row * cols + col];

        while (num >= 10) {
            int sum = 0;
            while (num > 0) {
                sum += num % 10;
                num /= 10;
            }
            num = sum;
        }
        digitSums[index] = num;
    }
}

__global__ void replaceElements(int *matrix, int *fact, int *digSum, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        if (row == col) {
            matrix[index] = 0; 
        } else {
            int factIndex = (row < col) ? ((col * (col - 1)) / 2 + row) : ((row * (row - 1)) / 2 + col);
            matrix[index] = (row < col) ? fact[factIndex] : digSum[factIndex];
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
    int eleNums = ((ROWS * COLS) - ROWS) / 2;  // this is the formula to calculate the number of elements above and belw
    int fact[eleNums], digSum[eleNums];

    int *d_matrix, *d_fact, *d_digSum;
    size_t sizeMatrix = ROWS * COLS * sizeof(int);
    size_t sizeFact = eleNums * sizeof(int);
    size_t sizeDigSum = eleNums * sizeof(int);

    cudaMalloc((void **)&d_matrix, sizeMatrix);
    cudaMalloc((void **)&d_fact, sizeFact);
    cudaMalloc((void **)&d_digSum, sizeDigSum);

    cudaMemcpy(d_matrix, matrix, sizeMatrix, cudaMemcpyHostToDevice);

    printf("Original Matrix:\n");
    printMatrix((int *)matrix, ROWS, COLS);

    computeFactorial<<<ROWS, COLS>>>(d_matrix, d_fact, ROWS, COLS);
    computeDigitSum<<<ROWS, COLS>>>(d_matrix, d_digSum, ROWS, COLS);

    cudaMemcpy(fact, d_fact, sizeFact, cudaMemcpyDeviceToHost);
    cudaMemcpy(digSum, d_digSum, sizeDigSum, cudaMemcpyDeviceToHost);

    replaceElements<<<ROWS, COLS>>>(d_matrix, d_fact, d_digSum, ROWS, COLS);

    cudaMemcpy(matrix, d_matrix, sizeMatrix, cudaMemcpyDeviceToHost);

    printf("\nModified Matrix:\n");
    printMatrix((int *)matrix, ROWS, COLS);

    cudaFree(d_matrix);
    cudaFree(d_fact);
    cudaFree(d_digSum);

    return 0;
}
