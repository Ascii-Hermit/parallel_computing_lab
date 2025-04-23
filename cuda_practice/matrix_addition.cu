#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define ROWS 3
#define COLS 3

__global__ void addMatrixRowWise(int* A, int* B, int* C){
    int row = threadIdx.x;
    for(int i = 0;i<COLS;i++){
        C[row*ROWS+i] = A[row*ROWS+i]+B[row*ROWS+i];
    }
}

__global__ void addMatrixColumnWise(int* A, int* B, int* C){
    int col = threadIdx.x;
    for(int i = 0;i<ROWS;i++){
        C[COLS*i+col] = A[COLS*i+col]+B[COLS*i+col];
    }
}

__global__ void addMatrix(int* A, int* B, int* C){
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row<ROWS && col< COLS){
        C[row*ROWS+col] = A[row*ROWS+col]+B[row*ROWS+col];
    }
}

int main(){
    int A[ROWS][COLS] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[ROWS][COLS] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[ROWS][COLS];

    int *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, sizeof(int) * ROWS * COLS);
    cudaMalloc((void **)&d_B, sizeof(int) * ROWS * COLS);
    cudaMalloc((void **)&d_C, sizeof(int) * ROWS * COLS);

    cudaMemcpy(d_A, A, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridDim(ROWS);
    dim3 blockDim(COLS);

    addMatrixColumnWise<<<gridDim, blockDim>>>(d_A,d_B,d_C);

    cudaMemcpy(C, d_C, sizeof(int) * ROWS * COLS, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ROWS;i++){
        for (int j = 0; j < COLS;j++){
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}