#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define ROWS 3
#define COLS 3
#define COMMON_DIM 3

__global__ void matrixMultiplyRowWise(int* A, int* B, int* C){
    int row = threadIdx.x;
    if(row<ROWS){
        for (int col = 0; col < COLS;col++){
            int sum = 0;
            for (int i = 0; i < COMMON_DIM;i++){
                sum += A[row*COMMON_DIM+i] * B[COLS*i+col];
            }
            C[row*ROWS+col] = sum;
        }
    }
}

__global__ void matrixMultiply(int* A, int* B, int* C){

    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row<ROWS && col<COLS){
        int sum = 0;
        for (int i = 0; i < COMMON_DIM;i++){
            sum += A[row*COMMON_DIM+i] * B[COLS*i+col];
        }
        C[row*ROWS+col] = sum;
    }
}

int main(){
    int A[ROWS][COMMON_DIM] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[COMMON_DIM][COLS] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[ROWS][COLS];

    int *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, sizeof(int) * ROWS * COMMON_DIM);
    cudaMalloc((void **)&d_B, sizeof(int) * COLS * COMMON_DIM);
    cudaMalloc((void **)&d_C, sizeof(int) * COLS * ROWS);

    cudaMemcpy(d_A, A, sizeof(int) * ROWS * COMMON_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * COLS * COMMON_DIM, cudaMemcpyHostToDevice);

    dim3 gridDim(ROWS);
    dim3 blockDim(COLS);
    matrixMultiplyRowWise<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, sizeof(int)*ROWS*COLS, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ROWS;i++){
        for (int j = 0; j < COLS;j++){
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }
    return 0;
}