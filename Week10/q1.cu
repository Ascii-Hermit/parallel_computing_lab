#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16 

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    for (int k = 0; k < width; ++k) {
        if (row < width && k < width && col < width) {
            Cvalue += A[row * width + k] * B[k * width + col];
        }
    }

    if (row < width && col < width) {
        C[row * width + col] = Cvalue;
    }
}

void matrixMultiply(float *A, float *B, float *C, int width) {
    float *d_A, *d_B, *d_C;

    size_t size = width * width * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);  
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);

    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
    int main() {
    int width = 4;  

    float *A = (float*)malloc(width * width * sizeof(float));
    float *B = (float*)malloc(width * width * sizeof(float));
    float *C = (float*)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; i++) {
        A[i] = rand() % 10;  
        B[i] = rand() % 10; 
    }
    printMatrix(A,width,width);
    printf("\n");
    printMatrix(B,width,width);
    
    matrixMultiply(A, B, C, width);

    printf("\n");
    printMatrix(C,width,width);
   
    free(A);
    free(B);
    free(C);

    return 0;
}
